import logging
import numpy as np
from tqdm import tqdm
import torch
import asyncio
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from nets.model_setting import rate_model_setting, bptt_model_setting
from spikingjelly.activation_based import functional
from utils.inc_net import get_convnet
from nets.layer import CustomEvaluator, CustomUpdataEvaluator
EPSILON = 1e-8

init_epoch = 150
init_milestones = [60, 120, 170]
init_lr_decay = 0.1


epochs = 150
lrate = 0.1
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 64
weight_decay = 2e-4
num_workers = 12
T = 2
class UpdataEvaluator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.evaluator = evaluator
        # self.model_type = model_type
        # self.model_time_step = model_time_step

    def forward(self, avg_fr: torch.Tensor, old_avg_fr: torch.Tensor, targets: torch.Tensor, known_classes):
        loss_clf = F.cross_entropy(avg_fr, targets)
        loss_kd = _KD_loss(
            avg_fr[:, : known_classes],
            old_avg_fr,
            T,
        )
        loss = loss_clf + loss_kd
        return loss


class iCaRL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = get_convnet(args)
        self.step_mode = args["step_mode"]
        self.T_max = args["T"]
        # bptt_model_setting(self._network, time_step=self.T_max, step_mode=self.step_mode)
        self._class_means = None
        self.lr = args["lr"]
        self.wd = args["wd"]
        self.swap_ratio = args["swap_ratio"]
        self.dataset = "img"
        if args["dataset"] == 'urbansoundswap' or args["dataset"] == 'dsadsswap':
            self.config = args["dvs_config"]
            self.data_ratio = 0.5
        if args["dataset"] == 'ncaltech101swap' or args["dataset"] == 'dvs128swap':
            self.config = args["dvs_config"]
            self.T_max = 8
            self.dataset = 'dvs'
            self.dynamic = False
            self.data_ratio = 0.5
        self.rate_flag = args["rate_flag"]
        self.dynamic = args["dynamic"]
        if self.rate_flag:
            rate_model_setting(self._network, time_step=self.T_max, step_mode=self.step_mode)
        else:
            bptt_model_setting(self._network, time_step=self.T_max, step_mode=self.step_mode)
        if "lamb" in args:
            self.lamb = args["lamb"]
            self.means = args["means"]
        else:
            self.lamb = 0
            self.means = 1.0
        init_eva = torch.nn.CrossEntropyLoss()
        up_eva = UpdataEvaluator()
        self.init_eva = CustomEvaluator(init_eva, "rate" if self.rate_flag else "bptt", args["T"])
        self.up_eva = CustomUpdataEvaluator(up_eva, "rate" if self.rate_flag else "bptt", args["T"])
    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.updata_classifier(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_train_dataset(
            np.arange(self._known_classes, self._total_classes),
            appendent=self._get_memory(),
            swap_ratio=self.swap_ratio
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        test_dataset = data_manager.get_test_dataset(
            np.arange(0, self._total_classes)
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.lr,
                weight_decay=self.wd,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                targets = targets.long()
                functional.reset_net(self._network)
                if self.dynamic:
                    self.T = np.random.randint(1, self.T_max + 1)
                    for name, module in self._network.named_modules():
                        setattr(module, 'time_step', self.T)
                else:
                    self.T = self.T_max
                if self.step_mode == 's':
                    if self.dataset == 'dvs':
                        inputs = inputs.permute(1, 0, 2, 3, 4)
                    # in_data = inputs.permute(1, 0, 2, 3, 4)
                    out_spikes = []
                    for t in range(self.T):
                        if self.dataset == 'dvs':
                            out = self._network(inputs[t])
                        else:
                            out = self._network(inputs)
                        out_spikes.append(out)
                    output = torch.stack(out_spikes, dim=0)
                    avg_fr = output if self.dynamic else output.mean(dim=0)
                else:
                    if self.dataset == 'dvs':
                        in_data = inputs.permute(1, 0, 2, 3, 4)
                    # in_data = inputs.permute(1, 0, 2, 3, 4)
                    else:
                        in_data, _ = torch.broadcast_tensors(inputs, torch.zeros((self.T,) + inputs.shape))
                    in_data = in_data.reshape(-1, *in_data.shape[2:])
                    output = self._network(in_data)
                    avg_fr = output if self.dynamic else output.mean(dim=0)

                if self.dynamic:
                    loss = self.init_eva(avg_fr, targets, T = self.T, lamb = self.lamb, means = self.means)
                else:
                    loss = self.init_eva(avg_fr, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                if  self.dynamic:
                    _, preds = torch.max(avg_fr.mean(dim=0), dim=1)
                else:
                    _, preds = torch.max(avg_fr, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (idxs, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                targets = targets.long()
                functional.reset_net(self._network)
                functional.reset_net(self._old_network)
                if self.dynamic:
                    self.T = np.random.randint(1, self.T_max + 1)
                    for name, module in self._network.named_modules():
                        setattr(module, 'time_step', self.T)
                    for name, module in self._old_network.named_modules():
                        setattr(module, 'time_step', self.T)
                else:
                    self.T = self.T_max
                if self.step_mode == 's':
                    if self.dataset == 'dvs':
                        inputs = inputs.permute(1, 0, 2, 3, 4)
                    out_spikes = []
                    old_out_spikes = []
                    for t in range(self.T):
                        if self.dataset == 'dvs':
                            out = self._network(inputs[t])
                            old_out = self._old_network(inputs[t])
                        else:
                            out = self._network(inputs)
                            old_out = self._old_network(inputs)
                        out_spikes.append(out)
                        old_out_spikes.append(old_out)
                    output = torch.stack(out_spikes, dim=0)
                    old_output = torch.stack(old_out_spikes, dim=0)
                    avg_fr = output if self.dynamic else output.mean(dim=0)
                    old_avg_fr = old_output if self.dynamic else old_output.mean(dim=0)
                else:
                    if self.dataset == 'dvs':
                        in_data = inputs.permute(1, 0, 2, 3, 4)
                    # in_data = inputs.permute(1, 0, 2, 3, 4)
                    else:
                        in_data, _ = torch.broadcast_tensors(inputs, torch.zeros((self.T,) + inputs.shape))
                    in_data = in_data.reshape(-1, *in_data.shape[2:])
                    output = self._network(in_data)
                    old_output = self._old_network(in_data)

                    avg_fr = output if self.dynamic else output.mean(dim=0)
                    old_avg_fr = old_output if self.dynamic else old_output.mean(dim=0)
                    # avg_fr = output.mean(dim=0)
                    # old_avg_fr = old_output.mean(dim=0)

                if self.swap_ratio > 0:
                    asyncio.run(train_loader.dataset._swap(idxs, targets))
                if self.dynamic:
                    loss = self.up_eva(avg_fr, old_avg_fr, targets, self._known_classes, T = self.T, lamb = self.lamb, means = self.means)
                else:
                    loss = self.up_eva(avg_fr, old_avg_fr, targets, self._known_classes)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                if  self.dynamic:
                    _, preds = torch.max(avg_fr.mean(dim=0), dim=1)
                else:
                    _, preds = torch.max(avg_fr, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)


    def eval_task(self, save_conf=False):
        y_pred, y_true = self._eval_snn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        nme_accy = None
        # if save_conf:
        #     _pred = y_pred.T[0]
        #     _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
        #     _target_path = os.path.join(self.args['logfilename'], "target.npy")
        #     np.save(_pred_path, _pred)
        #     np.save(_target_path, y_true)

        #     _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
        #     os.makedirs(_save_dir, exist_ok=True)
        #     _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
        #     with open(_save_path, "a+") as f:
        #         f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")

        return cnn_accy, nme_accy
def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
