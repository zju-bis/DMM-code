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
from utils.toolkit import target2onehot, tensor2numpy
from nets.model_setting import bptt_model_setting
from spikingjelly.activation_based import functional
from utils.inc_net import get_convnet, split_params
from nets.model_setting import rate_model_setting, bptt_model_setting
from spikingjelly.activation_based import functional
from utils.inc_net import get_convnet
from nets.layer import CustomEvaluator
EPSILON = 1e-8


init_epoch = 150
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 150
lrate = 0.1
milestones = [30, 50]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2


class Replay(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = get_convnet(args)
        self.step_mode = args["step_mode"]
        self.T_max = args["T"]
        self._class_means = None
        self.lr = args["lr"]
        self.wd = args["wd"]
        self.swap_ratio = args["swap_ratio"]
        self.dataset ='img'
        # self.replace = args["replace"]
        self.epoach = args["epoachs"]
        if args["dataset"] == 'ncaltech101swap':
            self.T_max = 8
            self.dataset = 'dvs'
        if args["dataset"] == 'dvs128swap':
            self.T_max = 8
            self.dataset ='dvs'
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
        self.evaluator = CustomEvaluator(init_eva, "rate" if self.rate_flag else "bptt", args["T"])
    def after_task(self):
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

        # Loader
        train_dataset = data_manager.get_train_dataset(
            np.arange(self._known_classes, self._total_classes),
            appendent=self._get_memory(),
            swap_ratio=self.swap_ratio
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_test_dataset(
            np.arange(0, self._total_classes)
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        # if self.replace == 'min':
        #     self.build_rehearsal_memory(data_manager, self.samples_per_class)
        # elif self.replace == 'max':
        #     self.build_rehearsal_memory_max(data_manager, self.samples_per_class)
        # elif self.replace == 'random':
        #     self.build_rehearsal_memory_random(data_manager, self.samples_per_class)
        # elif self.replace == 'entropy':
        #     self.build_rehearsal_memory_entropy(data_manager, self.samples_per_class)
        # elif self.replace == 'imblance':
        #     self.build_rehearsal_memory_imblance(data_manager, self.samples_per_class)
        
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        params = ([], [], [])
        params = split_params(self._network, params)
        params = [
            {'params': params[1], 'weight_decay': self.wd},
            {'params': params[2], 'weight_decay': 0}
        ]
        if self._cur_task == 0:
            optimizer = optim.SGD(
                params,
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                params,
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def print_memory_usage(self):
        allocated = torch.cuda.memory_allocated() / 1024**2  # 转换为MB
        cached = torch.cuda.memory_reserved() / 1024**2  # 转换为MB
        print(f"Allocated: {allocated:.2f} MB, Cached: {cached:.2f} MB")
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epoach))
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
                    for t in range(self.T_max):
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
                        in_data, _ = torch.broadcast_tensors(inputs, torch.zeros((self.T_max,) + inputs.shape))
                    in_data = in_data.reshape(-1, *in_data.shape[2:])
                    output = self._network(in_data)
                    avg_fr = output if self.dynamic else output.mean(dim=0)


                if self.dynamic:
                    loss = self.evaluator(avg_fr, targets, T = self.T, lamb = self.lamb, means = self.means)
                else:
                    loss = self.evaluator(avg_fr, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # acc
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
                    self.epoach,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epoach,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)
        # info = "Task"
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epoach))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (idxs, inputs, targets) in enumerate(train_loader):
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
                    out_spikes = []
                    for t in range(self.T_max):
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
                        in_data, _ = torch.broadcast_tensors(inputs, torch.zeros((self.T_max,) + inputs.shape))
                    in_data = in_data.reshape(-1, *in_data.shape[2:])
                    output = self._network(in_data)
                    avg_fr = output if self.dynamic else output.mean(dim=0)
                if self.swap_ratio > 0:
                    asyncio.run(train_loader.dataset._swap(idxs, targets))

                if self.dynamic:
                    loss = self.evaluator(avg_fr, targets, T = self.T, lamb = self.lamb, means = self.means)
                else:
                    loss = self.evaluator(avg_fr, targets)
                # loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # acc
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
                    self.epoach,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epoach,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        # info = "Task"
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