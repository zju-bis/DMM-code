import logging
import numpy as np
import torch
import asyncio
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from spikingjelly.activation_based import functional
from nets.model_setting import rate_model_setting, bptt_model_setting
from spikingjelly.activation_based import functional
from utils.inc_net import get_convnet
from nets.layer import CustomEvaluator, CustomUpdataEvaluator, BicUpdataEvaluator
epochs = 150
milestones = [60, 100, 140]
lrate_decay = 0.1
batch_size = 128
split_ratio = 0.1
T = 2
num_workers = 8

class UpdataEvaluator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.evaluator = evaluator
        # self.model_type = model_type
        # self.model_time_step = model_time_step

    def forward(self, avg_fr: torch.Tensor, old_avg_fr: torch.Tensor, targets: torch.Tensor, known_classes, lamda):
        clf_loss = F.cross_entropy(avg_fr, targets)
        hat_pai_k = F.softmax(old_avg_fr / T, dim=1)
        log_pai_k = F.log_softmax(
            old_avg_fr[:, : known_classes] / T, dim=1
        )
        distill_loss = -torch.mean(
            torch.sum(hat_pai_k * log_pai_k, dim=1)
        )
        loss = distill_loss * lamda + clf_loss * (1 - lamda)
        return loss
class BisEvaluator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.evaluator = evaluator
        # self.model_type = model_type
        # self.model_time_step = model_time_step

    def forward(self, avg_fr: torch.Tensor, targets: torch.Tensor):
        loss = F.cross_entropy(torch.softmax(avg_fr, dim=1), targets)
        return loss
class BiC(BaseLearner):
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
        self.epoach = args['epoachs']
        self.dataset = 'img'
        if args["dataset"] == 'ncaltech101swap':
            self.T_max = 14
            self.dataset = 'dvs'
        if args["dataset"] == 'dvs128swap':
            self.T_max = 20
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
        self.init_eva = CustomEvaluator(init_eva, "rate" if self.rate_flag else "bptt", args["T"])
        bis_eva = BisEvaluator()
        self.bis_eva = CustomEvaluator(bis_eva, "rate" if self.rate_flag else "bptt", args["T"])
        up_eva = UpdataEvaluator()
        self.up_eva = BicUpdataEvaluator(up_eva, "rate" if self.rate_flag else "bptt", args["T"])

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

        if self._cur_task >= 1:
            train_dset, val_dset = data_manager.get_train_val_dataset_with_split(
                np.arange(self._known_classes, self._total_classes),
                appendent=self._get_memory(),
                val_samples_per_class=int(
                    split_ratio * self._memory_size / self._known_classes
                ),
                swap_ratio = self.swap_ratio
            )
            self.val_loader = DataLoader(
                val_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            logging.info(
                "Stage1 dset: {}, Stage2 dset: {}".format(
                    len(train_dset), len(val_dset)
                )
            )
            self.lamda = self._known_classes / self._total_classes
            logging.info("Lambda: {:.3f}".format(self.lamda))
        else:
            train_dset = data_manager.get_train_dataset(
                np.arange(self._known_classes, self._total_classes),
                appendent=self._get_memory(),
                swap_ratio=self.swap_ratio
            )
        test_dset = data_manager.get_test_dataset(
            np.arange(0, self._total_classes)
        )

        self.train_loader = DataLoader(
            train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.test_loader = DataLoader(
            test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self._log_bias_params()
        self._stage1_training(self.train_loader, self.test_loader)
        if self._cur_task >= 1:
            self._stage2_bias_correction(self.val_loader, self.test_loader)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        self._log_bias_params()

    def _run(self, train_loader, test_loader, optimizer, scheduler, stage):
        for epoch in range(1, self.epoach + 1):
            self._network.train()
            losses = 0.0
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
                if stage == "training":
                    if self.dynamic:
                        clf_loss = self.init_eva(avg_fr, targets, T = self.T, lamb = self.lamb, means = self.means)
                    else:
                        clf_loss = self.init_eva(avg_fr, targets)
                    if self._old_network is not None:
                        functional.reset_net(self._old_network)
                        if self.dynamic:
                            for name, module in self._network.named_modules():
                                setattr(module, 'time_step', self.T)
                        if self.step_mode == 's':
                            if self.dataset == 'dvs':
                                inputs = inputs.permute(1, 0, 2, 3, 4)
                            out_spikes = []
                            for t in range(self.T_max):
                                if self.dataset == 'dvs':
                                    out = self._old_network(inputs[t])
                                else:
                                    out = self._old_network(inputs)
                                out_spikes.append(out)
                            output = torch.stack(out_spikes, dim=0)
                            avg_fr_old = output if self.dynamic else output.mean(dim=0)
                        else:
                            if self.dataset == 'dvs':
                                in_data = inputs.permute(1, 0, 2, 3, 4)
                            else:
                                in_data, _ = torch.broadcast_tensors(inputs, torch.zeros((self.T_max,) + inputs.shape))
                            in_data = in_data.reshape(-1, *in_data.shape[2:])
                            output = self._old_network(in_data).detach()
                            avg_fr_old = output if self.dynamic else output.mean(dim=0)
                        if self.dynamic:
                            loss = self.up_eva(avg_fr, avg_fr_old, targets, self._known_classes, self.lamda, T = self.T, lamb = self.lamb, means = self.means)
                        else:
                            loss = self.up_eva(avg_fr, avg_fr_old, targets, self._known_classes, self.lamda)
                        # hat_pai_k = F.softmax(avg_fr / T, dim=1)
                        # log_pai_k = F.log_softmax(
                        #     avg_fr[:, : self._known_classes] / T, dim=1
                        # )
                        # distill_loss = -torch.mean(
                        #     torch.sum(hat_pai_k * log_pai_k, dim=1)
                        # )
                        # loss = distill_loss * self.lamda + clf_loss * (1 - self.lamda)
                    else:
                        loss = clf_loss
                elif stage == "bias_correction":
                    if self.dynamic:
                        loss = self.bis_eva(avg_fr, targets, T = self.T, lamb = self.lamb, means = self.means)
                    else:
                        loss = self.bis_eva(avg_fr, targets)
                else:
                    raise NotImplementedError()

                if self.swap_ratio > 0:
                    asyncio.run(train_loader.dataset._swap(idxs, targets))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            train_acc = self._compute_accuracy(self._network, train_loader)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "{} => Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}".format(
                stage,
                self._cur_task,
                epoch,
                self.epoach,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            logging.info(info)

    def _stage1_training(self, train_loader, test_loader):
        """
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        """

        ignored_params = list(map(id, self._network.bias_layers.parameters()))
        base_params = filter(
            lambda p: id(p) not in ignored_params, self._network.parameters()
        )
        network_params = [
            {"params": base_params, "lr": self.lr, "weight_decay": self.wd},
            {
                "params": self._network.bias_layers.parameters(),
                "lr": 0,
                "weight_decay": 0,
            },
        ]
        optimizer = optim.SGD(
            network_params, lr=self.lr, momentum=0.9, weight_decay=self.wd
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=lrate_decay
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        self._run(train_loader, test_loader, optimizer, scheduler, stage="training")

    def _stage2_bias_correction(self, val_loader, test_loader):
        if isinstance(self._network, nn.DataParallel):
            self._network = self._network.module
        network_params = [
            {
                "params": self._network.bias_layers[-1].parameters(),
                "lr": self.lr,
                "weight_decay": self.wd,
            }
        ]
        optimizer = optim.SGD(
            network_params, lr=self.lr, momentum=0.9, weight_decay=self.wd
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=lrate_decay
        )

        self._network.to(self._device)

        self._run(
            val_loader, test_loader, optimizer, scheduler, stage="bias_correction"
        )

    def _log_bias_params(self):
        logging.info("Parameters of bias layer:")
        params = self._network.get_bias_params()
        for i, param in enumerate(params):
            logging.info("{} => {:.3f}, {:.3f}".format(i, param[0], param[1]))

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