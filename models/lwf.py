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
from nets.model_setting import bptt_model_setting
from spikingjelly.activation_based import functional
from utils.inc_net import get_convnet

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 160]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 250
lrate = 0.1
milestones = [60, 120, 180, 220]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2
lamda = 3


class LwF(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = get_convnet(args)
        self.step_mode = args["step_mode"]
        self.T_max = args["T"]
        bptt_model_setting(self._network, time_step=self.T_max, step_mode=self.step_mode)
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

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

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
                lr=init_lr,
                weight_decay=init_weight_decay,
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
            )
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
                    avg_fr = output.mean(dim=0)
                else:
                    if self.dataset == 'dvs':
                        in_data = inputs.permute(1, 0, 2, 3, 4)
                    # in_data = inputs.permute(1, 0, 2, 3, 4)
                    else:
                        in_data, _ = torch.broadcast_tensors(inputs, torch.zeros((self.T_max,) + inputs.shape))
                    in_data = in_data.reshape(-1, *in_data.shape[2:])
                    output = self._network(in_data)
                    avg_fr = output.mean(dim=0)

                loss = F.cross_entropy(avg_fr, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

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
            self._old_network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                targets = targets.long()
                functional.reset_net(self._network)
                functional.reset_net(self._old_network)
                if self.step_mode == 's':
                    if self.dataset == 'dvs':
                        inputs = inputs.permute(1, 0, 2, 3, 4)
                    # in_data = inputs.permute(1, 0, 2, 3, 4)
                    out_spikes = []
                    old_out_spikes = []
                    for t in range(self.T_max):
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
                    avg_fr = output.mean(dim=0)
                    old_avg_fr = old_output.mean(dim=0)
                else:
                    if self.dataset == 'dvs':
                        in_data = inputs.permute(1, 0, 2, 3, 4)
                    # in_data = inputs.permute(1, 0, 2, 3, 4)
                    else:
                        in_data, _ = torch.broadcast_tensors(inputs, torch.zeros((self.T_max,) + inputs.shape))
                    in_data = in_data.reshape(-1, *in_data.shape[2:])
                    output = self._network(in_data)
                    avg_fr = output.mean(dim=0)
                    old_out = self._old_network(inputs)
                    old_avg_fr = old_out.mean(dim=0)

                if (self._memory_per_class == 0):
                    fake_targets = targets - self._known_classes
                    loss_clf = F.cross_entropy(
                        avg_fr[:, self._known_classes :], fake_targets
                    )
                else:
                    loss_clf = F.cross_entropy(
                        avg_fr, targets
                    )

                loss_kd = _KD_loss(
                    avg_fr[:, : self._known_classes],
                    old_avg_fr,
                    T,
                )

                loss = lamda * loss_kd + loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                with torch.no_grad():
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
        return cnn_accy, nme_accy

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

