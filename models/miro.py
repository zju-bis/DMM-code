import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import DmmIncrementalNet
from nets.vgg import vggsnn_cifar
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
# import psutil
from nets import *
from nets.layer import CustomEvaluator, SwapEvaluator
from nets.model_setting import rate_model_setting, bptt_model_setting
from spikingjelly.activation_based import functional, neuron
from utils.inc_net import get_convnet
from utils import power_check as pc
import gc
import os
import optuna
import asyncio
import time
import json
import math
init_epoch = 1
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


lrate = 0.1
milestones = [40, 70]
lrate_decay = 0.1
batch_size = 128
num_workers = 8
# def split_params(model, paras=([], [], [])):
#     for n, module in model._modules.items():
#         paras[2].append(n)
#         paras[1].append(n)
#         if isinstance(module, neuron.LIFNode) and hasattr(module, "thresh"):
#             for name, para in module.named_parameters():
#                 paras[0].append(name)
#         elif 'batchnorm' in module.__class__.__name__.lower():
#             for name, para in module.named_parameters():
#                 paras[2].append(name)
#         elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.modules.conv._ConvNd):
#             paras[1].append("weight")
#             if module.bias is not None:
#                 paras[2].append("bias")
#         elif len(list(module.children())) > 0:
#             paras = split_params(module, paras)
#         elif module.parameters() is not None:
#             for name, para in module.named_parameters():
#                 paras[1].append(name)
#     return paras
def split_params(model, paras=([], [], [])):
    for n, module in model._modules.items():
        if isinstance(module, neuron.LIFNode) and hasattr(module, "thresh"):
            for name, para in module.named_parameters():
                paras[0].append(para)
        elif 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                paras[2].append(para)
        elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.modules.conv._ConvNd):
            paras[1].append(module.weight)
            if module.bias is not None:
                paras[2].append(module.bias)
        elif len(list(module.children())) > 0:
            paras = split_params(module, paras)
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                paras[1].append(para)
    return paras

class Miro(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.sched = args["scheduler"]
        self._network = get_convnet(args)
        self.rate_flag = args["rate_flag"]
        self.step = args["step"]
        self.alg = args["alg"]
        if self.alg == "best_history":
            self.best_score = -50000
            self.best_config = None
        elif self.alg == "dmm" or self.alg == "miro":
            self.n_trials = args["n_trials"]
        elif self.alg == "heuristic":
            self.precent = args["precent"]
        self.strategy = args["strategy"]
        self.step_mode = args["step_mode"]
        self.T_max = args["T"]
        self.dataset = "img"
        self.config = args["img_config"]
        self.edge = args["edge"]
        if "noise_prec" in args:
            self.noise_prec = args["noise_prec"]
        else:
            self.noise_prec = 0.0
        if self.edge:
            self.sum_memory = 0
            self.sum_pow = 0
        if args["budget"] == 'small':
            if args["dataset"] == 'ncaltech101swap' or args["dataset"] == 'dvs128swap' or args["dataset"] == 'urbansoundswap' or args["dataset"] == 'dsadsswap':
                self.budget = 1000
            else:
                self.budget = 10000
        elif args["budget"] == 'medium':
            if args["dataset"] == 'ncaltech101swap' or args["dataset"] == 'dvs128swap' or args["dataset"] == 'urbansoundswap' or args["dataset"] == 'dsadsswap':
                self.budget = 2000
            else:
                self.budget = 25000
        elif args["budget"] == 'large':
            if args["dataset"] == 'ncaltech101swap' or args["dataset"] == 'dvs128swap' or args["dataset"] == 'urbansoundswap' or args["dataset"] == 'dsadsswap':
                self.budget = 3000
            else:
                self.budget = 50000
        self.data_ratio = args["data_ratio"]
        if args["dataset"] == 'urbansoundswap' or args["dataset"] == 'dsadsswap':
            self.config = args["dvs_config"]
            self.data_ratio = 0.5
        if args["dataset"] == 'ncaltech101swap' or args["dataset"] == 'dvs128swap':
            self.config = args["dvs_config"]
            self.T_max = 8
            self.dataset = 'dvs'
            self.dynamic = False
            self.data_ratio = 0.5
        if "lamb" in args:
            self.lamb = args["lamb"]
            self.means = args["means"]
        else:
            self.lamb = 0
            self.means = 1.0
        self.dynamic = args["dynamic"]
        if self.rate_flag:
            rate_model_setting(self._network, time_step=self.T_max, step_mode=self.step_mode)
        else:
            bptt_model_setting(self._network, time_step=self.T_max, step_mode=self.step_mode)
        self._network.to(self._device)
        self.lr = args["lr"]
        self.optm = args["optm"]
        self.wd = args["wd"]
        self.swap_ratio = args["swap_ratio"]
        evaluator = torch.nn.CrossEntropyLoss()
        self.evaluator = CustomEvaluator(evaluator, "rate" if self.rate_flag else "bptt", args["T"])
        if args['alg'] == "heuristic":
            self.return_path = f"logs/{args['model_name']}/{args['dataset']}/{args['n_trials']}/{args['alg']}{args['rate_flag']}{args['dynamic']}/{args['budget']}{args['precent']}/{args['swap_policy']}/{args['seed']}/{args['swap_ratio']}"
        else:
            self.return_path = f"logs/{args['model_name']}/{args['dataset']}/{args['n_trials']}/{args['alg']}{args['rate_flag']}{args['dynamic']}/{args['budget']}{args['strategy']}/{args['swap_policy']}/{args['seed']}/{args['swap_ratio']}"
        os.makedirs(self.return_path, exist_ok=True)
        self.dataset_name = args["dataset"]
        self.budget_name = args["budget"]
        self.pretrain_epoach = args["pretrain_epoach"]
        self.trail = args["trail"]
        self.cutline = args["cutline"]
        self.epochs = args["epochs"]
        self.saved_config = []
        if self.edge:
            with open(os.path.join(self.return_path, 'saved_config.json'), 'r') as f:
                self.saved_config = json.load(f)
        self.len_per_cls = dict()
        self.offset = dict()
        self.rb_size = 0
        self.sum_t = 0
    def after_task(self):
        self._known_classes = self._total_classes
    def find_best_config(self, data_manager, test_loader, n_trials = 5,  sampler = None):
        best_config = self.config[0]
        data_manager.stream_resize(best_config[1], self._cur_task)
        # logging.info(best_config[1])
        self._resize_memory(data_manager, best_config[0])
        if self.swap_ratio > 0 and self._cur_task > 0:
            data_manager.reset_swap_class_dist()
            data_manager.swap_thr = (batch_size * best_config[0] * self.swap_ratio * self.epochs * 
                math.ceil((best_config[0] + best_config[1])/batch_size) / (self._known_classes * (best_config[0] + best_config[1])))

        train_dataset = data_manager.get_train_dataset(
            np.arange(self._known_classes, self._total_classes),
            appendent=self._get_memory(),
            swap_ratio=self.swap_ratio,
            noise_prec=self.noise_prec
        )
        # logging.info(len(train_dataset))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        # observation = np.full((2, len(self.config)), -1)
        
        torch.save(self._network.state_dict(), os.path.join(self.return_path, "org_net.pkl"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.return_path, "org_opt.pkl"))
        torch.save(self.scheduler.state_dict(), os.path.join(self.return_path, "org_sch.pkl"))

        self.train(train_loader, test_loader, self.pretrain_epoach, data_manager)
        torch.save(self._network.state_dict(), os.path.join(self.return_path, "pretrain_net.pkl"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.return_path, "pretrain_opt.pkl"))
        torch.save(self.scheduler.state_dict(), os.path.join(self.return_path, "pretrain_sch.pkl"))
        del train_dataset, train_loader
        gc.collect()
        best_config, best_score = None, 0
        class SkipTrial(Exception):
            pass
        # 使用Optuna进行启发式搜索
        def objective(trial):
            rb_size = trial.suggest_int("rb_size", self.config[0][1], self.config[0][0])
            st_size = trial.suggest_int("st_size", self.config[1][1], self.config[1][0])
            if rb_size + st_size > self.budget or rb_size * self.data_ratio // len(self.offset) < 1 or st_size  * self.data_ratio // data_manager._increments[self._cur_task] < 1:
                logging.info(f"Skipping trial with rb_size {rb_size} and st_size {st_size} (total size exceeds budget or number sample = 0)")
                raise optuna.TrialPruned()
            self._network.load_state_dict(torch.load(os.path.join(self.return_path, "pretrain_net.pkl"), map_location = self._device))
            self.optimizer.load_state_dict(torch.load(os.path.join(self.return_path, "pretrain_opt.pkl"), map_location = self._device))
            self.scheduler.load_state_dict(torch.load(os.path.join(self.return_path, "pretrain_sch.pkl"), map_location = self._device))
            data_manager.stream_resize(st_size * self.data_ratio, self._cur_task)
            self._resize_memory(data_manager, rb_size * self.data_ratio)
            if self.swap_ratio > 0 and self._cur_task > 0:
                data_manager.reset_swap_class_dist()
                data_manager.swap_thr = (batch_size * rb_size * self.swap_ratio * self.epochs * 
                    math.ceil((rb_size + st_size)/batch_size) / (self._known_classes * (rb_size + st_size)))
            logging.info(f"rb and st is {rb_size} {st_size}")
            train_dataset = data_manager.get_train_dataset(
                np.arange(self._known_classes, self._total_classes),
                appendent=self._get_memory(),
                swap_ratio=self.swap_ratio,
                noise_prec=self.noise_prec
            )
            # logging.info(len(train_dataset))
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            # pretrain_net = copy.deepcopy(self._network)
            self.train(train_loader, test_loader, self.trail, data_manager)
            acc = self._compute_accuracy(self._network, test_loader)
            energy = rb_size + st_size
            if energy != 0:
                eta = acc / energy
            else:
                eta = 0
            del train_dataset, train_loader
            logging.info(f'ACC: {acc}, ENERGY: {energy}, ETA: {eta}')
            gc.collect()
            if self.strategy == "HU":
                return acc, eta
            elif self.strategy == "LE":
                return acc, -energy
        trials = []
        if self.alg == "best_static" or self.alg == "best_history":
            study = optuna.create_study(directions=["maximize", "maximize"], sampler = sampler)
            study.optimize(objective, n_trials=len(sampler._all_grids))

            trials = [t for t in study.get_trials() if t.state == optuna.trial.TrialState.COMPLETE] # 先acc前cutline个 后eta排序选最大
        else:
            while(len(trials) < n_trials):
                # 创建Optuna study并进行优化
                study = optuna.create_study(directions=["maximize", "maximize"], sampler = sampler)
                study.optimize(objective, n_trials=n_trials - len(trials))
                # trial_with_highest_performance = max(study.best_trials, key=lambda t: t.values[2]) # study.best_trials 选取

                tmp_trials = [t for t in study.get_trials() if t.state == optuna.trial.TrialState.COMPLETE] # 先acc前cutline个 后eta排序选最大
                trials += tmp_trials
                # logging.info("trails are", tmp_trials, trials)
        sorted_trials_by_acc = sorted(trials, key=lambda t: t.values[0], reverse=True)
        top_trials_acc = sorted_trials_by_acc[:int(len(trials)*self.cutline)]
        trial_with_highest_performance = max(top_trials_acc, key=lambda t: t.values[1])

        # if self.time_log_file: time_st = time.perf_counter()
        # best_config = study.best_params
        # best_score = study.best_value
        best_config=trial_with_highest_performance.params
        best_score=trial_with_highest_performance.values[1]
        logging.info(f'BEST CONFIG: {best_config}')
        self._network.load_state_dict(torch.load(os.path.join(self.return_path, "org_net.pkl"), map_location = self._device))
        self.optimizer.load_state_dict(torch.load(os.path.join(self.return_path, "org_opt.pkl"), map_location = self._device))
        self.scheduler.load_state_dict(torch.load(os.path.join(self.return_path, "org_sch.pkl"), map_location = self._device))
        
        # if self.time_log_file: 
        #     with open(self.time_log_file,'a') as f: f.write(f'{self.task_id},-,Optimizer,Scorer,{time.perf_counter()-time_st}\n')
        return best_config, best_score

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        # logging.info(self._total_classes, data_manager._increments)
        self._network.updata_classifier(self._total_classes)
        self._network.to(self._device)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        test_dataset = data_manager.get_test_dataset(
            np.arange(0, self._total_classes)
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        if self._cur_task > 0:
            del self.optimizer
            gc.collect()
        params = ([], [], [])
        params = split_params(self._network, params)
        params = [
            {'params': params[1], 'weight_decay': self.wd},
            {'params': params[2], 'weight_decay': 0}
        ]
        if self.optm == 'sgdm':
            self.optimizer = optim.SGD(params, lr = self.lr, momentum=0.9)
        elif self.optm == 'adam':
            self.optimizer = optim.Adam(params, lr = self.lr, amsgrad=False)
        if self.sched == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=0, T_max = self.epochs)
        else:
            self.scheduler = None
        if self.edge and self.alg != "static" and self.alg != "heuristic":
            best_config = self.saved_config[self._cur_task]
        else:
            if self.alg == "carm":
                best_config = self._carm(data_manager)
            elif self.alg == "best_static":
                best_config = self._best_static(data_manager)
            elif self.alg == "static":
                best_config = self._static()
            elif self.alg == "best_history":
                best_config = self._best_history(data_manager)
            elif self.alg == "heuristic":
                best_config = self._heuristic(data_manager)
            elif self.alg == "dmm":
                best_config = self._dmm(data_manager)
            elif self.alg == "miro":
                best_config = self._miro(data_manager)
        logging.info(f"best config is {best_config}")
        data_manager.stream_resize(best_config['st_size'], self._cur_task)
        self._resize_memory(data_manager, best_config['rb_size'])
        if self.swap_ratio > 0 and self._cur_task > 0:
            data_manager.reset_swap_class_dist()
            data_manager.swap_thr = (batch_size * best_config['rb_size'] * self.swap_ratio * self.epochs * 
                math.ceil((best_config['rb_size'] + best_config['st_size'])/batch_size) / (self._known_classes * (best_config['rb_size'] + best_config['st_size'])) )
        train_dataset = data_manager.get_train_dataset(
            np.arange(self._known_classes, self._total_classes),
            appendent=self._get_memory(),
            swap_ratio=self.swap_ratio,
            noise_prec=self.noise_prec
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        st = time.time()
        # if self.edge:
        #     input = {"train_loader": train_loader, "test_loader": self.test_loader, "nepoach": self.epochs}
        #     pc.printFullReport(pc.getDevice())
        #     pl = pc.PowerLogger(interval=0.05)
        #     pl.start()
        #     time.sleep(5)
        #     pl.recordEvent(name='Process Start')
        #     self.train(train_loader, self.test_loader, self.epochs)
        #     time.sleep(5)
        #     pl.stop()
        #     filename = f'./{self.alg}/{self.dataset_name}/{self.strategy}/{self.budget_name}/'
        #     res = pl.showDataTraces(filename=filename)
        #     logging.info(str(pl.eventLog))
        #     pc.printFullReport(pc.getDevice())
        #     self.sum_pow += res
        #     logging.info(f"Sum Pow: {self.sum_pow}J")
        #     logging.info(f"Averge Pow: {self.sum_pow / (self._cur_task + 1) / self.epochs}J")
        #     logging.info(f"Sum Memory: {self.sum_memory} MB")
        #     logging.info(f"Average Memory: {self.sum_memory / (self._cur_task + 1)} MB")
        # else:
        self.train(train_loader, self.test_loader, self.epochs, data_manager)
        if self.edge:
            logging.info(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
            # power_usage = nvmlDeviceGetPowerUsage(self.handle) / 1000  # 功耗 (瓦特)

        et = time.time()
        self.sum_t += et - st
        logging.info(f"Time taken for training: { et - st}")
        logging.info(f"Time Sum for training: {self.sum_t}")
        logging.info(f"Time Average for training: {self.sum_t / (self._cur_task + 1) / self.epochs}")
        del train_dataset, train_loader
        gc.collect()
        self._update_memory(data_manager)

    def _best_static(self, data_manager):
        if self._cur_task == 0:
            class_num = data_manager.get_task_size(self._cur_task)
            file_num = 0
            for a in range(class_num):
                file_num = file_num + data_manager.num_per_class[a]
            best_config = {'rb_size': 0, 'st_size': file_num}
            self.saved_config.append(best_config)
            return best_config
        elif self._cur_task == 1:
            if self.dataset == 'dvs':
                step = 50
            else:
                step = 500
            search_space = {
                "rb_size": list(range(self.config[0][1], self.config[0][0], step)),
                "st_size": list(range(self.config[1][1], self.config[1][0], step))
            }
            sampler = optuna.samplers.GridSampler(search_space, seed = 1)
            self.static_config, _ = self.find_best_config(data_manager, self.test_loader,
                n_trials = 0, sampler = sampler)
            self.saved_config.append(self.static_config)
            return self.static_config
        else:
            self.saved_config.append(self.static_config)
            return self.static_config

    def _carm(self, data_manager):
        if self._cur_task == 0:
            class_num = data_manager.get_task_size(self._cur_task)
            file_num = 0
            for a in range(class_num):
                file_num = file_num + data_manager.num_per_class[a]
            best_config = {'rb_size': 0, 'st_size': file_num}
            self.saved_config.append(best_config)
            return best_config
        else:
            best_config = {'rb_size': self.budget * 4 // 5, 'st_size': self.budget // 5}
            return best_config

    def _static(self):
        print(self.config)
        return {'rb_size': self.config[0][1], 'st_size': self.config[0][0]}

    def _best_history(self, data_manager):
        if self._cur_task == 0:
            class_num = data_manager.get_task_size(self._cur_task)
            file_num = 0
            for a in range(class_num):
                file_num = file_num + data_manager.num_per_class[a]
            best_config = {'rb_size': 0, 'st_size': file_num}
            self.saved_config.append(best_config)
            return best_config
        elif self._cur_task > len(data_manager._increments) // 2:
            self.saved_config.append(self.best_config)
            return self.best_config
        else:
            if self.dataset == 'dvs':
                step = 50
            else:
                step = 500
            search_space = {
                "rb_size": list(range(self.config[0][1], self.config[0][0], step)),
                "st_size": list(range(self.config[1][1], self.config[1][0], step))
            }
            sampler = optuna.samplers.GridSampler(search_space, seed = 1)
            config, score = self.find_best_config(data_manager, self.test_loader,
                n_trials = 0, sampler = sampler)
            if score > self.best_score:
                self.best_score = score
                self.best_config = config
            self.saved_config.append(config)
            return config

    def _heuristic(self, data_manager):
        replay_files = 0
        for i in range(self._known_classes):
            replay_files += data_manager.num_per_class[i]
        stream_files = 0
        for i in range(self._known_classes, self._total_classes):
            stream_files += data_manager.num_per_class[i]
        st_size = int(stream_files / (stream_files + replay_files) * self.budget * self.precent)
        rb_size = int(replay_files / (stream_files + replay_files) * self.budget * self.precent)
        best_config = {'rb_size': rb_size, 'st_size': st_size}
        return best_config

    def _miro(self, data_manager):
        if self._cur_task == 0:
            class_num = data_manager.get_task_size(self._cur_task)
            file_num = 0
            for a in range(class_num):
                file_num = file_num + data_manager.num_per_class[a]
            best_config = {'rb_size': 0, 'st_size': file_num}
            self.saved_config.append(best_config)
        else:
            sampler = optuna.samplers.RandomSampler(seed=1)
            best_config, _ = self.find_best_config(data_manager, self.test_loader, sampler = sampler, n_trials = self.n_trials)
            self.saved_config.append(best_config)
        return best_config

    def _dmm(self, data_manager):
        if self._cur_task == 0:
            class_num = data_manager.get_task_size(self._cur_task)
            file_num = 0
            for a in range(class_num):
                file_num = file_num + data_manager.num_per_class[a]
            best_config = {'rb_size': 0, 'st_size': file_num}
            self.saved_config.append(best_config)
        else:
            sampler = optuna.samplers.TPESampler(seed=1)
            best_config, _ = self.find_best_config(data_manager, self.test_loader, sampler = sampler, n_trials = self.n_trials)
            self.saved_config.append(best_config)
        return best_config

    def eval_task(self, save_conf=False):
        y_pred, y_true = self._eval_snn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        nme_accy = None

        return cnn_accy, nme_accy

    def _update_memory(self, data_manager):
        num_classes = self._total_classes
        memory_per_cls = int(self.rb_size / num_classes)
        old_offset, old_len_per_cls = self.offset, self.len_per_cls
        self.len_per_cls = {x:memory_per_cls for x in range(num_classes)}
        self.offset = {x:x*memory_per_cls for x in range(num_classes)}
        lenth = len(self._data_memory)
        append_data = []
        append_target = []
        logging.info(f"task is {self._cur_task} upgrate memory len of data memory {lenth}")
        logging.info(f"old_offset {old_offset}")
        logging.info(f"old_len_per_cls {old_len_per_cls}")
        logging.info(f"offset {self.offset}")
        logging.info(f"len_per_cls {self.len_per_cls}")
        for i in range(num_classes):
            if i < self._known_classes:
                total = old_len_per_cls[i]
            else :
                data, targets, _ = data_manager.get_train_dataset(
                    np.arange(i, i + 1),
                    ret_data=True,
                    space = self.len_per_cls,
                    policy = 'random',
                    noise_prec=self.noise_prec
                )
                data = data[0]
                targets = targets[0]
                total = data.shape[0]
            if memory_per_cls >= total: 
                cls_offset = total
            else: cls_offset=memory_per_cls
            if i < len(self.offset)-1: 
                self.offset[i+1] = self.offset[i] + cls_offset
            self.len_per_cls[i] = cls_offset
            for j in range(cls_offset):
                st = self.offset[i]
                # Old data triming       
                if i < (len(old_offset)) and j < old_len_per_cls[i]:
                    # moving old data 
                    old_st = old_offset[i]
                    old_len = old_len_per_cls[i]
                    new_len = self.len_per_cls[i]
                    if new_len < old_len:
                        old_idx = old_st+old_len-new_len+j
                    else: 
                        old_idx = old_st + j
                    self._targets_memory[st+j] = self._targets_memory[old_idx]
                    self._data_memory[st+j] = self._data_memory[old_idx]
                    # logging.info(f'replay[{st+j}] <- replay[{old_idx}]')
                    # df.write(f'replay[{st+j}] <- replay[{old_idx}] = {old_filenames[old_idx]}\n')
                    # logging.info(old_targets[old_idx],old_filenames[old_idx])
                # Insert new samples from new classes
                else: 
                    stream_idx = total-cls_offset+j
                    # logging.info(f'2 replay[{st+j}] <- stream[{stream_idx}]',end=': ')
                    # logging.info(sub_stream_label[stream_idx],sub_stream_filenames[stream_idx])
                    if st+j >= lenth:
                        append_data.append(data[stream_idx])
                        append_target.append(targets[stream_idx])
                    else:
                        self._targets_memory[st+j] = targets[stream_idx]
                        self._data_memory[st+j] = data[stream_idx]
        if len(append_data) > 0:
            self._data_memory = np.concatenate([self._data_memory, np.array(append_data)])
            self._targets_memory = np.concatenate([self._targets_memory, np.array(append_target)])
    def _resize_memory(self, data_manager, new_size):
        new_size = int(new_size)
        if self._cur_task == 0:
            logging.info("No data memory for the first task")
            return
        if self.rb_size == new_size:
            logging.info("Memory size is already correct")
            return
        logging.info("Resizing memory from {} to {}".format(self.rb_size, new_size))
        logging.info(f"offset and len before Resizing memory {self.offset} {self.len_per_cls}")
        memory_per_cls = new_size // len(self.offset)
        if len(self._data_memory) < new_size:
            del self._data_memory, self._targets_memory
            gc.collect()
            space_per_class = {i:memory_per_cls for i in range(self._known_classes)}
            data_memory, targets_memory, _ = data_manager.get_train_dataset(
                np.arange(0, self._known_classes), ret_data = True, space = space_per_class, policy = 'random', noise_prec=self.noise_prec
            )
            self._data_memory = data_memory[0]
            self._targets_memory = targets_memory[0]
            self.len_per_cls = space_per_class
            self.offset = {i:i*memory_per_cls for i in range(self._known_classes)}
        else:
            mem_per_cls = memory_per_cls
            num_classes = len(self.offset) 
            new_len_per_cls =  {x:self.len_per_cls[x] for x in range(num_classes)}
            smaller_classes = {label:n_samples for i,(label,n_samples) in enumerate(self.len_per_cls.items()) if n_samples<mem_per_cls}
            smaller_labels = []
            all_file = new_size
            while len(smaller_classes)>0:
                new_len_per_cls.update(smaller_classes)
                smaller_labels.extend(list(smaller_classes.keys()))
                all_file -= sum(smaller_classes.values())
                num_classes  = len(self.offset)- len(smaller_labels)
                if num_classes ==0: break
                mem_per_cls = all_file//num_classes # Originaly was just deviding causing float value to appear, changed to only return in, could this have problems?
                smaller_classes = {label:n_samples for  i,(label,n_samples) in enumerate(self.len_per_cls.items()) if n_samples<mem_per_cls and label not in smaller_labels} 
                

            memory_per_cls = mem_per_cls
            logging.info(f"MEM PER CLS : {memory_per_cls}")
            
            old_offset, old_len_per_cls = self.offset, self.len_per_cls
            self.len_per_cls = new_len_per_cls
            for i in range(1, len(self.offset)): self.offset[i] = self.offset[i-1] + self.len_per_cls[i-1]
            for i in range(num_classes): 
                if self.len_per_cls[i] > memory_per_cls:
                    self.len_per_cls[i] = memory_per_cls
                    cls_offset = memory_per_cls
                else: cls_offset = self.len_per_cls[i]
                if i < len(self.offset)-1: 
                    self.offset[i+1] = self.offset[i] + cls_offset
                
                self.len_per_cls[i] = cls_offset      
                for j in range(cls_offset):
                    st = self.offset[i]
                    if i < (len(old_offset)) and j < old_len_per_cls[i]:
                        old_st = old_offset[i]
                        old_len = old_len_per_cls[i]
                        new_len = self.len_per_cls[i]
                        if new_len < old_len:
                            old_idx = old_st+old_len-new_len+j
                        else: 
                            old_idx = old_st + j
                        self._targets_memory[st+j] = self._targets_memory[old_idx]
                        self._data_memory[st+j] = self._data_memory[old_idx]
            self._data_memory = self._data_memory[:new_size]
            self._targets_memory = self._targets_memory[:new_size]
        logging.info(f"offset and len after Resizing memory {self.offset}\n{self.len_per_cls}")
        self.rb_size = new_size
        return new_size
    # def print_memory_usage(self):
    #     allocated = torch.cuda.memory_allocated() / 1024**2  # 转换为MB
    #     cached = torch.cuda.memory_reserved() / 1024**2  # 转换为MB
    #     logging.info(f"Allocated: {allocated:.2f} MB, Cached: {cached:.2f} MB")

    def train(self, train_loader, test_loader, nepoach, data_manager):
        prog_bar = tqdm(range(nepoach))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            # logging.info(len(train_loader))
            for i, (idxs, inputs, targets) in enumerate(train_loader):
                idxs, inputs, targets = idxs.to(self._device), inputs.to(self._device), targets.long().to(self._device)
                functional.reset_net(self._network)
                self.optimizer.zero_grad()
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
                    else:
                        in_data, _ = torch.broadcast_tensors(inputs, torch.zeros((self.T,) + inputs.shape))
                    in_data = in_data.reshape(-1, *in_data.shape[2:])
                    output = self._network(in_data)
                    avg_fr = output if self.dynamic else output.mean(dim=0)
                if self.swap_ratio > 0 and self._cur_task > 0:
                    out_logit = output if data_manager.sp_policy == "hybrid_loss" else  output.mean(dim=0)
                    evaluator = torch.nn.CrossEntropyLoss(reduction='none')
                    data_manager.swap_loss = SwapEvaluator(evaluator, "rate" if self.rate_flag else "bptt", self.T, self.dynamic)
                    sw_idx, sw_target = data_manager.swap_determine(idxs, out_logit.detach(), targets.detach())
                    asyncio.run(train_loader.dataset._swap(sw_idx, sw_target))
                # logging.info(avg_fr.shape)
                # 检查 avg_fr 和 targets 的梯度状态
                # logging.info(f"avg_fr.requires_grad: {avg_fr.requires_grad}, targets.requires_grad: {targets.requires_grad}, Tis {self.T}")

                if self.dynamic:
                    loss = self.evaluator(avg_fr, targets, T = self.T, lamb = self.lamb, means = self.means)
                else:
                    loss = self.evaluator(avg_fr, targets)
                # 检查损失是否具有梯度
                # logging.info(f"loss.requires_grad: {loss.requires_grad}")
                # loss = F.cross_entropy(avg_fr, targets)
                loss.backward()
                self.optimizer.step()
                losses += loss.cpu().item()
                avg_fr = avg_fr.detach()
                if  self.dynamic:
                    _, preds = torch.max(avg_fr.mean(dim=0), dim=1)
                else:
                    _, preds = torch.max(avg_fr, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                # del inputs, targets, loss, output, avg_fr
            if self.scheduler is not None:
                self.scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    nepoach,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    nepoach,
                    losses / len(train_loader),
                    train_acc,
                )
            # info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
            #         self._cur_task,
            #         epoch + 1,
            #         nepoach,
            #         losses / len(train_loader),
            #         train_acc,
            #     )
            prog_bar.set_description(info)
        gc.collect()
        logging.info(info)

    def lowest_energy(self, observation,cutline=0.2):
     
        # adjusting the grid
        skipped_configs = self.tested[-1].count(-1) + np.count_nonzero(observation[0]==0)
        if skipped_configs != 0: 
            cutline = self.cutline * ((len(self.scores[-1])-skipped_configs)/len(self.scores[-1]))
        else: cutline = self.cutline

        thres = max(int(len(observation[0])*cutline),1)
        # f = open(self.log_file, 'a')
        filtered_configs_idxs = np.argsort(-observation[0,:])[:thres]
        filtered_configs = observation[:,filtered_configs_idxs]
        best_config_idx = filtered_configs_idxs[np.argsort(filtered_configs[1])[0]]

        # scores = self.scores[-1]
        # for i in range(len(observation[0])):
        #     if i in filtered_configs_idxs: scores[i] +=1 
        #     if i == best_config_idx: scores[i] +=1
        # logging.info(f'Raw Scores: {scores}')

        # for i in range(len(observation[0])):
        #     f.write(f'{self.task_id}, {self.grid[-1][i][0]}, {self.grid[-1][i][1]}, - ,,, {scores[i]}, {observation[0,i]}, {observation[1,i]}\n')
        # f.close()
        return best_config_idx
    def highest_accuracy(self, observation):
        best_config_idx = np.argsort(observation[0,:])[-1]
        # scores = [1 if i == best_config_idx else 0 for i in range(len(observation[0]))]
        # f = open(self.log_file, 'a')
        # for i in range(len(observation[0])):
        
        #     f.write(f'{self.task_id},{self.grid[-1][i][0]},{self.grid[-1][i][1]},-,-,{scores[i]},{observation[0,i]},{observation[1,i]}\n')
        # f.close()
        return best_config_idx
    def most_efficient(self, observation):
        cutline = self.cutline
        # f = open(self.log_file, 'a')
        filtered_configs_idxs = np.argsort(-observation[0,:])[:int(len(observation[0])*cutline)]
        filtered_configs = observation[:,filtered_configs_idxs]
        med_idx = np.argsort(-filtered_configs[0,:])[int(len(filtered_configs[0])/2)]
        [med_acc,med_energy] = filtered_configs[:,med_idx]
        cutline = observation[0,filtered_configs_idxs[0:-1]] 
        acc_terms = self.acc_coeff *np.exp((observation[0]/med_acc))
        
        energy_terms = self.energy_coeff*(1-np.log(observation[1]/med_energy))
        scores = [acc_terms[i]+energy_terms[i] if i in filtered_configs_idxs else 0 for i in range(len(observation[0]))]
        # for i in range(len(observation[0])):
        #     f.write(f'{self.task_id}, {self.grid[-1][i][0]}, {self.grid[-1][i][1]}, {acc_terms[i]}, {energy_terms[i]} , {scores[i]}, {observation[0,i]}, {observation[1,i]}\n')
        # f.close()
        return np.argsort(scores)[-1]
    def highest_utility(self, observation):
        # # adjusting the grid
        # skipped_configs = self.tested[-1].count(-1) + np.count_nonzero(observation[0]==0)
        # if skipped_configs != 0: 
        #     cutline = self.cutline * ((len(self.scores[-1])-skipped_configs)/len(self.scores[-1]))
        # else: cutline = self.cutline
        cutline = self.cutline
        thres = max(int(len(observation[0])*cutline),1)
        # f = open(self.log_file, 'a')
        filtered_configs_idxs = np.argsort(-observation[0,:])[:thres]
        # filtered_configs = observation[:,filtered_configs_idxs]

        scores = np.zeros(len(observation[0]))
        for i in range(len(observation[0])):
            if i in filtered_configs_idxs: scores[i] = observation[0,i]/observation[1,i]
        # logging.info(np.argsort(scores)[0])
        best_config_idx =  np.argsort(scores)[-1]

        # for i in range(len(observation[0])):
        #     f.write(f'{self.task_id}, {self.grid[-1][i][0]}, {self.grid[-1][i][1]}, - ,,, {scores[i]}, {observation[0,i]}, {observation[1,i]}\n')
        # f.close()
        return best_config_idx
    def score_functions(self, observation):
        scorers = {'most_efficient':self.most_efficient,
                    'lowest_energy': self.lowest_energy,
                    'highest_accuracy':self.highest_accuracy,
                    'highest_ETA': self.highest_utility
                     }
        return scorers[self.score_policy](observation)