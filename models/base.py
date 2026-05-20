import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
from spikingjelly.activation_based import functional
import os

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        if args["dataset"] == "dvs128swap" or args["dataset"] == "urbansoundswap" or args["dataset"] == "dsadsswap":
            self.topk = 1
        else:
            self.topk = 5

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self._splites = None
    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def build_rehearsal_memory_max(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar_max(data_manager, per_class)

    def build_rehearsal_memory_random(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar_random(data_manager, per_class)

    def build_rehearsal_memory_entropy(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar_entropy(data_manager, per_class)

    def build_rehearsal_memory_imblance(self, data_manager, per_class):
        self._construct_exemplar_imblance(data_manager, self._memory_per_class)

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self, save_conf=False):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        if save_conf:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
            _target_path = os.path.join(self.args['logfilename'], "target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

            _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
            os.makedirs(_save_dir, exist_ok=True)
            _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
            with open(_save_path, "a+") as f:
                f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")

        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                functional.reset_net(model)
                if self.step_mode == 's':
                    if self.dataset == 'dvs':
                        inputs = inputs.permute(1, 0, 2, 3, 4)
                    out_spikes = []
                    for t in range(self.T_max):
                        if self.dataset == 'dvs':
                            out = model(inputs[t])
                        else:
                            out = model(inputs)
                        out_spikes.append(out)
                    output = torch.stack(out_spikes, dim=0)
                    outputs = output.mean(dim=0)
                else:
                    if self.dataset == 'dvs':
                        in_data = inputs.permute(1, 0, 2, 3, 4)
                        # in_data = inputs.permute(1, 0, 2, 3, 4)
                    else:
                        in_data, _ = torch.broadcast_tensors(inputs, torch.zeros((self.T_max,) + inputs.shape))
                    in_data = in_data.reshape(-1, *in_data.shape[2:])
                    output = model(in_data)
                    outputs = output.mean(dim=0)
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == targets).sum()
                total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_snn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            functional.reset_net(self._network)
            with torch.no_grad():
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
                    outputs = output.mean(dim=0)
                else:
                    if self.dataset == 'dvs':
                        in_data = inputs.permute(1, 0, 2, 3, 4)
                    else:
                        in_data, _ = torch.broadcast_tensors(inputs, torch.zeros((self.T_max,) + inputs.shape))
                    in_data = in_data.reshape(-1, *in_data.shape[2:])
                    output = self._network(in_data)
                    outputs = output.mean(dim=0)
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device), self.dataset)
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device), self.dataset)
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _extract_res(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_res(_inputs.to(self._device), self.dataset)
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_res(_inputs.to(self._device), self.dataset)
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # # Exemplar mean
            # idx_dataset = data_manager.get_train_dataset(
            #     [],
            #     mode="test",
            #     appendent=(dd, dt),
            # )
            # idx_loader = DataLoader(
            #     idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            # )
            # vectors, _ = self._extract_vectors(idx_loader)
            # vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            # mean = np.mean(vectors, axis=0)
            # mean = mean / np.linalg.norm(mean)

            # self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_train_dataset(
                np.arange(class_idx, class_idx + 1),
                ret_data=True,
            )
            data = data[0]
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)
            if len(vectors) < m:
                a = m
                v = []
                d = []
                v.append(vectors)
                d.append(data)
                a = a - len(vectors)
                while len(vectors) < a:
                    v.append(vectors)
                    d.append(data)
                    a = a - len(vectors)
                v.append(vectors[:a])
                d.append(data[:a])
                vectors = np.concatenate(v)
                data = np.concatenate(d)
            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # # Exemplar mean
            # idx_dataset = data_manager.get_train_dataset(
            #     [],
            #     mode="test",
            #     appendent=(selected_exemplars, exemplar_targets),
            # )
            # idx_loader = DataLoader(
            #     idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            # )
            # vectors, _ = self._extract_vectors(idx_loader)
            # vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            # mean = np.mean(vectors, axis=0)
            # mean = mean / np.linalg.norm(mean)

            # self._class_means[class_idx, :] = mean

    def _construct_exemplar_max(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_train_dataset(
                np.arange(class_idx, class_idx + 1),
                ret_data=True,
            )
            data = data[0]
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)
            if len(vectors) < m:
                a = m
                v = []
                d = []
                v.append(vectors)
                d.append(data)
                a = a - len(vectors)
                while len(vectors) < a:
                    v.append(vectors)
                    d.append(data)
                    a = a - len(vectors)
                v.append(vectors[:a])
                d.append(data[:a])
                vectors = np.concatenate(v)
                data = np.concatenate(d)
            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmax(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # # Exemplar mean
            # idx_dataset = data_manager.get_train_dataset(
            #     [],
            #     mode="test",
            #     appendent=(selected_exemplars, exemplar_targets),
            # )
            # idx_loader = DataLoader(
            #     idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            # )
            # vectors, _ = self._extract_vectors(idx_loader)
            # vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            # mean = np.mean(vectors, axis=0)
            # mean = mean / np.linalg.norm(mean)

            # self._class_means[class_idx, :] = mean

    def _construct_exemplar_random(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_train_dataset(
                np.arange(class_idx, class_idx + 1),
                ret_data=True,
            )
            data, targets, idx_dataset = data_manager.get_train_dataset(
                np.arange(class_idx, class_idx + 1),
                ret_data=True,
                space = [m] * self._total_classes,
                policy = 'random'
            )
            data = data[0]

            # Select
            selected_exemplars = np.array(data)
            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(data)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # # Exemplar mean
            # idx_dataset = data_manager.get_train_dataset(
            #     [],
            #     mode="test",
            #     appendent=(selected_exemplars, exemplar_targets),
            # )
            # idx_loader = DataLoader(
            #     idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            # )
            # vectors, _ = self._extract_vectors(idx_loader)
            # vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            # mean = np.mean(vectors, axis=0)
            # mean = mean / np.linalg.norm(mean)

            # self._class_means[class_idx, :] = mean

    def _construct_exemplar_imblance(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        class_num = self._total_classes - self._known_classes
        proportions = np.random.rand(class_num)
        proportions /= proportions.sum()  # 归一化，使比例之和为1
        m_sum = m * class_num - class_num * 1
        splits = np.full(class_num, 1)
        # 根据比例拆分值并四舍五入为整数
        add_splits = np.round(proportions * m_sum).astype(int)
        # 调整总和以确保与原值一致
        while sum(add_splits) != m_sum:
            diff = m_sum - sum(add_splits)
            idx = np.random.choice(class_num)
            add_splits[idx] += diff
        splits += add_splits

        if self._splites is None:
            self._splites = np.array(splits)
        else:
            self._splites = np.concatenate((self._splites, np.array(splits)))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_train_dataset(
                np.arange(class_idx, class_idx + 1),
                ret_data=True,
            )
            data, targets, idx_dataset = data_manager.get_train_dataset(
                np.arange(class_idx, class_idx + 1),
                ret_data=True,
                space = self._splites,
                policy = 'random'
            )
            data = data[0]
            # print(data.shape)
            # Select
            selected_exemplars = np.array(data)
            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

    def _construct_exemplar_entropy(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_train_dataset(
                np.arange(class_idx, class_idx + 1),
                ret_data=True,
            )
            data = data[0]
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_res(idx_loader)
            if len(vectors) < m:
                a = m
                v = []
                d = []
                v.append(vectors)
                d.append(data)
                a = a - len(vectors)
                while len(vectors) < a:
                    v.append(vectors)
                    d.append(data)
                    a = a - len(vectors)
                v.append(vectors[:a])
                d.append(data[:a])
                vectors = np.concatenate(v)
                data = np.concatenate(d)
            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            EPSILON = 1e-8  # 避免 log(0) 的问题
            entropies = []
            
            for vector in vectors:
                # 将向量归一化为概率分布
                # probabilities = vector / np.sum(vector)
                exp_x = np.exp(vector - np.max(vector))
                probabilities = exp_x / np.sum(exp_x)
                probabilities = np.clip(probabilities, EPSILON, 1)  # 限制范围避免数值问题
                # 计算熵
                entropy = -np.sum(probabilities * np.log(probabilities)) / np.log(len(probabilities))
                entropies.append(entropy)

            ent = np.array(entropies)
            idxs = np.argpartition(-ent, m)[:m]
            # print(ent)
            # print(idxs)
            selected_exemplars = data[idxs]
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # # Exemplar mean
            # idx_dataset = data_manager.get_train_dataset(
            #     [],
            #     mode="test",
            #     appendent=(selected_exemplars, exemplar_targets),
            # )
            # idx_loader = DataLoader(
            #     idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            # )
            # vectors, _ = self._extract_vectors(idx_loader)
            # vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            # mean = np.mean(vectors, axis=0)
            # mean = mean / np.linalg.norm(mean)

            # self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means
