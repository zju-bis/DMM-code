import logging
import numpy as np
import asyncio
import random
import math
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCIFARSWAP, iImageNetSwap, NCaltech101Swap, DVS128Swap, UrbanSound8KSwap, DSADSSwap
from tqdm import tqdm
import os
import io
import directio
import json
class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, threshold = 1.0, swap_policy = None):
        self.dataset_name = dataset_name
        if dataset_name != "cifarswap" and dataset_name != "imagenetswap" and dataset_name != "ncaltech101swap" and dataset_name != "dvs128swap" and dataset_name != "urbansoundswap" and dataset_name != "dsadsswap":
            self._setup_data(dataset_name, shuffle, seed)
        elif dataset_name == "ncaltech101swap":
            self.orders = 101
        elif dataset_name == "dvs128swap":
            self.orders = 11
        elif dataset_name == "urbansoundswap":
            self.orders = 10
        elif dataset_name == "dsadsswap":
            self.orders = 19
        else:
            self.orders = 100
        assert init_cls <= self.orders, "No enough classes."
        self.rp_len = 0
        self.sp_policy = swap_policy
        self.swap_determine = self.swap_policy(swap_policy)
        self.swap_class_dist = {}
        self._swap_loss = None
        self.threshold = threshold
        self.softmax = torch.nn.Softmax(dim=1)
        self._get_loss = False
        self._get_entropy = False
        self._increments = [init_cls]
        # increment = random.randint(2, 20)
        while sum(self._increments) + increment < self.orders:
            self._increments.append(increment)
            # increment = random.randint(2, 20)
        offset = self.orders - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)
        if (self.dataset_name == 'cifarswap'):
            self.num_per_class = [500] * 100
            self.stream_space_perclass = 500
        if (self.dataset_name == 'imagenetswap'):
            self.num_per_class = [1300] * 100
            self.stream_space_perclass = 1300
        elif (self.dataset_name == 'ncaltech101swap'):
            with open("data/Caltech101/frames_number_8_split_by_number/len_per_class.json", "r") as f:
                self.num_per_class = json.load(f)
            self.stream_space_perclass = 1000
        elif (self.dataset_name == 'dvs128swap'):
            with open("data/DVS128Gesture/frames_number_8_split_by_number/train/len_per_class.json", "r") as f:
                self.num_per_class = json.load(f)
            self.stream_space_perclass = 1000
        elif (self.dataset_name == 'urbansoundswap'):
            with open("data/UrbanSound8K/npy/len_per_class.json", "r") as f:
                self.num_per_class = json.load(f)
            self.stream_space_perclass = 1000
        elif (self.dataset_name == 'dsadsswap'):
            self.num_per_class = [384] * 19
            self.stream_space_perclass = 1000
    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]
    
    def get_accumulate_tasksize(self,task):
        return sum(self._increments[:task+1])
    
    def get_total_classnum(self):
        return len(self._class_order)

    def reset_swap_class_dist(self):
        self.swap_class_dist = {}

    def swap_policy(self, swap_base):
        policies = {
            "entropy" : self.entropy,
            "hybrid_threshold" : self.hybrid,
            "prediction" : self.prediction,
            "random" : self.random,
            "random_fixed": self.random_fixed,
            "pure_random" : self.pure_random,
            "opposite": self.hybrid_opposite,
            "hybrid_ratio" : self.hybrid_ratio,
            "hybrid_balanced" : self.hybrid_balanced,
            "hybrid_balanced_p" : self.hybrid_balanced_p,
            #"hybrid_random" : self.hybrid_random,
            "hybrid_loss" : self.hybrid_loss,
            "all" : self.all
        }
        if swap_base is None:
            return self.random
        else : return policies[swap_base]

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []


        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def stream_resize(self, size, task_id):
        self.stream_space_perclass = size // self._increments[task_id]

    def get_train_dataset(
        self, indices, appendent=None, ret_data=False, policy='back', space = None, swap_ratio = 0.0, mode = "train", noise_prec = 0.0
    ):
        idata = _get_idata(self.dataset_name)
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf
        if policy == 'back':
            idata.load_data(indices, int(self.stream_space_perclass), self.num_per_class, policy= policy)
        else:
            idata.load_data(indices, space, self.num_per_class, policy= policy)
        x, y = idata.train_data, idata.train_targets
        if mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])


        data, targets = [], []

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            self.rp_len = len(appendent_data)
            # print("appendent_data len", len(appendent_data))
            # print("appendent_targets len", len(appendent_targets))
            data.append(appendent_data)
            targets.append(appendent_targets)

        data.append(x)
        targets.append(y)

        if ret_data:
            return data, targets, SwapDataset(data, targets, trsf, self.num_per_class, idata.train_path, idata.label_to_class, swap_ratio, noise_prec)
        else:
            return SwapDataset(data, targets, trsf, self.num_per_class, idata.train_path, idata.label_to_class, swap_ratio, noise_prec)

    def get_test_dataset(
        self, indices, appendent=None, ret_data=False
    ):
        idata = _get_idata(self.dataset_name)
        self.use_path = idata.use_path
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf
        idata.load_test_data(indices)
        x, y = idata.test_data, idata.test_targets

        trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])


        data, targets = [], []

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data.append(x)
        targets.append(y)
        # data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, SwapDataset(data, targets, trsf, train_path = idata.train_path)
        else:
            return SwapDataset(data, targets, trsf, train_path = idata.train_path)

    def get_finetune_dataset(self,known_classes,total_classes,source,mode,appendent,type="ratio"):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))
        val_data = []
        val_targets = []

        old_num_tot = 0
        appendent_data, appendent_targets = appendent

        for idx in range(0, known_classes):
            append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                       low_range=idx, high_range=idx+1)
            num=len(append_data)
            if num == 0:
                continue
            old_num_tot += num
            val_data.append(append_data)
            val_targets.append(append_targets)
        if type == "ratio":
            new_num_tot = int(old_num_tot*(total_classes-known_classes)/known_classes)
        elif type == "same":
            new_num_tot = old_num_tot
        else:
            assert 0, "not implemented yet"
        new_num_average = int(new_num_tot/(total_classes-known_classes))
        for idx in range(known_classes,total_classes):
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            val_indx = np.random.choice(len(class_data),new_num_average, replace=False)
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
        val_data=np.concatenate(val_data)
        val_targets = np.concatenate(val_targets)
        return DummyDataset(val_data, val_targets, trsf, self.use_path)

    def get_train_val_dataset_with_split(
        self, indices, appendent=None, val_samples_per_class=0, swap_ratio = 0.0
    ):
        idata = _get_idata(self.dataset_name)
        self._train_trsf = idata.train_trsf
        self._common_trsf = idata.common_trsf
        
        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            idx = [idx]
            idata.load_data(idx, int(self.stream_space_perclass), self.num_per_class, policy= "back")
            class_data, class_targets = idata.train_data, idata.train_targets
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])
        
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)
        train_data, train_targets = np.concatenate(train_data), np.concatenate(train_targets)

        trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])


        data, targets = [], []
        data_val, targets_val = [], []
        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                data_val.append(append_data[val_indx])
                targets_val.append(append_targets[val_indx])
                data.append(append_data[train_indx])
                targets.append(append_targets[train_indx])

        data_val, targets_val = [np.concatenate(data_val)], [np.concatenate(targets_val)]
        data, targets = [np.concatenate(data)], [np.concatenate(targets)]
        data.append(train_data)
        targets.append(train_targets)
        data_val.append(val_data)
        targets_val.append(val_targets)

        return SwapDataset(data, targets, trsf, self.num_per_class, idata.train_path, idata.label_to_class, swap_ratio), SwapDataset(data, targets, trsf, self.num_per_class, idata.train_path, idata.label_to_class, swap_ratio)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        self.orders = len(self._class_order)
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        
        if isinstance(x,np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        return x_return, y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))
    @property
    def swap_thr(self):
        return self._swap_thr
    
    @swap_thr.setter
    def swap_thr(self, thr):
        self._swap_thr = thr

    
    @property
    def swap_loss(self):
        return self._swap_loss
    
    @swap_loss.setter
    def swap_loss(self, loss):
        self._swap_loss = loss
    
    def to_onehot(self, targets, n_classes):
        onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
        onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
        return onehot

    def get_replay_index(self, idxs, targets, data_ids=None):

        if self.rp_len is None:
            if data_ids is not None:
                return idxs, targets, data_ids
            else:
                return idxs, targets

        else:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
            
            if data_ids is not None:
                return idxs[replay_index_of_idxs], targets[replay_index_of_idxs], data_ids[replay_index_of_idxs]
            else:
                return idxs[replay_index_of_idxs], targets[replay_index_of_idxs]

    def prediction(self, idxs, outputs, targets, data_ids=None):
        #
        # determine what to swap based on mis-prediction
        #
        predicts = torch.max(outputs, dim=1)[1]
        selected_idx = (predicts.cpu() == targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        if data_ids is not None:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx], data_ids[selected_idx])
        else:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx])

    def entropy(self, idxs, outputs, targets, data_ids=None):
        #
        # determine what to swap based on entropy (threshold = 1.0 : lower is easy and swap, higher is hard and preserve)
        #
        # print("idxs is ", idxs, "rp_len is ", self.rp_len)
        soft_output = self.softmax(outputs)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        selected_idx = (entropy.cpu() < self.threshold).squeeze().nonzero(as_tuple=True)[0]

        if data_ids is not None:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx], data_ids[selected_idx])
        else:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx])


    def hybrid(self, idxs, outputs, targets, data_ids=None):
        #
        # determine what to swap based on entropy (threshold : lower is easy and swap, higher is hard and preserve)
        #        
        #print(idxs,outputs, targets)
        soft_output = self.softmax(outputs)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        entropy_batch = (entropy.cpu() < self.threshold).squeeze()
        #
        # if wrong predicted sample with low entropy, don't make it swap (make swap FALSE)
        #
        predicts = torch.max(outputs, dim=1)[1]
        prediction_batch = (predicts.cpu() == targets.cpu()).squeeze()

        selected_idx = (torch.logical_and(entropy_batch, prediction_batch)).nonzero(as_tuple=True)[0]
        if data_ids is not None:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx], data_ids[selected_idx])
        else:
            return self.get_replay_index(idxs[selected_idx], targets[selected_idx])

    #@profile
    
    def random_fixed(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold
        #print("SWAP RATIO : ", swap_ratio)
        
        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)
        
        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))
        
        replay_output = outputs[replay_index_of_idxs]
        replay_idxs = idxs[replay_index_of_idxs]
        replay_targets = targets[replay_index_of_idxs]
        
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs]
        
        if data_ids is not None:
            return replay_idxs[:how_much_swap], replay_targets[:how_much_swap], replay_data_ids[:how_much_swap]
        else:
            return replay_idxs[:how_much_swap], replay_targets[:how_much_swap]
            

    def random(self, idxs, outputs, targets, data_ids=None):
        
        
        swap_ratio = self.threshold
        #print("SWAP RATIO : ", swap_ratio)
        
        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)
        
        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))
        
        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()
        #print("replay len in batch : ", len(replay_index_of_idxs))
        #print("how much swap : ", how_much_swap)
        
        self.get_loss(replay_output, replay_targets)
        self.get_entropy(replay_output, replay_targets)
        
        selected_index = np.random.choice(len(replay_idxs), how_much_swap, replace=False)

        assert len(selected_index) == how_much_swap

        
        if data_ids is not None:
            return replay_idxs[selected_index], replay_targets[selected_index], replay_data_ids[selected_index]
        else:
            return replay_idxs[selected_index], replay_targets[selected_index]
    
    
    def pure_random(self, idxs, outputs, targets, data_ids=None):

        swap_ratio = self.threshold
        #print("SWAP RATIO : ", swap_ratio)

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        idx_to_pick = np.linspace(0, len(replay_index_of_idxs), num=how_much_swap, dtype = int, endpoint=False)
        #print("init idx_to_pick : ", idx_to_pick)

        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        torch.set_printoptions(precision=4,sci_mode=False)
        
        soft_output = self.softmax(replay_output)

        torch.set_printoptions(precision=4,sci_mode=False)

        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        torch.set_printoptions(precision=4,sci_mode=False)

        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]
    
            r_range = np.where(idx_to_pick < len(r_idxs))
            sorted_r = torch.argsort(r_entropy)[idx_to_pick[r_range]]
            #print("idx_to_pick for right precition : ", idx_to_pick[r_range])

            
            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]


            if len(sorted_r) < how_much_swap:
                #print("wrong_prediction count...")
                idx_to_pick = np.delete(idx_to_pick, r_range)
                #print("idx_to_pick before subtract len : ", idx_to_pick)
                #print("len of replay samples in this batch : ", len(r_idxs))
                idx_to_pick = idx_to_pick - len(r_idxs)
                #print("idx_to_pick after subtract len : ", idx_to_pick)
                #print("\n\n")

                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]

                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]
                
                sorted_w = torch.argsort(w_entropy, descending=True)[idx_to_pick]


                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]


                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)
        

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids


        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        #print("selected_idxs : ", selected_idxs)
        #print("selected_targets : ", selected_targets)
        #print("\n")

        
        assert len(selected_idxs) == how_much_swap

        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets


    def hybrid_opposite(self, idxs, outputs, targets, data_ids=None):

        swap_ratio = self.threshold
        #print("SWAP RATIO : ", swap_ratio)


        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        #print("idxs : ",idxs)
        #print("targets : ", targets)

        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        #print("replay_index_of_idxs : ", replay_index_of_idxs)
        #print("replay idxs : ", replay_idxs)
        #print("replay_targets : ", replay_targets)
        #print("entropy : ", entropy)
        
        #print("entropy size : ", entropy.shape)
        
        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
            #print("w_pred(idx) : " , w_predicted)
            #print("w_pred size : ", w_predicted.shape)

            w_idxs = replay_idxs[w_predicted]
            w_entropy = entropy[w_predicted]
            w_targets = replay_targets[w_predicted]
            if data_ids is not None:
                w_data_ids = replay_data_ids[w_predicted]

            #print("w_idxs : ", w_idxs)
            #print("w_targets : ", w_targets)
            #print("w_entropy : ", w_entropy)
            
            sorted_w = torch.argsort(w_entropy)[:how_much_swap].to(w_idxs.device)

            #print("sorted_w : ", sorted_w)

            selected_w_idxs = w_idxs[sorted_w]
            selected_w_targets = w_targets[sorted_w]
            if data_ids is not None:
                selected_w_data_ids = w_data_ids[sorted_w]

            #print("selected_w_idxs : ", selected_w_idxs)
            #print("selected_w_targets : ", selected_w_targets)

            if len(sorted_w) < how_much_swap:

                r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                r_idxs = replay_idxs[r_predicted]
                r_entropy = entropy[r_predicted]
                r_targets = replay_targets[r_predicted]
                    
                if data_ids is not None:
                    r_data_ids = replay_data_ids[r_predicted]

                #print("r_pred(idx) : " , r_predicted)
                #print("r_idxs : ", r_idxs)
                #print("r_targets : ", r_targets)
                #print("r_entropy : ", r_entropy)
                
                sorted_r = torch.argsort(r_entropy)[-(how_much_swap-len(sorted_w)):]

                #print("sorted_r : ", sorted_r)

                selected_r_idxs = r_idxs[sorted_r]
                selected_r_targets = r_targets[sorted_r]
                    
                if data_ids is not None:
                    selected_r_data_ids = r_data_ids[sorted_r]

                    
                #print("selected_r_idxs : ", selected_r_idxs)
                #print("selected_r_targets : ", selected_r_targets)

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_w_idxs
                selected_targets = selected_w_targets
                if data_ids is not None:
                    selected_data_ids = selected_w_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        #print("selected_idxs : ", selected_idxs)
        #print("selected_targets : ", selected_targets)
        #print("\n")
        
        assert len(selected_idxs) == how_much_swap


        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets

    def hybrid_ratio(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold
        #print("SWAP RATIO : ", swap_ratio)

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        #print("total batch len : ", len(idxs))
        #print("replay batch len : ", replay_index_of_idxs)
        #print("how_much_swap : ", how_much_swap)


        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        #print("replay_index_of_idxs : ", replay_index_of_idxs)
        #print("replay idxs : ", replay_idxs)
        #print("replay_targets : ", replay_targets)
        #print("entropy : ", entropy)
        
        #print("entropy size : ", entropy.shape)
        
        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
            #print("r_pred(idx) : " , r_predicted)
            #print("r_pred size : ", r_predicted.shape)

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]

            #print("r_idxs : ", r_idxs)
            #print("r_targets : ", r_targets)
            #print("r_entropy : ", r_entropy)
            
            sorted_r = torch.argsort(r_entropy)[:how_much_swap]

            #print("sorted_r : ", sorted_r)

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            #print("selected_r_idxs : ", selected_r_idxs)
            #print("selected_r_targets : ", selected_r_targets)

            if len(sorted_r) < how_much_swap:

                #print("== WE NEED MORE SAMPLE TO SWAP EVEN IF ITS WRONG PRED!!")
                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                #print("w_pred(idx) : " , w_predicted)
                #print("w_idxs : ", w_idxs)
                #print("w_targets : ", w_targets)
                #print("w_entropy : ", w_entropy)
                
                w_how_much_swap = how_much_swap-len(sorted_r)
                sorted_w = torch.argsort(w_entropy, descending=True)[:w_how_much_swap]

                #print("sorted_w : ", sorted_w)

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                    
                #print("selected_w_idxs : ", selected_w_idxs)
                #print("selected_w_targets : ", selected_w_targets)

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        assert len(selected_idxs) == how_much_swap

        #print("selected_targets : ", selected_targets)
        #print("\n")

        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets

    def get_entropy(self, outputs, targets):
        
        if self._get_entropy == False:
            return

        print("GET ENTROPY IS CALLED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        soft_output = self.softmax(outputs)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        #
        # if wrong predicted sample with low entropy, don't make it swap (make swap FALSE)
        #
        predicts = torch.max(outputs, dim=1)[1]
        r_predicted = (predicts.cpu() == targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        r_entropy = entropy[r_predicted]
            
        self.data_correct_entropy.extend(r_entropy.tolist())

        w_predicted = (predicts.cpu() != targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        w_entropy = entropy[w_predicted]
            
        self.data_wrong_entropy.extend(w_entropy.tolist())


    def get_loss(self, outputs, targets):

        if self._get_loss == False:
            return
        
        print("GET LOSS IS CALLED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        try:
            loss = self.swap_loss(outputs, targets)
        except ValueError:
            #print(outputs.shape)
            targets_one_hot = self.to_onehot(targets, outputs.shape[1])
            loss = self.swap_loss(outputs, targets_one_hot)

            loss = loss.view(loss.size(0), -1)
            loss = loss.mean(-1)
        #
        # if wrong predicted sample with low entropy, don't make it swap (make swap FALSE)
        #
        predicts = torch.max(outputs, dim=1)[1]

        #print(loss.shape, outputs.shape, predicts.shape)

        r_predicted = (predicts.cpu() == targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        r_loss = loss[r_predicted]
            
        self.data_correct_loss.extend(r_loss.tolist())

        w_predicted = (predicts.cpu() != targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        w_loss = loss[w_predicted]
            
        self.data_wrong_loss.extend(w_loss.tolist())


    def hybrid_loss(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold
        mean_outputs = outputs.mean(dim=0)
        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        total_how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))
        # print("id len", len(replay_index_of_idxs))
        # print("rp_len", self.rp_len)
        #print("total batch len : ", len(idxs))
        #print("replay batch len : ", replay_index_of_idxs)
        #print("how_much_swap : ", how_much_swap)

        replay_out = outputs[:, replay_index_of_idxs]
        #get the number of each class inside the batch
        batch_dist = {}
        replay_output = mean_outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        
        self.get_loss(replay_output, replay_targets)
        self.get_entropy(replay_output, replay_targets)
        

        for cls in replay_targets:
            if cls.item() not in batch_dist:
                batch_dist[cls.item()] = 1
            else:
                batch_dist[cls.item()] += 1

        #print("BEFORE select batch_dist : ", batch_dist)
        expected = 0
        for key in batch_dist.keys():
            batch_dist[key] = math.modf( batch_dist[key] * swap_ratio )
            expected += int(batch_dist[key][1])

        shortage = total_how_much_swap - expected
        #print(total_how_much_swap, expected)

        #print(shortage)

        if shortage > 0:
            get_dec_samples = list(filter(lambda x: x[1][0] != 0, batch_dist.items()))
            selected_dec_samples = random.sample(get_dec_samples, shortage)
            
            for t in selected_dec_samples:
                k, _ = t
                batch_dist[k] = tuple((batch_dist[k][0] ,batch_dist[k][1] + 1))
        
        #print("BEFORE : ", batch_dist)

        if replay_output.nelement() == 0:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)
            
        #print("batch_dist {k : (decimal, how_much_swap_for_this_class)} : ", batch_dist)
        #print("total swap num : ", total_how_much_swap)
        #separate batch into class-wise
        for i, (k,v) in enumerate(batch_dist.items()):
            #print("current class.. ", k)
            #print("how much swap in current class... ", int(v[1]))
            how_much_swap = int(v[1])

            cur_cls_idx = (replay_targets==k).squeeze().nonzero(as_tuple=True)[0]
            
            cur_replay_output = replay_output[cur_cls_idx]
            cur_replay_idxs = replay_idxs[cur_cls_idx]
            cur_replay_targets = replay_targets[cur_cls_idx]            
            if data_ids is not None:
                cur_replay_data_ids = replay_data_ids[cur_cls_idx]
            
            
            try:
                loss = self.swap_loss(replay_out, replay_targets).cpu()
            except ValueError:
                #print(outputs.shape)
                targets_one_hot = self.to_onehot(targets, outputs.shape[1])
                loss = self.swap_loss(outputs, targets_one_hot).cpu()
                loss = loss.view(loss.size(0), -1)
                loss = loss.mean(-1)
            
            # print("LOSS : ", loss, loss.shape)
            # print("target : ", targets, targets.shape)
            #print("replay_index_of_idxs : ", replay_index_of_idxs)
            #print("replay idxs : ", replay_idxs)
            #print("replay_targets : ", replay_targets)
            #print("entropy : ", entropy)
            #print("entropy size : ", entropy.shape)
            
            predicts = torch.max(cur_replay_output, dim=1)[1]
            # print("r_pred size : ", cur_replay_output.shape)
            r_predicted = (predicts.cpu() == cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
            #print("r_pred(idx) : " , r_predicted)
            # print("r_pred size : ", r_predicted.shape)

            r_idxs = cur_replay_idxs[r_predicted]
            r_loss = loss[r_predicted]
            r_targets = cur_replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = cur_replay_data_ids[r_predicted]

            #print("r_idxs : ", r_idxs)
            #print("r_targets : ", r_targets)
            #print("r_loss : ", r_loss)
            
            sorted_r = torch.argsort(r_loss)[:how_much_swap]
            #print("how_much_swap : ", how_much_swap)
            #print("sorted_r : ", sorted_r)

            #to check this code validate
            selected_r_loss = r_loss[sorted_r]

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            #print("selected_r_idxs : ", selected_r_idxs)
            #print("selected_r_targets : ", selected_r_targets)

            if len(sorted_r) < how_much_swap:

                #print("== WE NEED MORE SAMPLE TO SWAP EVEN IF ITS WRONG PRED!!")
                w_predicted = (predicts.cpu() != cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = cur_replay_idxs[w_predicted]
                w_loss = loss[w_predicted]

                w_targets = cur_replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                #print("w_pred(idx) : " , w_predicted)
                #print("w_idxs : ", w_idxs)
                #print("w_targets : ", w_targets)
                #print("w_loss : ", w_loss.shape)

                #print("w_idxs : ", w_idxs)
                #print("w_targets : ", w_targets)
                #print("w_loss : ", w_loss)
                
                w_how_much_swap = how_much_swap-len(sorted_r)
                #######
                sorted_w = torch.argsort(w_loss)[:w_how_much_swap]

                #print("sorted_w : ", sorted_w)
                #######
                ####### sorted_w = torch.argsort(w_entropy)[:w_how_much_swap]
                #######
                ####### sorted_w = np.random.choice(len(w_entropy), w_how_much_swap, replace=False)
                
                #print("sorted_w : ", sorted_w)

                #to check this code validate
                selected_w_loss = w_loss[sorted_w]

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                    
                #print("selected_w_idxs : ", selected_w_idxs)
                #print("selected_w_targets : ", selected_w_targets)

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids

        
            if i==0:                
                total_selected_idxs = selected_idxs
                total_selected_targets = selected_targets
                if data_ids is not None:
                    total_selected_data_ids = selected_data_ids

            else:
                total_selected_idxs = torch.cat((total_selected_idxs,selected_idxs),dim=-1)
                total_selected_targets = torch.cat((total_selected_targets,selected_targets),dim=-1)
                if data_ids is not None:
                    total_selected_data_ids = torch.cat((total_selected_data_ids,selected_data_ids),dim=-1)

        
        assert len(total_selected_idxs) == total_how_much_swap

        if data_ids is not None:
            return total_selected_idxs, total_selected_targets, total_selected_data_ids
        else:
            return total_selected_idxs, total_selected_targets


    def hybrid_balanced_p(self, idxs, outputs, targets, data_ids=None):
        
        #print("\n")
        swap_ratio = self.threshold

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        total_how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))


        #print("total_how_much_swap : ", total_how_much_swap)


        #get the number of each class inside the batch
        batch_dist = {}
        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()
        
        for cls in replay_targets:
            if cls.item() not in batch_dist:
                batch_dist[cls.item()] = 1
            else:
                batch_dist[cls.item()] += 1

        #print("BEFORE : ", batch_dist)
        expected = 0
        for key in batch_dist.keys():
            batch_dist[key] = math.modf( batch_dist[key] * swap_ratio )
            expected += int(batch_dist[key][1])

        shortage = total_how_much_swap - expected
        #print("shortage : ", shortage)

        #print(shortage)

        if shortage > 0:
            get_dec_samples = list(filter(lambda x: x[1][0] != 0, batch_dist.items()))
            selected_dec_samples = random.sample(get_dec_samples, shortage)
            
            #p = [w[0] for (_,w) in get_dec_samples]
            #l = [k for (k,_) in get_dec_samples]
            
            #selected_dec_samples = np.random.choice(l, size=shortage, replace=False, p=[i/sum(p) for i in p] ) #balanced_p_v2

            
            #print(selected_dec_samples)

            for t in selected_dec_samples:
                k, _ = t
                #k = t
                batch_dist[k] = tuple((batch_dist[k][0] ,batch_dist[k][1] + 1))
        
        #print("AFTER : ", batch_dist)

        if replay_output.nelement() == 0:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        #separate batch into class-wise
        for i, (k,v) in enumerate(batch_dist.items()):

            #print("current class : ", k)
            how_much_swap = int(v[1])
            if how_much_swap == 0:
                continue
            
            
            cur_cls_idx = (replay_targets==k).squeeze().nonzero(as_tuple=True)[0]
            
            cur_replay_output = replay_output[cur_cls_idx]
            cur_replay_idxs = replay_idxs[cur_cls_idx]
            cur_replay_targets = replay_targets[cur_cls_idx]            
            if data_ids is not None:
                cur_replay_data_ids = replay_data_ids[cur_cls_idx]
            
            soft_output = self.softmax(cur_replay_output)
            entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
            #print("cur_cls_idx : ", cur_cls_idx)
            #print("cur_replay_idxs : ", cur_replay_idxs)
            #print("cur_replay_targets : ", cur_replay_targets)
            #print("entropy : ", entropy)
            #print("entropy size : ", entropy.shape)
            
            predicts = torch.max(cur_replay_output, dim=1)[1]
            #print("prediction : ", predicts)
            r_predicted = (predicts.cpu() == cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
            #print("r_pred(idx) : " , r_predicted)
            #print("r_pred size : ", r_predicted.shape)

            r_idxs = cur_replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = cur_replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = cur_replay_data_ids[r_predicted]

            #print("r_idxs : ", r_idxs)
            #print("r_targets : ", r_targets)
            #print("r_entropy : ", r_entropy)
            sorted_r = torch.argsort(r_entropy)[:how_much_swap]


            #to check this code validate
            selected_r_entropy = r_entropy[sorted_r]


            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            #print("selected_r_idxs : ", selected_r_idxs)
            #print("selected_r_targets : ", selected_r_targets)

            if len(sorted_r) < how_much_swap:

                #print("== WE NEED MORE SAMPLE TO SWAP EVEN IF ITS WRONG PRED!!")
                w_predicted = (predicts.cpu() != cur_replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = cur_replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = cur_replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                #print("w_pred(idx) : " , w_predicted)
                #print("w_idxs : ", w_idxs)
                #print("w_targets : ", w_targets)
                #print("w_entropy : ", w_entropy)
                
                w_how_much_swap = how_much_swap-len(sorted_r)
                #######
                sorted_w = torch.argsort(w_entropy, descending=True)[:w_how_much_swap]
                #######
                ####### sorted_w = torch.argsort(w_entropy)[:w_how_much_swap]
                #######
                ####### sorted_w = np.random.choice(len(w_entropy), w_how_much_swap, replace=False)
                
                #print("sorted_w : ", sorted_w)

                #to check this code validate
                selected_w_entropy = w_entropy[sorted_w]

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                    
                #print("selected_w_idxs : ", selected_w_idxs)
                #print("selected_w_targets : ", selected_w_targets)

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids

            try:
                total_selected_idxs = torch.cat((total_selected_idxs,selected_idxs),dim=-1)
                total_selected_targets = torch.cat((total_selected_targets,selected_targets),dim=-1)
                if data_ids is not None:
                    total_selected_data_ids = torch.cat((total_selected_data_ids,selected_data_ids),dim=-1)

            except:                
                total_selected_idxs = selected_idxs
                total_selected_targets = selected_targets
                if data_ids is not None:
                    total_selected_data_ids = selected_data_ids

        assert len(total_selected_idxs) == total_how_much_swap

        if data_ids is not None:
            return total_selected_idxs, total_selected_targets, total_selected_data_ids
        else:
            return total_selected_idxs, total_selected_targets



    def hybrid_balanced(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold * 0.4
        #print("SWAP RATIO : ", swap_ratio)

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))

        #print("total batch len : ", len(idxs))
        #print("replay batch len : ", replay_index_of_idxs)
        #print("how_much_swap : ", how_much_swap)


        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        #print("replay_index_of_idxs : ", replay_index_of_idxs)
        #print("replay idxs : ", replay_idxs)
        #print("replay_targets : ", replay_targets)
        #print("entropy : ", entropy)
        
        #print("entropy size : ", entropy.shape)
        
        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
            #print("r_pred(idx) : " , r_predicted)
            #print("r_pred size : ", r_predicted.shape)

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]

            #print("r_idxs : ", r_idxs)
            #print("r_targets : ", r_targets)
            #print("r_entropy : ", r_entropy)
            
            sorted_r_org = torch.argsort(r_entropy)

            selected = []
            filled_counter = 0

            for i, idx in enumerate(sorted_r_org):

                if filled_counter >= how_much_swap:
                    break

                label = r_targets[idx].item()
                if label in self.swap_class_dist:
                    if self.swap_class_dist[label] + 1 <= self.swap_thr:
                        self.swap_class_dist[label] += 1
                        filled_counter +=1
                        selected.append(i)
                    else:
                        continue
                else:
                    self.swap_class_dist[label] = 1
                    filled_counter +=1
                    selected.append(i)
                #print("LABEL : ", label)
                #print("class dist : ", self.swap_class_dist)
                #print("filled_count, how_much_swap : ", filled_counter, how_much_swap)

            sorted_r = sorted_r_org[selected][:how_much_swap]

            #print("sorted_r : ", sorted_r)

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            #print("selected_r_idxs : ", selected_r_idxs)
            #print("selected_r_targets : ", selected_r_targets)



            if len(sorted_r) < how_much_swap:

                #print("== WE NEED MORE SAMPLE TO SWAP EVEN IF ITS WRONG PRED!!")
                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                #print("w_pred(idx) : " , w_predicted)
                #print("w_idxs : ", w_idxs)
                #print("w_targets : ", w_targets)
                #print("w_entropy : ", w_entropy)
                
                w_how_much_swap = how_much_swap-len(sorted_r)
                sorted_w_org = torch.argsort(w_entropy, descending=True)


                selected = []
                filled_counter = 0

                for i, idx in enumerate(sorted_w_org):
                    if filled_counter >= w_how_much_swap:
                        break
                    label = w_targets[idx].item()
                    if label in self.swap_class_dist:
                        if self.swap_class_dist[label] + 1 <= self.swap_thr:
                            self.swap_class_dist[label] += 1
                            filled_counter +=1
                            selected.append(i)
                        else:
                            continue
                    else:
                        self.swap_class_dist[label] = 1
                        filled_counter +=1
                        selected.append(i)
                    
                    #print("LABEL : ", label)
                    #print("class dist : ", self.swap_class_dist)
                    #print("filled_count, how_much_swap : ", filled_counter, w_how_much_swap)

                sorted_w = sorted_w_org[selected][:w_how_much_swap]
                    
                #print("sorted_w : ", sorted_w)

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                    
                #print("selected_w_idxs : ", selected_w_idxs)
                #print("selected_w_targets : ", selected_w_targets)

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        if len(selected_idxs) != how_much_swap:
            print("ADDITIONAL SWAP CAND SELECTION!!!")
            for unselected in sorted_r_org:
                
                if len(selected_idxs) == how_much_swap:
                    break

                if unselected not in sorted_r:
                    # print(selected_idxs)
                    # print(r_idxs[unselected])
                    selected_idxs = torch.cat((selected_idxs,r_idxs[unselected].reshape(1)),dim=-1)
                    selected_targets = torch.cat((selected_targets,r_targets[unselected].reshape(1)),dim=-1)
                    if data_ids is not None:
                        selected_data_ids = torch.cat((selected_data_ids,r_data_ids[unselected].reshape(1)),dim=-1)
                    self.swap_class_dist[r_targets[unselected].item()] += 1
            
            for unselected in sorted_w_org:
                if len(selected_idxs) == how_much_swap:
                    break

                if unselected not in sorted_w:
                    # print(selected_idxs)
                    # print(w_idxs[unselected])
                    selected_idxs = torch.cat((selected_idxs,w_idxs[unselected].reshape(1)),dim=-1)
                    selected_targets = torch.cat((selected_targets,w_targets[unselected].reshape(1)),dim=-1)
                    if data_ids is not None:
                        selected_data_ids = torch.cat((selected_data_ids,w_data_ids[unselected].reshape(1)),dim=-1)
                    self.swap_class_dist[w_targets[unselected].item()] += 1


        assert len(selected_idxs) == how_much_swap

        #print("selected_targets : ", selected_targets)
        #print("\n")

        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets

    
    #@profile
    def all(self, idxs, outputs=None, targets=None, data_ids=None):
    
        ######### changed for time measurement
        if targets is not None and data_ids is not None:   
            return self.get_replay_index(idxs, targets, data_ids)
        else:
            return self.get_replay_index(idxs, targets)
            

    def hybrid_random(self, idxs, outputs, targets, data_ids=None):
        
        swap_ratio = self.threshold
        #print("SWAP RATIO : ", swap_ratio)

        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        else:
            replay_index_of_idxs = torch.arange(0, len(idxs), dtype=torch.long)

        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))
        #print("HOW MUCH SWAP ? ", how_much_swap)

        replay_output = outputs[replay_index_of_idxs].clone().detach()
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if data_ids is not None:
            replay_data_ids = data_ids[replay_index_of_idxs].clone().detach()

        soft_output = self.softmax(replay_output)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        #print("replay_index_of_idxs : ", replay_index_of_idxs)
        #print("replay idxs : ", replay_idxs)
        #print("replay_targets : ", replay_targets)
        #print("entropy : ", entropy)
        
        #print("entropy size : ", entropy.shape)
        
        if replay_output.nelement() != 0:
            predicts = torch.max(replay_output, dim=1)[1]
            r_predicted = (predicts.cpu() == replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
            #print("r_pred(idx) : " , r_predicted)
            #print("r_pred size : ", r_predicted.shape)

            r_idxs = replay_idxs[r_predicted]
            r_entropy = entropy[r_predicted]
            r_targets = replay_targets[r_predicted]
            if data_ids is not None:
                r_data_ids = replay_data_ids[r_predicted]

            #print("r_idxs : ", r_idxs)
            #print("r_targets : ", r_targets)
            #print("r_entropy : ", r_entropy)
            
            sorted_r = torch.argsort(r_entropy)[:how_much_swap]

            #print("sorted_r : ", sorted_r)

            selected_r_idxs = r_idxs[sorted_r]
            selected_r_targets = r_targets[sorted_r]
            if data_ids is not None:
                selected_r_data_ids = r_data_ids[sorted_r]

            #print("selected_r_idxs : ", selected_r_idxs)
            #print("selected_r_targets : ", selected_r_targets)

            if len(sorted_r) < how_much_swap:

                #print("== WE NEED MORE SAMPLE TO SWAP EVEN IF ITS WRONG PRED!!")
                w_predicted = (predicts.cpu() != replay_targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
                w_idxs = replay_idxs[w_predicted]
                w_entropy = entropy[w_predicted]
                w_targets = replay_targets[w_predicted]
                    
                if data_ids is not None:
                    w_data_ids = replay_data_ids[w_predicted]

                #print("w_pred(idx) : " , w_predicted)
                #print("w_idxs : ", w_idxs)
                #print("w_targets : ", w_targets)
                #print("w_entropy : ", w_entropy)
                
                w_how_much_swap = how_much_swap-len(sorted_r)

                idx_to_pick = np.linspace(0, len(w_idxs), num=w_how_much_swap, dtype = int, endpoint=False)
                sorted_w = torch.argsort(w_entropy, descending=True)[idx_to_pick]

                #print("sorted_w : ", sorted_w)

                selected_w_idxs = w_idxs[sorted_w]
                selected_w_targets = w_targets[sorted_w]
                    
                if data_ids is not None:
                    selected_w_data_ids = w_data_ids[sorted_w]

                    
                #print("selected_w_idxs : ", selected_w_idxs)
                #print("selected_w_targets : ", selected_w_targets)

                selected_idxs = torch.cat((selected_r_idxs,selected_w_idxs),dim=-1)
                selected_targets = torch.cat((selected_r_targets,selected_w_targets),dim=-1)
                if data_ids is not None:
                    selected_data_ids = torch.cat((selected_r_data_ids,selected_w_data_ids),dim=-1)

            else:
                selected_idxs = selected_r_idxs
                selected_targets = selected_r_targets
                if data_ids is not None:
                    selected_data_ids = selected_r_data_ids
        
        else:
            if data_ids is not None:
                return torch.empty(0), torch.empty(0), torch.empty(0)
            else:
                return torch.empty(0), torch.empty(0)

        assert len(selected_idxs) == how_much_swap

        #print("selected_targets : ", selected_targets)
        #print("\n")

        if data_ids is not None:
            return selected_idxs, selected_targets, selected_data_ids
        else:
            return selected_idxs, selected_targets

class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            if self.dataset_name == "Caltech101Swap" or self.dataset_name == "Dvs128Swap":
                image = self.trsf(self.images[idx])
            else:
                image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label

class SwapDataset(Dataset):
    def __init__(self, images, labels, trsf, num_per_class = False, train_path = None, class_to_label = None, swap_ratio = 0.0, noise_prec = 0.0):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.num_per_class = num_per_class
        self.rp_len = len(images[0])
        self.train_path = train_path
        self.class_to_label = class_to_label
        self.swap_ratio = swap_ratio
        self.noise_prec = noise_prec
        if 'cifar100' in self.train_path:
            self.suffix = ".png"
            self.resize = None
        elif 'imagenet100' in self.train_path:
            self.resize = transforms.Compose([
                    transforms.RandomResizedCrop(64),
                    transforms.RandomHorizontalFlip(),
                ])
            self.suffix = ".JPEG"
        elif 'Caltech101' in self.train_path or 'DVS128Gesture' in self.train_path:
            self.resize = transforms.Resize(size=(64, 64), interpolation = transforms.InterpolationMode.NEAREST)
            self.suffix = ".npz"
        elif 'UrbanSound8K' in self.train_path or 'dsads' in self.train_path:
            self.suffix = ".npy"
    def __len__(self):
        if len(self.images) == 1:
            return len(self.images[0])
        else:
            return len(self.images[0]) + len(self.images[1])

    def __getitem__(self, idx):
        if idx < self.rp_len:
            id1 = 0
            id2 = idx
        else:
            id1 = 1
            id2 = idx - self.rp_len
        # if self.use_path:
        #     image = self.trsf(pil_loader(self.images[idx]))
        # else:
        if 'Caltech101' in self.train_path or 'DVS128Gesture' in self.train_path or 'UrbanSound8K' in self.train_path or 'dsads' in self.train_path:
            image = self.trsf(self.images[id1][id2])
        else:
            image = self.trsf(Image.fromarray(self.images[id1][id2]))
        label = self.labels[id1][id2]

        return idx, image, label

    def random(self, idxs, targets):
        swap_ratio = self.swap_ratio


        # idxs and targets are lists now 
        idxs, targets = torch.tensor(idxs),torch.tensor(targets)
        if self.rp_len is not None:
            replay_index_of_idxs = (idxs < self.rp_len).squeeze().nonzero(as_tuple=True)[0]
        replay_idxs = idxs[replay_index_of_idxs].clone().detach()
        replay_targets = targets[replay_index_of_idxs].clone().detach()
        if (swap_ratio==1): return replay_idxs, replay_targets
        how_much_swap = math.ceil(swap_ratio * len(replay_index_of_idxs))
        selected_index = np.random.choice(len(replay_idxs), how_much_swap, replace=False)

        assert len(selected_index) == how_much_swap

        return replay_idxs[selected_index], replay_targets[selected_index] #, outputs[selected_index]

    async def _swap_main(self, label, swap_idx):
        # try:
        prefix = f'{self.class_to_label[label.item()]}_'
        num_file = self.num_per_class[label]
        replace_file = self.train_path + '/' + prefix + str(random.randint(0,num_file - 1)) + self.suffix
        return await self._get_data(swap_idx, replace_file)

    async def _swap(self, what_to_swap, labels):
        # fin_idx, fin_target = self.random(what_to_swap, labels)
        cos = [ self._swap_main(label, idx ) for label, idx in zip(labels, what_to_swap) ]
        res = await asyncio.gather(*cos)
        return res
    async def _get_data(self, idx, filename):
        if 'png' in filename or 'JPEG' in filename:
            vec = await self._get_img(filename)
            if self.resize is not None:
                vec = self.resize(vec)
        elif 'npy' in filename:
            vec = torch.from_numpy(np.load(filename)).float().unsqueeze(0)
        else:
            vec = self.resize(torch.from_numpy(np.load(filename)['frames']).float())
        if self.noise_prec > 0 and self.noise_prec > random.random():
            vec = np.array(vec)
            noise = np.random.randint(0, 256, vec.shape)
            # print("change data noise is", noise)
            # print("change data vec is", vec)
            vec = vec + noise
        self.images[0][idx] = vec
        return True
    async def _get_img(self, filename):
        f = os.open( filename, os.O_RDONLY | os.O_DIRECT)

        os.lseek(f,0,0)
        actual_size = os.path.getsize(filename)
        block_size = 512 * math.ceil(actual_size / 512)
        fr = directio.read(f, block_size)
        os.close(f)
        
        data = io.BytesIO(fr[:actual_size])
        

        img = Image.open(data)
        img = img.convert('RGB')
        
        return img
    
def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "cifarswap":
        return iCIFARSWAP()
    elif name == "imagenetswap":
        return iImageNetSwap()
    elif name == "ncaltech101swap":
        return NCaltech101Swap()
    elif name == "urbansoundswap":
        return UrbanSound8KSwap()
    elif name == "dsadsswap":
        return DSADSSwap()
    elif name == "dvs128swap":
        return DVS128Swap()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
