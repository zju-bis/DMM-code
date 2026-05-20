import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import os
import json
import torch
from PIL import Image

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFARSWAP(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]
    label_to_class = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge',
                      'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
                      'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                      'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
                      'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
                      'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
                      'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
                      'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    class_order = np.arange(100).tolist()
    train_path = './data/cifar100/train'
    def download_data(self):
        self.train_data = None
        self.train_targets = None
        self.test_targets = None
        self.test_data = None
    def load_data(self, labels, spaces, num_per_class, policy = 'back'):
        train_data = []
        train_targets = []
        for label in labels:
            if policy == 'random':
                space = spaces[label]
                idxs = []
                while space > num_per_class[label]:
                    idxs.append(np.random.choice(num_per_class[label], num_per_class[label], replace=False))
                    space -= num_per_class[label]
                idxs.append(np.random.choice(num_per_class[label], space, replace=False))
                idx = np.concatenate(idxs)
            else:
                num = min(spaces, num_per_class[label])
                idx = np.arange(num_per_class[label] - num, num_per_class[label])
            for _, i in enumerate(idx):
                imgname = f'{self.label_to_class[label]}_{i}.png'
                # print(imgname)
                img = Image.open(os.path.join(self.train_path, imgname))
                # img = np.array(img).transpose(2, 0, 1)
                img = np.array(img)
                train_data.append(img)
                train_targets.append(label)
        self.train_data = np.array(train_data)
        self.train_targets = np.array(train_targets)

    def load_test_data(self, labels):
        path = './data/cifar100/test'
        test_data = []
        test_targets = []
        for label in labels:
            for i in range(100):
                imgname = f'{self.label_to_class[label]}_{i}.png'
                img = Image.open(os.path.join(path, imgname))
                # img = np.array(img).transpose(2, 0, 1)
                img = np.array(img)
                test_data.append(img)
                test_targets.append(label)
        self.test_data = np.array(test_data)
        self.test_targets = np.array(test_targets)

class NCaltech101Swap(iData):
    def __init__(self):
        self.use_path = False
        self.train_trsf = [
            # transforms.ToTensor()
        ]
        self.resize = transforms.Resize(size=(64, 64), interpolation = transforms.InterpolationMode.NEAREST)
        self.test_trsf = []
        self.common_trsf = []
        self.train_path = './data/Caltech101/frames_number_8_split_by_number'
        with open(os.path.join(self.train_path, "index_to_label.json"), "r") as f:
            self.label_to_class = json.load(f)
        with open(os.path.join(self.train_path, "len_per_class.json"), "r") as f:
            self.len_per_class = json.load(f)
        self.class_order = np.arange(101).tolist()
    def download_data(self):
        self.train_data = None
        self.train_targets = None
        self.test_targets = None
        self.test_data = None
    def load_data(self, labels, spaces, num_per_class, policy = 'back'):
        train_data = []
        train_targets = []
        for label in labels:
            len = int(self.len_per_class[label] * 0.9)
            if policy == 'random':
                space = spaces[label]
                idxs = []
                while space > len:
                    idxs.append(np.random.choice(len, len, replace=False))
                    space -= len
                idxs.append(np.random.choice(len, space, replace=False))
                idx = np.concatenate(idxs)
            else:
                num = min(spaces, len)
                idx = np.arange(self.len_per_class[label] - num, self.len_per_class[label])
            for _, i in enumerate(idx):
                imgname = f'{self.label_to_class[label]}_{i}.npz'
                # print(imgname)
                img = self.resize(torch.from_numpy(np.load(os.path.join(self.train_path, imgname))['frames']).float())
                # img = np.array(img).transpose(2, 0, 1)
                train_data.append(img)
                train_targets.append(label)
        self.train_data = np.array(train_data)
        self.train_targets = np.array(train_targets)

    def load_test_data(self, labels):
        path = './data/Caltech101/frames_number_8_split_by_number'
        test_data = []
        test_targets = []
        for label in labels:
            for i in range(int(self.len_per_class[label] * 0.9), self.len_per_class[label]):
                imgname = f'{self.label_to_class[label]}_{i}.npz'
                img = self.resize(torch.from_numpy(np.load(os.path.join(self.train_path, imgname))['frames']).float())
                # img = np.array(img).transpose(2, 0, 1)
                test_data.append(img)
                test_targets.append(label)
        self.test_data = np.array(test_data)
        self.test_targets = np.array(test_targets)

class DSADSSwap(iData):
    def __init__(self):
        self.use_path = False
        self.train_trsf = [
            # transforms.ToTensor()
        ]
        self.test_trsf = []
        self.common_trsf = []
        self.train_path = './data/dsads/train'
        self.test_path = './data/dsads/test'
        self.label_to_class = {0:0, 1:1, 2: 2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18}
        self.len_per_class = [384] * 19
        self.class_order = np.arange(19).tolist()
    def download_data(self):
        self.train_data = None
        self.train_targets = None
        self.test_targets = None
        self.test_data = None
    def load_data(self, labels, spaces, num_per_class, policy = 'back'):
        train_data = []
        train_targets = []
        for label in labels:
            len = 384
            if policy == 'random':
                space = spaces[label]
                idxs = []
                while space > len:
                    idxs.append(np.random.choice(len, len, replace=False))
                    space -= len
                idxs.append(np.random.choice(len, space, replace=False))
                idx = np.concatenate(idxs)
            else:
                num = min(spaces, len)
                idx = np.arange(self.len_per_class[label] - num, self.len_per_class[label])
            for _, i in enumerate(idx):
                imgname = f'{self.label_to_class[label]}_{i}.npy'
                # print(imgname)
                img = torch.from_numpy(np.load(os.path.join(self.train_path, imgname),allow_pickle= True)).unsqueeze(0)
                # print(img.shape)
                # img = np.array(img).transpose(2, 0, 1)
                train_data.append(img)
                train_targets.append(label)
        self.train_data = np.array(train_data)
        self.train_targets = np.array(train_targets)

    def load_test_data(self, labels):
        test_data = []
        test_targets = []
        for label in labels:
            for i in range(0, 96):
                imgname = f'{self.label_to_class[label]}_{i}.npy'
                img = torch.from_numpy(np.load(os.path.join(self.test_path, imgname),allow_pickle= True)).unsqueeze(0)
                test_data.append(img)
                test_targets.append(label)
        print(len(test_data))
        self.test_data = np.array(test_data)
        self.test_targets = np.array(test_targets)

class UrbanSound8KSwap(iData):
    def __init__(self):
        self.use_path = False
        self.train_trsf = [
            # transforms.ToTensor()
        ]
        self.test_trsf = []
        self.common_trsf = []
        self.train_path = './data/UrbanSound8K/npy'
        self.label_to_class = {0:0, 1:1, 2: 2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}
        with open(os.path.join(self.train_path, "len_per_class.json"), "r") as f:
            self.len_per_class = json.load(f)
        self.class_order = np.arange(10).tolist()
    def download_data(self):
        self.train_data = None
        self.train_targets = None
        self.test_targets = None
        self.test_data = None
    def load_data(self, labels, spaces, num_per_class, policy = 'back'):
        train_data = []
        train_targets = []
        for label in labels:
            len = int(self.len_per_class[label] * 0.9)
            if policy == 'random':
                space = spaces[label]
                idxs = []
                while space > len:
                    idxs.append(np.random.choice(len, len, replace=False))
                    space -= len
                idxs.append(np.random.choice(len, space, replace=False))
                idx = np.concatenate(idxs)
            else:
                num = min(spaces, len)
                idx = np.arange(self.len_per_class[label] - num, self.len_per_class[label])
            for _, i in enumerate(idx):
                imgname = f'{self.label_to_class[label]}_{i}.npy'
                # print(imgname)
                img = torch.from_numpy(np.load(os.path.join(self.train_path, imgname),allow_pickle= True)).unsqueeze(0)
                # print(img.shape)
                # img = np.array(img).transpose(2, 0, 1)
                train_data.append(img)
                train_targets.append(label)
        self.train_data = np.array(train_data)
        self.train_targets = np.array(train_targets)

    def load_test_data(self, labels):
        test_data = []
        test_targets = []
        for label in labels:
            for i in range(int(self.len_per_class[label] * 0.9), self.len_per_class[label]):
                imgname = f'{self.label_to_class[label]}_{i}.npy'
                img = torch.from_numpy(np.load(os.path.join(self.train_path, imgname),allow_pickle= True)).unsqueeze(0)
                test_data.append(img)
                test_targets.append(label)
        self.test_data = np.array(test_data)
        self.test_targets = np.array(test_targets)

class DVS128Swap(iData):
    def __init__(self):
        self.use_path = False
        self.train_trsf = [
            # transforms.ToTensor()
        ]
        self.resize = transforms.Resize(size=(64, 64), interpolation = transforms.InterpolationMode.NEAREST)
        self.test_trsf = []
        self.common_trsf = []
        self.train_path = './data/DVS128Gesture/frames_number_8_split_by_number/train'
        self.test_path = './data/DVS128Gesture/frames_number_8_split_by_number/test'
        self.label_to_class = {0:0, 1:1, 2: 2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10}
        with open(os.path.join(self.train_path, "len_per_class.json"), "r") as f:
            self.len_per_class = json.load(f)
        with open(os.path.join(self.test_path, "len_per_class.json"), "r") as f:
            self.test_len_per_class = json.load(f)
        self.class_order = np.arange(101).tolist()
    def download_data(self):
        self.train_data = None
        self.train_targets = None
        self.test_targets = None
        self.test_data = None
    def load_data(self, labels, spaces, num_per_class, policy = 'back'):
        train_data = []
        train_targets = []
        for label in labels:
            len = self.len_per_class[label]
            if policy == 'random':
                space = spaces[label]
                idxs = []
                while space > len:
                    idxs.append(np.random.choice(len, len, replace=False))
                    space -= len
                idxs.append(np.random.choice(len, space, replace=False))
                idx = np.concatenate(idxs)
            else:
                num = min(spaces, len)
                idx = np.arange(self.len_per_class[label] - num, self.len_per_class[label])
            for _, i in enumerate(idx):
                imgname = f'{str(label)}_{i}.npz'
                img = self.resize(torch.from_numpy(np.load(os.path.join(self.train_path, imgname))['frames']).float())
                train_data.append(img)
                train_targets.append(label)
        self.train_data = np.array(train_data)
        self.train_targets = np.array(train_targets)

    def load_test_data(self, labels):
        test_data = []
        test_targets = []
        for label in labels:
            for i in range(0, self.test_len_per_class[label]):
                imgname = f'{str(label)}_{i}.npz'
                img = self.resize(torch.from_numpy(np.load(os.path.join(self.test_path, imgname))['frames']).float())
                # img = np.array(img).transpose(2, 0, 1)
                test_data.append(img)
                test_targets.append(label)
        self.test_data = np.array(test_data)
        self.test_targets = np.array(test_targets)

class iImageNetSwap(iData):
    def __init__(self):
        self.use_path = False
        self.train_trsf = [
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.Resize(72),
            transforms.CenterCrop(64),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.final_train_trsf = transforms.Compose(self.train_trsf)
        self.final_test_trsf = transforms.Compose(self.test_trsf)
        with open("./data/imagenet100/Labels.json", 'r') as f:
            json_data = json.loads(f.read())
        self.label_to_class = list(json_data.keys())
        del json_data
        self.class_order = np.arange(100).tolist()
        self.train_path = './data/imagenet100/train'

    def download_data(self):
        self.train_data = None
        self.train_targets = None
        self.test_targets = None
        self.test_data = None
    def load_data(self, labels, spaces, num_per_class, policy = 'back'):
        path = './data/imagenet100/train'
        train_data = []
        train_targets = []
        for label in labels:
            if policy == 'random':
                space = spaces[label]
                idxs = []
                while space > num_per_class[label]:
                    idxs.append(np.random.choice(1300, num_per_class[label], replace=False))
                    space -= num_per_class[label]
                idxs.append(np.random.choice(1300, space, replace=False))
                idx = np.concatenate(idxs)
            else:
                num = min(spaces, num_per_class[label])
                idx = np.arange(1300 - num, 1300)
            for _, i in enumerate(idx):
                imgname = f'{self.label_to_class[label]}_{i}.JPEG'
                # print(imgname)
                img = Image.open(os.path.join(path, imgname)).convert('RGB')
                # img = np.array(img).transpose(2, 0, 1)
                img = np.array(self.final_train_trsf(img))
                train_data.append(img)
                train_targets.append(label)
        print(len(train_data))
        self.train_data = np.array(train_data)
        self.train_targets = np.array(train_targets)

    def load_test_data(self, labels):
        path = './data/imagenet100/val'
        test_data = []
        test_targets = []
        for label in labels:
            for i in range(50):
                imgname = f'{self.label_to_class[label]}_{i}.JPEG'
                img = Image.open(os.path.join(path, imgname)).convert('RGB')
                # img = np.array(img).transpose(2, 0, 1)
                img = np.array(self.final_test_trsf(img))
                test_data.append(img)
                test_targets.append(label)
        self.test_data = np.array(test_data)
        self.test_targets = np.array(test_targets)

class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "data/imagenet100/train/"
        test_dir = "data/imagenet100/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
