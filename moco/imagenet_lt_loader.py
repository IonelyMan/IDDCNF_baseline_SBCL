import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class ImageNetLT_moco(Dataset):
    """Training split in class directories, with the original MoCo sample format."""

    def __init__(self, root, transform=None, class_balance=False):
        folder = ImageFolder(root)
        if folder.classes != ['benign', 'mal']:
            raise ValueError("Expected exactly the class directories 'benign' and 'mal' in " + root)
        self.img_path = [path for path, _ in folder.samples]
        self.labels = folder.targets
        self.num_classes = len(folder.classes)
        self.new_labels = []
        self.transform = transform
        self.class_balance = class_balance
        self.class_data = [[] for _ in range(self.num_classes)]
        for index, label in enumerate(self.labels):
            self.class_data[label].append(index)
        self.cls_num_list = [len(indices) for indices in self.class_data]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.class_balance:
            label = random.randint(0, self.num_classes - 1)
            index = random.choice(self.class_data[label])
        else:
            label = self.labels[index]
        cluster_label = self.new_labels[index] if self.new_labels else -1
        with open(self.img_path[index], 'rb') as f:
            img = Image.open(f).convert('RGB')
        if len(self.transform) == 2:
            return [self.transform[0](img), self.transform[1](img)], label, cluster_label
        return self.transform[0](img), label, cluster_label


class ImageNetLT_val(Dataset):
    """Validation split using the same two class names and original transforms."""

    def __init__(self, root, transform=None, class_balance=False):
        folder = ImageFolder(root)
        if folder.classes != ['benign', 'mal']:
            raise ValueError("Expected exactly the class directories 'benign' and 'mal' in " + root)
        self.img_path = [path for path, _ in folder.samples]
        self.labels = folder.targets
        self.num_classes = len(folder.classes)
        self.transform = transform
        self.class_balance = class_balance
        self.class_data = [[] for _ in range(self.num_classes)]
        for index, label in enumerate(self.labels):
            self.class_data[label].append(index)
        self.cls_num_list = [len(indices) for indices in self.class_data]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.class_balance:
            label = random.randint(0, self.num_classes - 1)
            index = random.choice(self.class_data[label])
        else:
            label = self.labels[index]
        with open(self.img_path[index], 'rb') as f:
            img = Image.open(f).convert('RGB')
        if len(self.transform) == 2:
            return [self.transform[0](img), self.transform[1](img)], label
        return self.transform[0](img), label
