import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def one_hot_encode(labels, num_classes):
    return torch.nn.functional.one_hot(labels, num_classes).float()

def color_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def gray_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


class CustomDatasetWithLabel(Dataset):
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]  # Get the transformed image and original label
        label = one_hot_encode(torch.tensor(label), self.num_classes)  # Transform the label
        return img, label  # Return the transformed image and label


class DataLoaderClass:
    def __init__(self, config):
        assert config["data_config"]["dataset"] in ["MNIST", "CIFAR10"], "Dataset not supported"
        self.config = config
        self.num_classes = None
        self.base_transform = None

        self._dataset_settings()
        
    def _dataset_settings(self):
        if self.config["data_config"]["dataset"] == "MNIST":
            self.base_transform = gray_transform()
            self.num_classes = 10
        
        elif self.config["data_config"]["dataset"] == "CIFAR10":
            self.base_transform = color_transform()
            self.num_classes = 10

    def load_data(self, config):
        assert config["data_config"]["dataset"] in ["MNIST", "CIFAR10"], "Dataset not supported"
        
        if config["data_config"]["dataset"] == "MNIST":
            dataset = torchvision.datasets.MNIST(root=config["data_config"]["data_dir"], train=True, download=True, 
                                                 transform=self.base_transform)
        
        elif config["data_config"]["dataset"] == "CIFAR10":
            dataset = torchvision.datasets.CIFAR10(root=config["data_config"]["data_dir"], train=True, download=True, 
                                                   transform=self.base_transform)
        
        return CustomDatasetWithLabel(dataset, self.num_classes)

    def get_data_loader(self, config, fid_images=False):
        dataset = self.load_data(config)
        
        if fid_images:
            return DataLoader(dataset, batch_size=config["eval_config"]["num_fid_images"], shuffle=False, num_workers=config["data_config"]["num_workers"])
        
        else:
            return DataLoader(dataset, batch_size=config["data_config"]["batch_size"], shuffle=config["data_config"]["shuffle"], num_workers=config["data_config"]["num_workers"])
