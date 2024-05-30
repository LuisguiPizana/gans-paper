import torch
import torchvision
import torchvision.transforms as transforms

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

def load_data(config):
    assert config["data_config"]["dataset"] in ["MNIST", "CIFAR10"], "Dataset not supported"
    if config["data_config"]["dataset"] == "MNIST":
        return torchvision.datasets.MNIST(root = config["data_config"]["data_dir"], train = True, transform = gray_transform(), download = True)
    elif config["data_config"]["dataset"] == "CIFAR10":
        return torchvision.datasets.CIFAR10(root = config["data_config"]["data_dir"], train = True, transform = color_transform(), download = True)

def get_data_loader(config, fid_images = False):
    dataset = load_data(config)
    if fid_images:
        return torch.utils.data.DataLoader(dataset, batch_size = config["eval_config"]["num_fid_images"], shuffle = False, num_workers = config["data_config"]["num_workers"])
    else:
        return torch.utils.data.DataLoader(dataset, batch_size = config["data_config"]["batch_size"], shuffle = config["data_config"]["shuffle"], num_workers = config["data_config"]["num_workers"])