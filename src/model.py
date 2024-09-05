import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config

        self.joint_layers = nn.Sequential(
            nn.Linear(config["num_classes"] + config["latent_size"], config["joint_hidden_1"]),
            nn.BatchNorm1d(config["joint_hidden_1"]),
            nn.LeakyReLU(negative_slope=config["leaky_relu"]),
            nn.Linear(config["joint_hidden_1"], config["joint_hidden_2"]),
            nn.BatchNorm1d(config["joint_hidden_2"]), 
            nn.LeakyReLU(negative_slope=config["leaky_relu"]),
            nn.Linear(config["joint_hidden_2"], config["joint_hidden_3"]),
            nn.BatchNorm1d(config["joint_hidden_3"]),
            nn.Linear(config["joint_hidden_3"], 784),
            nn.Tanh() #They use a sigmoid layer, this means the dataprocessing is from 0 to 1.
        )
 
    def forward(self, x):
        img, label = x
        input = torch.cat((img, label), axis = 1)
        return self.joint_layers(input).view(-1, 1, 28, 28) #Avoid hardcoding. To do. Unflattening.

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config

        self.joint_layers = nn.Sequential(
            nn.Linear(config["num_classes"] + config["latent_size"], config["joint_hidden_1"]),
            nn.LeakyReLU(negative_slope=config["leaky_relu"]),
            nn.Linear(config["joint_hidden_1"], config["joint_hidden_2"]),
            nn.LeakyReLU(negative_slope=config["leaky_relu"]),
            nn.Linear(config["joint_hidden_2"], 1),
            nn.Sigmoid() #They use a sigmoid layer, this means the dataprocessing is from 0 to 1.
        )


    def forward(self, x):
        img, label = x
        flat_img = img.view(-1, 784) #We could specify the batch size and leave -1 for the rest of the dimensions.
        input = torch.cat((flat_img, label), axis = 1)
        return self.joint_layers(input)

class GAN:
    def __init__(self, config):
        self.config = config
        self.generator = Generator(config["generator"])
        self.discriminator = Discriminator(config["discriminator"])
        self.real_label = 1.0
        self.fake_label = 0.0
