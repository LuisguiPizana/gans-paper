import torch
import torch.nn as nn

class Maxout(nn.Module):
    def __init__(self, in_features, out_features, pieces):
        super(Maxout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pieces = pieces
        self.linear = nn.Linear(in_features, out_features * pieces)

    def forward(self, x):
        x = self.linear(x)
        # Reshape x so that the pieces dimension is second, then take max over dim=1
        x = x.view(-1, self.pieces, self.out_features)
        x, _ = x.max(dim=1)
        return x

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Linear(config["latent_size"], config["hidden_1"]),
            nn.BatchNorm1d(config["hidden_1"]), 
            nn.LeakyReLU(negative_slope=config["leaky_relu_1"]),
            nn.Linear(config["hidden_1"], config["hidden_2"]),
            nn.BatchNorm1d(config["hidden_2"]),
            nn.LeakyReLU(negative_slope=config["leaky_relu_2"]),
            nn.Linear(config["hidden_2"], config["hidden_3"]),
            nn.BatchNorm1d(config["hidden_3"]),
            nn.LeakyReLU(negative_slope=config["leaky_relu_3"]),
            nn.Linear(config["hidden_3"], 784), #The output of 784 is because the MNIST dataset has 784 features.
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, config, img_shape):
        super(Discriminator, self).__init__()
        self.config = config
        self.img_channels = img_shape[0]
        self.img_height = img_shape[1]
        self.img_width = img_shape[2]
        # Using Maxout, you can specify the number of pieces you want in each layer
        self.model = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(self.img_channels, self.config["conv_1_filters"], self.config["conv_1_kernel"])
            nn.LeakyReLU(negative_slope=config["leaky_relu_slope"])

            nn.Conv2d(self.config["conv_1_filters"], self.config["conv_2_filters"], self.config["conv_2_kernel"])
            nn.LeakyReLU(negative_slope=config["leaky_relu_slope"])

            nn.Conv2d(self.config["conv_2_filters"], self.config["conv_3_filters"], self.config["conv_3_kernel"])
            nn.LeakyReLU(negative_slope=config["leaky_relu_slope"])
            
            nn.Flatten()
            nn.Linear(config["hidden_2"] * self.img_with * self.img_height, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class GAN:
    def __init__(self, config):
        self.config = config
        if config["data_config"]["dataset"] == "MNIST":
            self.image_shape = (1, 28, 28)
        elif config["data_config"]["dataset"] == "CIFAR10":
            self.image_shape = (3, 32, 32)
        self.generator = Generator(config["generator"], self.image_shape)
        self.discriminator = Discriminator(config["discriminator"], self.image_shape)
        self.real_label = 0.9
        self.fake_label = 0.0
