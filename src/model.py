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
            nn.ReLU(),
            nn.Linear(config["hidden_1"], config["hidden_2"]),
            nn.ReLU(),
            nn.Linear(config["hidden_2"], config["hidden_3"]),
            nn.ReLU(),
            nn.Linear(config["hidden_3"], 784), #The output of 784 is because the MNIST dataset has 784 features.
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        # Using Maxout, you can specify the number of pieces you want in each layer
        self.model = nn.Sequential(
            Maxout(config["latent_size"], config["hidden_1"], pieces=config["maxout_1_pieces"]),
            nn.Dropout(config["dropout_1"]),
            Maxout(config["hidden_1"], config["hidden_2"], pieces=config["maxout_2_pieces"]),
            nn.Dropout(config["dropout_2"]),
            nn.Linear(config["hidden_2"], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class GAN:
    def __init__(self, config):
        self.config = config
        self.generator = Generator(config["generator"])
        self.discriminator = Discriminator(config["discriminator"])
        self.real_label = 0.9
        self.fake_label = 0.1
