import torch

class Generator(torch.nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.model = torch.nn.Sequential(
            torch.nn.Linear(config["latent_size"], config["hidden_1"]), 
            torch.nn.ReLU(),
            torch.nn.Linear(config["hidden_1"], config["hidden_2"]),
            torch.nn.ReLU(),
            torch.nn.Linear(config["hidden_2"], 784), #The output of 784 is because the MNIST dataset has 784 features.
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(torch.nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.model = torch.nn.Sequential(
            torch.nn.Linear(config["latent_size"], config["hidden_1"]),
            torch.nn.ReLU(),
            torch.nn.Dropout(config["dropout_1"]),
            torch.nn.Linear(config["hidden_1"], config["hidden_2"]),
            torch.nn.ReLU(),
            torch.nn.Dropout(config["dropout_2"]),
            torch.nn.Linear(config["hidden_2"], 1), #Binnary classification
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class GAN:
    def __init__(self, config):
        self.config = config
        self.generator = Generator(config["generator"])
        self.discriminator = Discriminator(config["discriminator"])
        self.real_label = 1
        self.fake_label = 0
