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

        self.label_layer = nn.Sequential(
            nn.Linear(config["num_classes"], config["label_hidden_units"]),
            nn.Dropout(config["label_dropout"]),
            nn.ReLU(True)
        )

        self.img_layer = nn.Sequential(
            nn.Linear(config["latent_size"], config["img_hidden_units"]),
            nn.Dropout(config["img_dropout"]),
            nn.ReLU(True)

        )

        self.joint_layers = nn.Sequential(
            nn.Linear(config["label_hidden_units"] + config["img_hidden_units"], config["joint_hidden_1"]),
            nn.Dropout(config["joint_dropout"]), 
            nn.LeakyReLU(negative_slope=config["leaky_relu"]),
            nn.Linear(config["joint_hidden_1"], 784),
            nn.Tanh() #They use a sigmoid layer, this means the dataprocessing is from 0 to 1.
        )
 
    def forward(self, x):
        img, label = x
        img_layer_output = self.img_layer(img)
        label_layer_output = self.label_layer(label)
        joint_representation = torch.cat((img_layer_output, label_layer_output), axis = 1)
        return self.joint_layers(joint_representation).view(-1, 1, 28, 28) #Avoid hardcoding. To do.

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config

        self.img_layer = nn.Sequential(
           Maxout(config["latent_size"], config["img_maxout_units"], config["img_maxout_pieces"]),
           nn.Dropout(config["img_dropout"])
        )

        self.label_layer = nn.Sequential(
            Maxout(config["num_classes"], config["label_maxout_units"], pieces=config["label_maxout_pieces"]),
            nn.Dropout(config["label_dropout"]),
        )

        self.joint_layers = nn.Sequential(
            Maxout(config["label_maxout_units"] + config["img_maxout_units"], config["joint_maxout_units"], pieces=config["joint_maxout_pieces"]),
            nn.Dropout(config["joint_dropout"]),
            nn.Linear(config["joint_maxout_units"], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        img, label = x
        img = img + torch.rand_like(img) * self.config["img_noise_rate"]
        flat_img = img.view(-1, 784)
        img_layer_output = self.img_layer(flat_img)
        label_layer_output = self.label_layer(label)
        label_layer_output = label_layer_output + torch.rand_like(label_layer_output) * self.config["embedding_noise_rate"]
        joint_representation = torch.cat((img_layer_output, label_layer_output), axis = 1)
        return self.joint_layers(joint_representation)

class GAN:
    def __init__(self, config):
        self.config = config
        self.generator = Generator(config["generator"])
        self.discriminator = Discriminator(config["discriminator"])
        self.real_label = 0.85
        self.fake_label = 0.0
