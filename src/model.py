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
    def __init__(self):
        super(Generator, self).__init__()
        # Start from a latent vector of a certain size, e.g., 100
        self.latent_size = 100
        
        # First fully connected layer
        self.fc = nn.Linear(self.latent_size, 7*7*128)  # Upscale to a 7x7 image with 128 channels
        
        # Transpose convolutions
        self.conv_trans1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Output size: 14x14
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.conv_trans2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Output size: 28x28
        self.batch_norm2 = nn.BatchNorm2d(32)
        
        # Final layer to produce output image
        self.conv_final = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)  # Output channel: 1 for grayscale image
        
        # Activation functions
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()  # Tanh to output values between -1 and 1

    def forward(self, x):
        # Input x is the latent vector
        x = self.fc(x)
        x = x.view(-1, 128, 7, 7)  # Reshape to a 7x7 image with 128 channels
        x = self.relu(self.batch_norm1(self.conv_trans1(x)))
        x = self.relu(self.batch_norm2(self.conv_trans2(x)))
        x = self.tanh(self.conv_final(x))
        return x

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
