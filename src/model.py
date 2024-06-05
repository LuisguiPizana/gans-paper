import torch.nn as nn

def compute_output_size(input_size, conv_layers):
    # Input_size: [batch_size]
    height, width = input_size
    for layer in conv_layers:
        if isinstance(layer, nn.Conv2d):
            stride = layer.stride
            padding = layer.padding
            dilation = layer.dilation
            kernel_size = layer.kernel_size
            height = (height + 2*padding - dilation * (kernel_size -1) - 1) // stride + 1
            width = (width + 2*padding - dilation * (kernel_size -1) - 1) // stride + 1

    return height, width


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
    def __init__(self, config, img_shape):
        super(Generator, self).__init__()
        self.config = config
        self.img_channels = img_shape[0]
        self.img_height = img_shape[1]
        self.img_width = img_shape[2]

        self.model = nn.Sequential(
            # Linear layer and unflattening.
            nn.Linear(config["latent_size"], config["linear_layer"] * config["img_height_1"] ** 2),
            nn.BatchNorm1d(config["linear_layer"] * config["img_height_1"] ** 2),
            nn.Unflatten(1 , (config["linear_layer"],  config["img_height_1"],  config["img_height_1"])),
            # First convolutional layer.
            nn.ConvTranspose2d(config["linear_layer"],
                               config["conv_1_filters"],
                               kernel_size = config["kernel_size_1"],
                               stride = config["stride_1"],
                               padding=config["padding_1"] ),
            nn.ReLu(True),
            nn.BatchNorm2d(config["conv_1_filters"]),
            # Second convolutional layer.
            nn.ConvTranspose2d(config["conv_1_filters"],
                    config["conv_2_filters"],
                    kernel_size = config["kernel_size_2"],
                    stride = config["stride_2"],
                    padding=config["padding_2"] ),
            nn.ReLu(True),
            # Output TConv layer.
            nn.ConvTranspose2d(config["conv_1_filters"],
                    config["conv_2_filters"],
                    kernel_size = config["kernel_size_3"],
                    stride = config["stride_3"],
                    padding=config["padding_3"] ),
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

        conv2d_1 = nn.Conv2d(self.img_channels, self.config["conv_1_filters"], self.config["conv_1_kernel"])
        conv2d_2 = nn.Conv2d(self.config["conv_1_filters"], self.config["conv_2_filters"], self.config["conv_2_kernel"])
        conv2d_3 = nn.Conv2d(self.config["conv_2_filters"], self.config["conv_3_filters"], self.config["conv_3_kernel"])

        self.out_height, self.out_width =  compute_output_size([self.img_height, self.img_width], [conv2d_1, conv2d_2, conv2d_3])
        
        self.model = nn.Sequential(
            # First convolutional layer
            conv2d_1,
            nn.LeakyReLU(negative_slope=config["leaky_relu_slope"]),
            nn.BatchNorm2d(self.config["conv_1_filters"]),
            # Second convolutional layer
            conv2d_2,
            nn.LeakyReLU(negative_slope=config["leaky_relu_slope"]),
            nn.BatchNorm2d(self.config["conv_2_filters"]),
            # Third convolutional layer
            conv2d_3,
            nn.LeakyReLU(negative_slope=config["leaky_relu_slope"]),
            nn.BatchNorm2d(self.config["conv_3_filters"]),
            # Linear output layer
            nn.Flatten(),
            nn.Linear(config["hidden_2"] * self.out_height * self.out_width, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        reshaped_x = x.view(-1, self.img_channels, self.img_height, self.img_width)
        return self.model(reshaped_x)

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
