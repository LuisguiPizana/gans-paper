
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import json
import model
import logging
from tqdm import tqdm
from custom_experiment_tracking import CustomExperimentTracker

torch.autograd.set_detect_anomaly(True)

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
    if config["dataset"] == "MNIST":
        return torchvision.datasets.MNIST(root = config["data_dir"], train = True, transform = gray_transform(), download = True)
    else:
        return torchvision.datasets.CIFAR10(root = config["data_dir"], train = True, transform = color_transform(), download = True)

def get_data_loader(config):
    dataset = load_data(config)
    return torch.utils.data.DataLoader(dataset, batch_size = config["batch_size"], shuffle = config["shuffle"], num_workers = config["num_workers"])


def instantiate_optimizer(model, config, optimizer_type):
    """
    model: This is the PyTorch model (or a specific part of it, like the generator or discriminator in a GAN) for which the optimizer is being created. The function expects that this model will have trainable parameters.
    config: A dictionary that contains all the configuration settings, including those specifically for the optimizer.
    optimizer_type: A string key (e.g., "generator" or "discriminator") that specifies which optimizer configuration to use from the config dictionary.
    """
    optimizer_config = config["optimizer"][optimizer_type]
    optimizer_class = getattr(torch.optim, optimizer_config["type"])
    return optimizer_class(model.parameters(), **optimizer_config["params"])


def instantiate_scheduler(optimizer, config, scheduler_type):
    """
    optimizer: This is the PyTorch optimizer for which the scheduler is being created.
    config: A dictionary that contains all the configuration settings, including those specifically for the scheduler.
    scheduler_type: A string key (e.g., "generator" or "discriminator") that specifies which scheduler configuration to use from the config dictionary.
    """
    scheduler_config = config["scheduler"][scheduler_type]
    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_config["type"])
    return scheduler_class(optimizer, **scheduler_config["params"])


class GanTrainer:
    def __init__(self, config):
        self.config = config
        self.data_loader = get_data_loader(config["data_config"])
        self.gan = model.GAN(config["model_config"])
        self.criterion = torch.nn.BCELoss()
        #Instantiante optimizers
        self.optimizer_g = instantiate_optimizer(self.gan.generator, config["train_config"], "generator")
        self.optimizer_d = instantiate_optimizer(self.gan.discriminator, config["train_config"], "discriminator")
        #Instantiate schedulers
        self.scheduler_g = instantiate_scheduler(self.optimizer_g, config["train_config"], "generator")
        self.scheduler_d = instantiate_scheduler(self.optimizer_d, config["train_config"], "discriminator")
        #Experiment tracking
        if config["eval_config"]["track_experiment"]:
            self.experiment = CustomExperimentTracker(config)
        else:
            self.experiment = None

    def train(self):
        for epoch in range(self.config["train_config"]["epochs"]):
            for i, data in tqdm(enumerate(self.data_loader, 0)):
                #Update discriminator: maximize log(D(x)) + log(1 - D(G(z))) where G(z) is the generated image given the random noise z.
                #Train with all-real batch
                self.gan.discriminator.zero_grad()
                real_data = data[0].view(-1, 784) #Reshapes [batch_size, 1, 28, 28] to [batch_size, 784]. This has to be automated to handle various shapes of data.
                true_label = torch.full((real_data.size(0),), 1, dtype = torch.float)
                truth_output = self.gan.discriminator(real_data).view(-1)
                errD_real = self.criterion(truth_output, true_label)
                errD_real.backward()

                #Train with all-fake batch
                noise = torch.randn(self.config["data_config"]["batch_size"], self.config["model_config"]["generator"]["latent_size"])
                fake_data = self.gan.generator(noise)
                fake_label = torch.full((fake_data.size(0),), self.gan.fake_label, dtype = torch.float)
                fake_output = self.gan.discriminator(fake_data).view(-1)
                errD_fake = self.criterion(fake_output, fake_label)
                errD_fake.backward()

                self.optimizer_d.step()
                self.experiment.log_gradients(self.optimizer_d, "discriminator")
                self.scheduler_d.step()

                if i % self.config["train_config"]["k"] == 0:
                    self.gan.generator.zero_grad()
                    gen_fake_data = self.gan.generator(noise)
                    gen_fake_output = self.gan.discriminator(gen_fake_data).view(-1)
                    #Update generator: maximize log(D(G(z)))
                    gen_label = torch.full((fake_data.size(0),), self.gan.real_label, dtype = torch.float)
                    #We use the same fake_data output from the discriminator to update the generator
                    errG = self.criterion(gen_fake_output, gen_label)
                    errG.backward()

                    self.optimizer_g.step()
                    self.experiment.log_gradients(self.optimizer_g, "generator")
                    self.scheduler_g.step()

                if self.experiment is not None:
                    self.experiment.total_iterations += 1
                    self.experiment.log_metrics(errD_real, errD_fake, errG)
                    self.experiment.save_samples(fake_data)
                    self.experiment.save_checkpoint(
                        self.gan, self.optimizer_g, self.optimizer_d, self.scheduler_g, self.scheduler_d, errG, errD_real + errD_fake, epoch
                        )

            print(f"Epoch {epoch} completed")

        print("Training completed")





def main():

    parser = argparse.ArgumentParser(description="GAN Training")

    parser.add_argument("--config", type=str, required=True, help="Configuration file.")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    trainer = GanTrainer(config)

    trainer.train()

    return



if __name__ == "__main__":
    main()

