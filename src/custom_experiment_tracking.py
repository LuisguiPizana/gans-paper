import datetime
import os
import logging
import json
from torchvision.utils import save_image
import torch
from torchvision.models import inception_v3
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import entropy


def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger with a single file handler."""
    handler = logging.FileHandler(log_file)  
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if not logger.handlers:  # Check if the logger already has handlers
        logger.setLevel(level)
        logger.addHandler(handler)
    
    return logger

#############################################################################
# Evaluation
#############################################################################

def inception_score(images, batch_size=32, resize=False, splits=10):
    N = len(images)

    # Load the Inception v3 model
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()

    def get_pred(x):
        if resize:
            if len(x.shape) == 2:
                x = x.view(x.size(0), 1, int(x.size(1)**0.5), int(x.size(1)**0.5)) #Assuming the input is a square image
            elif len(x.shape) == 3:  
                x = x.unsqueeze(1) 
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1) #If the input is grayscale, repeat the image 3 times to get 3 channels.
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
    
    # Prepare data loader
    dataloader = DataLoader(images, batch_size=batch_size)

    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch_size_i = batch.size(0)
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

    # Compute the mean kl-divergence
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


class CustomExperimentTracker:
    def __init__(self, config):
        self.config = config
        self.id = self._create_experiment_id()
        self.directory = self._create_experiment_directory()
        #Loggers
        self.metrics_logs_path, self.metrics_logs = self._create_metrics_log()
        self.gradients_logs_path, self.gradients_logs = self._create_gradients_log()
        self.is_logs_path, self.is_logs = self._create_inception_logs()

        self.sample_path = self._create_sample_directory()
        self.checkpoint_path = self._create_checkpoint_directory()
        self.total_iterations = 0

        self._update_and_save_config()

    def generate_images(self, gan_generator, num_images):
        gan_generator.eval()
        latent_dim = self.config["model_config"]["generator"]["latent_size"]
        noise = torch.randn(num_images, latent_dim)
        print("Num images: ", num_images)
        print("Shape of noise: ", noise.shape)
        with torch.no_grad():
            fake_images = gan_generator(noise)
        return fake_images

    # Create Experiment
    def _create_experiment_id(self):
        """
        Create an experiment ID based on the current date and experiment number in date.        
        """
        today = datetime.datetime.now().strftime("%Y-%m-%d")

        all_entries = os.listdir(self.config["eval_config"]["experiments_dir"])
        todays_dirs = [entry for entry in all_entries if entry.split("_")[0] == today]

        experiment_id = f"{today}_{len(todays_dirs) + 1}"
        return experiment_id
    
    def _create_experiment_directory(self):
        experiment_path = os.path.join(self.config["eval_config"]["experiments_dir"], self.id)
        os.mkdir(experiment_path)        
        return experiment_path
    
    # Training Loggers
    def _create_metrics_log(self):
        metric_logs_path = os.path.join(self.directory, "metrics_logs.txt")
        metrics_logger = setup_logger("metrics_logger", metric_logs_path)
        return metric_logs_path, metrics_logger

    
    def log_metrics(self, errD_real, errD_fake, errG, lrD, lrG):
        if self.total_iterations % self.config["eval_config"]["metrics_log_interval"] == 0:
            self.metrics_logs.info({"Iteration": self.total_iterations,"Discriminator Real Loss": errD_real.item(), "Discriminator Fake Loss": errD_fake.item(),
                                    "Discriminator Total Loss": errD_real.item() + errD_fake.item(), "Generator Loss": errG.item(), "Discriminator LR": lrD, "Generator LR": lrG})

    # Inception Score
    def _create_inception_logs(self):
        is_logs_path = os.path.join(self.directory, "inception_score_logs.txt")
        is_logger = setup_logger("inception_score_logger", is_logs_path)
        return is_logs_path, is_logger

    def _compute_inception_score(self, gan_generator):
            fake_images = self.generate_images(gan_generator, self.config["eval_config"]["num_inception_images"])
            print("Fake images shape: ", fake_images.shape)
            is_mean, is_std = inception_score(fake_images, batch_size = self.config["data_config"]["batch_size"], resize=True, splits=10)
            return is_mean, is_std
    
    def log_inception_score(self, gan_generator):
        if self.total_iterations % self.config["eval_config"]["is_log_interval"] == 0:
            is_mean, is_std = self._compute_inception_score(gan_generator)
            self.is_logs.info({"Iteration": self.total_iterations, "Inception Score Mean": is_mean, "Inception Score Std": is_std})

    # Gradient Loggers
    def _create_gradients_log(self):
        gradients_logs_path = os.path.join(self.directory, "gradients_logs.txt")
        gradients_logger = setup_logger("gradients_logger", gradients_logs_path)
        return gradients_logs_path, gradients_logger
    
    def log_gradients(self, optimizer, component = "generator"):
        if self.total_iterations % self.config["eval_config"]["gradients_log_interval"] == 0:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        grad_data = param.grad.data
                        norm = grad_data.norm().item()
                        mean = grad_data.mean().item()
                        std = grad_data.std().item()

                        self.gradients_logs.info({"Iteration" : self.total_iterations,  "Component": component, "Gradient Norm": norm, "Gradient Mean": mean, "Gradient Std": std})

    # Sampling
    def _create_sample_directory(self):
        sample_path = os.path.join(self.directory, "samples")
        os.mkdir(sample_path)
        return sample_path
    
    def save_samples(self, samples):
        if self.total_iterations % self.config["eval_config"]["sample_interval"] == 0:
            sample_path = os.path.join(self.sample_path, f"sample_{self.total_iterations}.png")
            save_image(samples.view(-1, 1, 28, 28), sample_path)

    # Checkpointing
    def _create_checkpoint_directory(self):
        checkpoint_path = os.path.join(self.directory, "checkpoints")
        os.mkdir(checkpoint_path)
        return checkpoint_path
    
    def save_checkpoint(self, model, optimizer_g, optimizer_d, lrs_g, lrs_d, loss_g, loss_d, epoch):
        
        if self.total_iterations % self.config["eval_config"]["checkpoint_interval"] == 0:
            checkpoint = {
                "generator": model.generator.state_dict(),
                "discriminator": model.discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "lrs_g": lrs_g,
                "lrs_d": lrs_d,
                "loss_g": loss_g,
                "loss_d": loss_d,
                "epoch": epoch
            }
            checkpoint_path = os.path.join(self.checkpoint_path, f"checkpoint_it_{epoch}.pt")
            torch.save(checkpoint, checkpoint_path)

    # Update Config
    def _update_and_save_config(self):
        self.config["eval_config"]["experiment_directory"] = self.directory
        self.config["eval_config"]["experiment_id"] = self.id
        self.config["eval_config"]["metrics_logs_path"] = self.metrics_logs_path
        self.config["eval_config"]["gradients_logs_path"] = self.gradients_logs_path
        self.config["eval_config"]["sample_path"] = self.sample_path
        self.config["eval_config"]["checkpoint_path"] = self.checkpoint_path
        
        with open(os.path.join(self.directory, "config.json"), "w") as f:
            json.dump(self.config, f, indent = 4)
