import datetime
import os
import logging
import json
from torchvision.utils import save_image
import torch

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


class CustomExperimentTracker:
    def __init__(self, config):
        self.config = config
        self.id = self._create_experiment_id()
        self.directory = self._create_experiment_directory()
        #Loggers
        self.metrics_logs_path, self.metrics_logs = self._create_metrics_log()
        self.gradients_logs_path, self.gradients_logs = self._create_gradients_log()

        self.sample_path = self._create_sample_directory()
        self.checkpoint_path = self._create_checkpoint_directory()
        self.total_iterations = 0

        self._update_and_save_config()

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
    
    def _create_metrics_log(self):
        metric_logs_path = os.path.join(self.directory, "metrics_logs.txt")
        metrics_logger = setup_logger("metrics_logger", metric_logs_path)
        return metric_logs_path, metrics_logger

    
    def log_metrics(self, errD_real, errD_fake, errG):
        if self.total_iterations % self.config["eval_config"]["metrics_log_interval"] == 0:
            self.metrics_logs.info({"Iteration": self.total_iterations,"Discriminator Real Loss": errD_real.item(), "Discriminator Fake Loss": errD_fake.item(), "Discriminator Total Loss": errD_real.item() + errD_fake.item(), "Generator Loss": errG.item()})

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


    def _create_sample_directory(self):
        sample_path = os.path.join(self.directory, "samples")
        os.mkdir(sample_path)
        return sample_path
    
    def save_samples(self, samples):
        if self.total_iterations % self.config["eval_config"]["sample_interval"] == 0:
            sample_path = os.path.join(self.sample_path, f"sample_{self.total_iterations}.png")
            save_image(samples.view(-1, 1, 28, 28), sample_path)


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

    def _update_and_save_config(self):
        self.config["eval_config"]["experiment_directory"] = self.directory
        self.config["eval_config"]["experiment_id"] = self.id
        self.config["eval_config"]["metrics_logs_path"] = self.metrics_logs_path
        self.config["eval_config"]["gradients_logs_path"] = self.gradients_logs_path
        self.config["eval_config"]["sample_path"] = self.sample_path
        self.config["eval_config"]["checkpoint_path"] = self.checkpoint_path
        
        with open(os.path.join(self.directory, "config.json"), "w") as f:
            json.dump(self.config, f, indent = 4)
