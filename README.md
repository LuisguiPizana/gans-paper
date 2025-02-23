# GANs-Papers Repository

This repository contains three main branches that each train a different GAN model. We use UV as our package manager to ensure consistent dependency management across branches. The first section of this README explains how to set up the environment, train the model, and launch the Streamlit app. The second section will explain with more detail the content of each branch and share some results on the MNIST dataset.

---

## General Setup

### 1. Clone the Repository

Clone the repository and move to the repository directory:

```bash
git clone git@github.com:LuisguiPizana/gans-paper.git
cd GANs-Paper
```

### 2. Install UV

Dependencies and enviroments are managed through UV. You can install it with pip.
```bash
pip install uv
```

### 3. Installing Dependencies
There is no need to create a virtual enviroment manually, UV will handle it automaticaly creating the .venv folder in your repository, just run:
```bash
uv sync
```

If you prefere you can also create your own enviroment and activate it. In this case make sure your enviroment is active before running the previous command. 

### 4. Training the Model
In order to train the model, run the following command from your repository root:
```bash
uv run ./src/train.py --config ./config/config.json
```

### 5. Training Evaluation and Experiment Tracking

To launch the Streamlit experiment tracking app run:
```bash
uv run streamlit run ./metric_dashboard/app.py
```
The functionalities of this custom experiment tracking are limited. Feel free to use another tool like MLFlow or TensorBoard instead.

---
## Experiments on Branches

All the experiments were done using the MNIST dataset. The objective of the project was to test four different architectures presented in the GAN's literature, these being the original architecture from the Generative Adversarial Nets paper, the DCGAN, the Conditional GAN and an MLPGAN. 

### 1. Maxout GAN

![Training Sample - Conditional GAN Architecture](.\images\maxout-gan-sample.png)



### 2. 
![Training Sample - DCGAN Architecture](.\images\dcgan-sample.png)



### 3. 


### 4. Conditional GAN

asdfasdf  

![Training Sample - Conditional GAN Architecture](.\images\conditional-gan-sample.png)

