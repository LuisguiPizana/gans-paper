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

### 2. Create and Activate a Virtual Environment

Dependencies are managed through UV. I recommend using Python's built-in venv to created an isolated enviroment located at the repository root.

#### Create a Virtual Enviroment
```bash
python -m venv .venv
```

#### Activate It
- Windows
```bash
.venv\Scripts\activate
```

- macOS/Linux:
```bash
source .venv/bin/activate
```

### 3. Install Dependencies using UV
The uv.yaml file contains the dependencies with their respective versions. Make sure it's in the root folder. With the activated enviroment run:

```bash
uv install
```

### 4. Training the Model
In order to train the model, run the following command from your repository root:
```bash
uv run train
```

It calls the train.py file located in the src directory. You can run the file directly but make sure to add as a flag the config.json file located inside the config directory.

### 5. Training Evaluation and Experiment Tracking

To launch the Streamlit experiment tracking app run:
```bash
run uv run lauch-st-app
```
The functionalities of this custom experiment tracking are limited. Feel free to use another tool like MLFlow or TensorBoard instead.

---
## Experiments on Branches

All the experiments were done using the MNIST dataset. The objective of the project was to test four different architectures presented in the GAN's literature, these being the original architecture from the Generative Adversarial Nets paper, the DCGAN, the Conditional GAN and an MLPGAN. 

### 1. 



### 2. 




### 3. 


### 4. 

