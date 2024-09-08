---
title: IDCGAN IGAN
emoji: 🖼
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 4.42.0
app_file: app.py
pinned: false
license: mit
---

You can try it out by clicking [here](https://huggingface.co/spaces/Seif-Yasser/iDCGAN-iGAN)

# iDCGAN-iGAN

## Model Description
iDCGAN-iGAN is a Deep Convolutional Generative Adversarial Network (DCGAN) that was trained to generate images based on input noise. The model was implemented using PyTorch and is a variant of the DCGAN architecture.

This model is suitable for tasks such as image generation and creative applications, where synthetic imagery is required.

## Training Procedure
The iDCGAN-iGAN was trained using adversarial training where a generator network learns to create realistic images, and a discriminator network learns to distinguish between real and fake images. The training process aims to improve both networks over time.

- **Generator Architecture**: Convolutional layers with transposed convolutions, batch normalization, and ReLU activations.
- **Discriminator Architecture**: Convolutional layers with LeakyReLU activations and batch normalization.

### Hyperparameters:
- Optimizer: `Adam`
- Learning Rate: `0.0002`
- Beta1: `0.5`
- Beta2: `0.999`
- device: `cuda`


## Model Output
The model generates a synthetic image given a latent vector as input. The latent vector is a 1D tensor sampled from a Gaussian distribution.

## How to Use the Model

Here's an example of how to load and use the generator model:

```python
import torch
from huggingface_hub import hf_hub_download
z_dim=64
model_path = hf_hub_download(
    repo_id="Seif-Yasser/iDCGAN-iGAN", filename="Space/iDCGAN-iGAN/models/gan_generator.pth")
generator = Generator(z_dim)
generator.load_state_dict(torch.load(
    model_path, map_location=torch.device('cpu')))
generator.eval()

# Generate an image
latent_vector = torch.randn(1, 64, 1, 1)  # Example latent vector
with torch.no_grad():
    generated_image = model(latent_vector)
