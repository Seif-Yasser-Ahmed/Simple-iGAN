import torch
import gradio as gr
from huggingface_hub import hf_hub_download
import numpy as np
from DCGAN import Generator
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
# Load pre-trained generator model


def load_generator():
    model_path = hf_hub_download(
        repo_id="Seif-Yasser/iDCGAN-iGAN", filename="Space/iDCGAN-iGAN/models/gan_generator.pth")
    generator = Generator(64)
    generator.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
    generator.eval()
    # generate an image and save it
    # latent_vector = torch.randn(1, 64, 1, 1)
    # with torch.no_grad():
    # generated_image = generator(latent_vector)
    # image_tensor = (generated_image + 1) / 2
    # image_unflat = image_tensor.detach().cpu()
    # image_unflat = image_unflat.squeeze().permute(
    # 0, 2, 3, 1).numpy()  # Convert image to NumPy for visualization
    # image_grid = make_grid(image_unflat[:num_images], nrow=5)
    # plt.imsave("generated_image2.png", image_unflat)

    return generator


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imsave("generated_image2.png",
               image_grid.permute(1, 2, 0).squeeze().numpy())
    return image_grid.permute(1, 2, 0).squeeze().numpy()


# Initialize the generator model
generator = load_generator()

# Function to generate images

tens = torch.rand(10, 1, 64, 1, 1)


def generate_images(latent_dim):
    # Create latent vector based on the slider value
    # latent_vector = torch.randn(1, 64, 1, 1)
    latent_vector = tens[latent_dim]
    # with torch.no_grad():
    generated_image = generator(latent_vector)
    image = show_tensor_images(generated_image)
    return image
    # Convert to NumPy
    generated_image = generated_image.squeeze().permute(0, 2, 3, 1).numpy()
    # save the generated image
    plt.imsave("generated_image3.png", generated_image)

    return generated_image

# Gradio interface for live image generation


def gradio_interface(latent_dim):
    print(latent_dim)
    image = generate_images(latent_dim-1)
    return image


# Create Gradio Interface with a live slider
interface = gr.Interface(
    fn=gradio_interface,
    # live=True enables real-time updates
    inputs=[gr.Slider(minimum=1, maximum=10, step=1,
                      label="Latent Dimension")],
    outputs="image",
    title="iDCGAN Image Generator",
    description="Adjust the slider to generate a new image.",
    live=True  # Enable real-time updates
)

# Launch the Gradio interface
interface.launch()
