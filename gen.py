import argparse
import uuid
import torch
from torchvision.utils import save_image
from gan import Generator  # assuming the Generator class is defined in gan.py

def generate_image(generator_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the generator
    generator = Generator().to(device)
    state_dict = torch.load(generator_path, map_location=device)
    generator.load_state_dict(state_dict['generator_state_dict'])
    generator.eval()

    while True:
        # Generate a single image
        z = torch.randn(1, 100).to(device)  # assuming the input noise z is of size 100
        with torch.no_grad():
            fake_image = generator(z)

        # Rescale the image to the range [0, 1]
        fake_image = (fake_image + 1) / 2

        # Generate a unique ID for the image
        image_id = uuid.uuid4()

        # Save the generated image
        save_image(fake_image, f'generated_image_{image_id}.png')

        # Ask the user if they want to generate another image
        user_input = input("Do you want to generate another image? (yes/no): ")
        if user_input.lower() != 'yes':
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images with a GAN.')
    parser.add_argument('generator_path', type=str, help='Path to the generator state.')
    args = parser.parse_args()

    generate_image(args.generator_path)