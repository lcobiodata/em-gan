import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import mrcfile
from PIL import Image
import argparse
import sys
from tqdm import tqdm

# Data preprocessing
def convert_mrcs_to_png(mrcs_directory, png_directory, augment_data=False):
    """
    Convert .mrcs files to PNG format
    :param mrcs_directory: Directory containing .mrcs files
    :param png_directory: Directory to save PNG files
    :param augment_data: Boolean flag to decide whether to augment data by rotating images
    """
    png_directory = os.path.join(png_directory, 'class1')
    os.makedirs(png_directory, exist_ok=True)
    for filename in os.listdir(mrcs_directory):
        if filename.endswith(".mrcs"):
            with mrcfile.open(os.path.join(mrcs_directory, filename), mode='r') as mrc:
                data = mrc.data
                for i in range(data.shape[0]):
                    # Normalize pixel values to [0, 255]
                    normalized_data = ((data[i] - data[i].min()) * (255 - 0) / (data[i].max() - data[i].min())) + 0
                    img = Image.fromarray(normalized_data.astype(np.uint8)).convert('L')
                    img.save(os.path.join(png_directory, f"{filename}_{i}.png"))
                    if augment_data:
                        # Rotate the image by 90, 180, and 270 degrees and save
                        for angle in [90, 180, 270]:
                            img_rotated = img.rotate(angle)
                            img_rotated.save(os.path.join(png_directory, f"{filename}_{i}_rot{angle}.png"))
    print(f"Converted .mrcs files from {mrcs_directory} to PNG format in {png_directory}")

def get_smallest_image_size(png_directory):
    """
    Get the smallest width and height of the images in the given directory
    :param png_directory: Directory containing the images
    :return: smallest_width, smallest_height
    """
    image_files = [f for f in os.listdir(png_directory) if f.endswith('.png')]

    smallest_width = float('inf')
    smallest_height = float('inf')

    for image_file in image_files:
        with Image.open(os.path.join(png_directory, image_file)) as img:
            width, height = img.size
            smallest_width = min(smallest_width, width)
            smallest_height = min(smallest_height, height)

    return smallest_width, smallest_height

# Fully Connected Generator
class FCGenerator(nn.Module):
    def __init__(self, image_size):
        super(FCGenerator, self).__init__()
        self.image_size = image_size
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, self.image_size * self.image_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, self.image_size, self.image_size)

class ConvGenerator(nn.Module):
    def __init__(self, image_size):
        super(ConvGenerator, self).__init__()
        self.init_size = image_size // 4

        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.model = nn.Sequential(
            nn.Linear(self.image_size * self.image_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Training function
def train_gan(generator, discriminator, dataloader, epochs, device, save_path=None):
    """
    Train GAN model
    :param generator: Generator model
    :param discriminator: Discriminator model
    :param dataloader: DataLoader
    :param epochs: Number of epochs
    :param device: Device (cuda or cpu)
    :param save_path: Path to save and load GAN state
    """
    criterion = nn.BCELoss()
    optimizer_g = optim.RMSprop(generator.parameters(), lr=0.0002)
    optimizer_d = optim.RMSprop(discriminator.parameters(), lr=0.0002)

    # Define the learning rate schedulers
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=30, gamma=0.1)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=30, gamma=0.1)

    start_epoch = 0

    # Load previous training state if save_path exists
    if save_path and os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        start_epoch = checkpoint['epoch']

    # tqdm progress bar for epochs
    with tqdm(total=epochs, initial=start_epoch, desc="Epochs", unit="epoch", position=1, leave=True) as pbar_epochs:
        for epoch in range(start_epoch, epochs):
            for i, (real_images, _) in enumerate(dataloader):
                real_images = real_images.to(device)
                batch_size = real_images.size(0)

                # real_labels = torch.ones(batch_size, 1).to(device)
                real_labels = torch.full((batch_size, 1), 0.9, device=device)  # Label smoothing
                # fake_labels = torch.zeros(batch_size, 1).to(device)
                fake_labels = torch.full((batch_size, 1), 0.1, device=device)  # Label smoothing

                # Train discriminator
                fake_images = generator(torch.randn(batch_size, 100).to(device))

                outputs = discriminator(real_images)
                d_loss_real = criterion(outputs, real_labels)

                outputs = discriminator(fake_images.detach())
                d_loss_fake = criterion(outputs, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()

                # Train generator
                fake_images = generator(torch.randn(batch_size, 100).to(device))
                outputs = discriminator(fake_images)
                g_loss = criterion(outputs, real_labels)

                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()

                # Simple progress bar for steps
                progress_bar(i, len(dataloader), epoch, epochs, d_loss.item(), g_loss.item())

            # Step the learning rate schedulers
            scheduler_g.step()
            scheduler_d.step()

            # Save model state at the end of each epoch
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
            }, save_path if save_path else "gan_state.pth")

            print("\n", end="")
            pbar_epochs.update(1)

            # Save real and fake images
            if (epoch + 1) % 10 == 0:
                save_images(real_images, fake_images, epoch + 1, batch_size)

# Progress bar
def progress_bar(iteration, total, epoch, epochs, d_loss, g_loss, bar_length=50):
    """
    Print progress bar
    :param iteration: Current iteration
    :param total: Total iterations
    :param epoch: Current epoch
    :param epochs: Total epochs
    :param d_loss: Discriminator loss
    :param g_loss: Generator loss
    :param bar_length: Length of progress bar
    """
    progress = (iteration + 1) / total
    arrow = '#' * int(round(progress * bar_length))
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f'\rEpoch [{epoch}/{epochs}], Step [{iteration + 1}/{total}], d_loss: {d_loss:.2f}, g_loss: {g_loss:.2f}, Progress: [{arrow + spaces}] {progress * 100:.2f}%')
    sys.stdout.flush()

# Save images
def save_images(real_images, fake_images, epoch, batch_size):
    """
    Save real and fake images
    :param real_images: Real images
    :param fake_images: Fake images
    :param epoch: Current epoch
    :param batch_size: Batch size
    """
    os.makedirs('./images/real', exist_ok=True)
    os.makedirs('./images/fake', exist_ok=True)
    nrow = int(np.floor(np.sqrt(batch_size)))  # Automatically calculate nrow
    real_images = denorm(real_images)
    fake_images = denorm(fake_images)
    save_image(real_images, f'./images/real/real_images_{epoch}.png', nrow=nrow)
    save_image(fake_images, f'./images/fake/fake_images_{epoch}.png', nrow=nrow)

# Denormalize images
def denorm(x):
    """
    Denormalize images
    :param x: Input tensor
    :return: Denormalized tensor
    """
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Main script
def main(args):
    """
    Main function
    :param args: Command line arguments
    """
    input_data_dir = args.input_data_dir
    mrcs_directory = os.path.join(input_data_dir, 'mrcs')
    png_directory = os.path.join(input_data_dir, 'png')

    # Check if mrcs_directory exists
    if not os.path.exists(mrcs_directory):
        print(f"Error: {mrcs_directory} does not exist.")
        return

    # Check if png_directory exists, if not, create it
    if not os.path.exists(png_directory):
        os.makedirs(png_directory)

    # Convert .mrcs files to PNG format
    convert_mrcs_to_png(mrcs_directory, png_directory, augment_data=args.augment_data)

    # Get the smallest image size in the dataset
    actual_width, actual_height = get_smallest_image_size(png_directory)

    # Use the minimum of the actual image size and the user-specified size
    image_size = min(args.image_size, actual_width, actual_height)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Use the chosen image size
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.ImageFolder(root=png_directory, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.use_conv:
        generator = ConvGenerator(args.image_size).to(device)
    else:
        generator = FCGenerator(args.image_size).to(device)
    discriminator = Discriminator(args.image_size).to(device)

    epochs = args.epochs
    train_gan(generator, discriminator, dataloader, args.epochs, device, args.gan_state_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a GAN on mrcs images converted to PNG.')
    parser.add_argument('--input-data-dir', type=str, required=True, help='Directory of the input data')
    parser.add_argument('--image-size', type=int, default=64, help='The size of the images to generate')
    parser.add_argument('--augment-data', action='store_true', help='Augment data by rotating images')
    parser.add_argument('--use-conv', action='store_true', help='Use Convolutional GAN (DCGAN) instead of Fully Connected GAN (FCGAN)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--gan-state-path', type=str, default=None, help='Path to save and load GAN state')

    args = parser.parse_args()
    main(args)