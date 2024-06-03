import os
import torch
import math
import uuid
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
import argparse
from gan import FCGenerator  # assuming the FCGenerator class is defined in gan.py

# Define the function to rotate a 3D tensor
def rotate_3d_tensor(tensor, angle, axis):
    angle_rad = torch.tensor(angle * math.pi / 180.0)  # Convert to radians
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    
    if axis == 'X':
        rotation_matrix = torch.tensor([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ]).to(tensor.device)
    elif axis == 'Y':
        rotation_matrix = torch.tensor([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ]).to(tensor.device)
    elif axis == 'Z':
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ]).to(tensor.device)

    # Get the grid coordinates
    x = torch.linspace(-1, 1, tensor.size(0)).to(tensor.device)
    y = torch.linspace(-1, 1, tensor.size(1)).to(tensor.device)
    z = torch.linspace(-1, 1, tensor.size(2)).to(tensor.device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    # Apply the rotation
    rotated_grid = torch.mm(grid, rotation_matrix.T).reshape(tensor.size(0), tensor.size(1), tensor.size(2), 3)

    # Interpolate the tensor values at the rotated grid coordinates
    rotated_tensor = torch.nn.functional.grid_sample(tensor.unsqueeze(0).unsqueeze(0), rotated_grid.unsqueeze(0), align_corners=True)
    rotated_tensor = rotated_tensor.squeeze(0).squeeze(0)
    
    return rotated_tensor

def generate_images(generator_path, dataset_path, output_dir, image_size=64):
    # Generate a unique ID
    unique_id = uuid.uuid4()

    # Define step angle for each axis
    images_per_axis = 4
    step_angle = 360 / images_per_axis

    # Define the transformation to apply to the images
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    # Load the dataset
    png_directory = os.path.join(dataset_path, 'png')
    dataset = datasets.ImageFolder(root=png_directory, transform=transform)

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=images_per_axis * 3, shuffle=True)

    # Get a batch of real images
    real_images, _ = next(iter(dataloader))

    # Create a grid of real images
    image_grid = make_grid(real_images, nrow=3)

    # Save the grid of real images
    save_image(image_grid, os.path.join(output_dir, 'real', f'real_images_grid_{unique_id}.png'))

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the generator
    generator = FCGenerator(image_size).to(device)
    state_dict = torch.load(generator_path, map_location=device)
    generator.load_state_dict(state_dict['generator_state_dict'])
    generator.eval()

    # Define dimensions
    cube_dim = (10, 10, 10)  # Dimensions of the 3D noise cube (10x10x10)

    # Generate a single 3D random noise cube
    noise_cube = torch.randn(cube_dim).to(device)

    # List to store the generated images
    generated_images = []

    for i in range(images_per_axis):
        # Define angle for rotation
        angle = i * step_angle

        # List to store the 2D projections
        projections = []

        # Loop through axes, rotate, project, and reshape
        for axis_index, axis in enumerate(['X', 'Y', 'Z']):
            # Rotate the noise cube
            rotated_noise_cube = rotate_3d_tensor(noise_cube, angle, axis)

            # Project the 3D noise cube to 2D planes
            noise_2d_projection = rotated_noise_cube.sum(dim=axis_index)  # Project along the current axis

            # Reshape the 2D projection to 1D vector
            noise_1d = noise_2d_projection.view(-1)  # Resulting shape will be (100,)

            # Add the 1D projection to the list
            projections.append(noise_1d)

        # Convert the list of 1D projections to a tensor
        projections_tensor = torch.stack(projections, dim=0)  # Resulting shape will be (3, 100)

        # Generate a batch of images
        with torch.no_grad():
            fake_images = generator(projections_tensor)

        # Rescale the images to the range [0, 1]
        fake_images = (fake_images + 1) / 2

        # Add the generated images to the list
        generated_images.extend(fake_images)

    # Convert the list of generated images to a tensor
    images_tensor = torch.stack(generated_images, dim=0)  # Resulting shape will be (images_per_axis * 3, C, H, W)

    # Create a grid of images
    image_grid = make_grid(images_tensor, nrow=3)  # The grid will have 3 columns (X, Y, Z)

    # Save the grid of images with a unique file name
    save_image(image_grid, os.path.join(output_dir, 'fake', f'fake_images_grid_{unique_id}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images with a GAN.')
    parser.add_argument('generator_path', type=str, help='Path to the generator state.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset.')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save the generated images.')
    parser.add_argument('--image-size', type=int, default=64, help='The size of the images to generate.')
    args = parser.parse_args()

    generate_images(args.generator_path, args.dataset_path, args.output_dir, args.image_size)