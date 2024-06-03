# em-gan
Training a Generative Adversarial Networks for data augmentation in Cryo-EM Single Particle Analysis

It includes both fully connected (FC) and convolutional (Conv) generator architectures, a fully connected discriminator, and the GAN training loop with progress bars for both epochs and steps. Here’s a brief explanation and any minor tweaks to ensure compatibility and clarity:

## Data Preprocessing:

Converts .mrcs files to PNG format.
Augments data by rotating images if specified.

## Generator Models:

### FCGenerator: 
A fully connected generator that outputs an image of a specified size.

### ConvGenerator: 
A convolutional generator that uses transpose convolution layers to create images, with an optional boost mode for enhanced capacity.

## Discriminator:
A fully connected discriminator that takes an image, flattens it, and classifies it as real or fake.

## Training Loop:
Uses RMSprop optimizer with learning rate schedulers.
Includes progress bars using tqdm for epochs and a custom progress bar for steps.
Saves the model state and generated images at specified intervals.