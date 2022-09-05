# The module jupytext is used to treat this .py file as a jupyter notebook file. To keep the output after every session, go to "File" -> "Jupytext" -> "Pair Notebook with ipynb document". This generates a file PY_FILENAME.ipynb.


# # Imports

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
import torch.nn as nn

# # Load MNIST Dataset

# +
folder_path = "./data/mnist"

train_dataset = datasets.MNIST(root=folder_path, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root=folder_path, train=False, transform=transforms.ToTensor(), download=True)
# -

# # Data Exploration

# +
train_image_data = train_dataset.data.numpy()
train_labels = train_dataset.targets.numpy()

assert (train_image_data.shape == np.array([60000, 28, 28])).all()
assert (train_labels.shape == np.array([60000,])).all()


# -

def plot_images(images):
    """Plots the first 9 images."""
    amount_images_to_show = 9
    grid_shape = np.array([3, 3])
    for image_index in range(amount_images_to_show):
        plt.subplot(grid_shape[0], grid_shape[1], image_index + 1)
        plt.xticks([])
        plt.yticks([])
        image_to_show = images[image_index]
        plt.imshow(image_to_show, cmap="gray")
    plt.show()


plot_images(train_image_data)

assert (train_labels[:9] == np.array([5, 0, 4, 1, 9, 2, 1, 3, 1])).all()


# # Forward Noise Process

def add_noise(image, beta):
    alpha = 1 - beta
    alpha_cum = torch.prod(alpha)
    random_array = torch.randn(image.shape)
    noisy_image = random_array * (1 - alpha_cum) + torch.sqrt(alpha_cum) * image
    return noisy_image


variances = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
noisy_images = add_noise(train_dataset.data, variances)
plot_images(noisy_images)

variances = torch.ones(900) * 0.015
image = train_dataset.data[0]
noisy_images = torch.zeros([9, image.shape[0], image.shape[1]])
for i in range(9):
    noisy_image = add_noise(image, variances[:i * 100 + 1])
    noisy_images[i] = noisy_image
plot_images(noisy_images)


# # Noise Prediction Model Implementation
# Uses a UNET architecture to predict the noise of given noisy images.

# +
class Conv2dBlock(nn.Module):
    def __init__(self, amount_channels_input, amount_channels_output):
        super(Conv2dBlock, self).__init__()
        self.layers = []
        self.layers.append(nn.Conv2d(amount_channels_input, amount_channels_output, 3, padding=1))
        self.layers.append(nn.BatchNorm2d(amount_channels_output))
        self.layers.append(nn.Conv2d(amount_channels_output, amount_channels_output, 3, padding=1))
        self.layers.append(nn.BatchNorm2d(amount_channels_output))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, amount_channels_input, amount_channels_output):
        super(EncoderBlock, self).__init__()
        self.conv_block = Conv2dBlock(amount_channels_input, amount_channels_output)
        self.downsampling_layer = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv_block(x)
        skip_values = x
        x = self.downsampling_layer(x)
        return x, skip_values

    
class DecoderBlock(nn.Module):
    def __init__(self, amount_channels_input, amount_channels_output):
        super(DecoderBlock, self).__init__()
        self.conv_block = Conv2dBlock(amount_channels_output * 2, amount_channels_output)
        self.upsampling_layer = nn.ConvTranspose2d(amount_channels_input, amount_channels_output, 2, 2)
        
    def forward(self, x, skip_values):
        x = self.upsampling_layer(x)
        x = torch.cat([x, skip_values], axis=1)
        x = self.conv_block(x)
        return x
    

class NoisePredictionUnet(nn.Module):
    def __init__(self, amount_channels):
        super(NoisePredictionUnet, self).__init__()
        self.layers = []
        amount_channels_with_timestep = amount_channels + 1
        self.layers.append(EncoderBlock(amount_channels_with_timestep, amount_channels * 2))
        self.layers.append(Conv2dBlock(amount_channels * 2, amount_channels * 4))
        self.layers.append(DecoderBlock(amount_channels * 4, amount_channels * 2))
        self.layers.append(nn.Conv2d(amount_channels * 2, amount_channels, 1))

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, noisy_images, timestep):
        x = torch.cat([noisy_images, torch.ones(noisy_images.shape) * timestep], axis = 1)
        x, skip_values = self.layers[0](x)
        x = self.relu(x)
        x = self.layers[1](x)
        x = self.relu(x)
        x = self.layers[2](x, skip_values)
        x = self.relu(x)
        x = self.layers[3](x)
        x = self.sigmoid(x)
        return x


# -

# # Training

batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = NoisePredictionUnet(1)
for data, labels in train_loader:
    timestep = 3
    predicted_noises = model(data, timestep)
    assert predicted_noises.shape == data.shape
    break

# +
variances = torch.ones(1000) * 0.015
timesteps = torch.Tensor(range(variances.shape[0]))

epochs = 10
for epoch_index in range(epochs):
    for batch_index, train_data_batch in enumerate(train_loader):
        ...

# # Evaluation


# -

# # Sampling

# +
def denoising_process(images, beta, noise_predictor, simple_variance=False):
    "sample image"
    
    images_size = images.shape
    timesteps = beta.shape[0] - 1
    alpha = 1 - beta
    alpha_cum = torch.cumprod(alpha, dim=0)
    if simple_variance:
        variances = beta
    else:
        alpha_cum_t_minus_1 = torch.cat([torch.Tensor([0]), alpha_cum[:-1]], axis=0)
        variances = (1-alpha_cum_t_minus_1)/(1-alpha_cum)
        variances = variances * beta
    
    x_t = images
    
    for timestep in range(timesteps, 0, -1):
        predicted_noise = noise_predictor(x_t, timestep)
        z = torch.normal(torch.zeros(images_size), torch.ones(images_size))
        if timestep == 1:
          z = torch.zeros(images_size)
        x_t = variances[timestep] * z + (x_t - (1-alpha[timestep])/torch.sqrt(1-alpha_cum[timestep])*predicted_noise) \
          / torch.sqrt(alpha[timestep])     
    return x_t

amount_channels = 1
test_images, test_labels = next(iter(test_loader)) 
test_image = test_images[0] 
test_label = test_labels[0] 

def fake_noise_pred(image, timestep):
    return torch.normal(torch.zeros_like(image), torch.ones_like(image))

image_size = test_image.shape
beta = torch.ones(10) * 0.15

samples = denoising_process(torch.ones(test_images.shape), beta, fake_noise_pred)

assert samples.shape == test_images.shape

plt.subplot(1,2,1)
plt.imshow(samples[0][0], cmap="gray")
plt.subplot(1,2,2)
plt.imshow(samples[1][0], cmap="gray")
plt.show()