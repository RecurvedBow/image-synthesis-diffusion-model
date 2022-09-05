# The module jupytext is used to treat this .py file as a jupyter notebook file. To keep the output after every session, go to "File" -> "Jupytext" -> "Pair Notebook with ipynb document". This generates a file PY_FILENAME.ipynb.

# # Imports

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
import torch.nn as nn

if torch.cuda.is_available():
    print("Using GPU.")
    device = "cuda"
else:
    print("Using CPU.")
    device = "cpu"

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

# +
def add_noise(image, beta):
    alpha = 1 - beta
    alpha_cum = torch.prod(alpha)
    return apply_noise(image, alpha_cum)

def apply_noise(image, alpha_cum):
    random_array = torch.randn(image.shape).to(device)
    noisy_image = random_array * (1 - alpha_cum) + torch.sqrt(alpha_cum) * image
    noisy_image = noisy_image.to(device)
    return noisy_image


# -

variances = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).to(device)
noisy_images = add_noise(train_dataset.data.to(device), variances).cpu()
plot_images(noisy_images)

variances = torch.ones(100).to(device) * 0.125
image = train_dataset.data[0].to(device)
noisy_images = torch.zeros([9, image.shape[0], image.shape[1]])
for i in range(9):
    index_stepsize = variances.shape[0] / 8
    noisy_image = add_noise(image, variances[:int(i * index_stepsize) + 1])
    noisy_images[i] = noisy_image
plot_images(noisy_images)


# # Noise Prediction Model Implementation
# Uses a UNET architecture to predict the noise of given noisy images.

# +
class Conv2dBlock(nn.Module):
    def __init__(self, amount_channels_input, amount_channels_output):
        super(Conv2dBlock, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(amount_channels_input, amount_channels_output, 3, padding=1),
                                              nn.BatchNorm2d(amount_channels_output),
                                              nn.ReLU(),
                                              nn.Conv2d(amount_channels_output, amount_channels_output, 3, padding=1),
                                              nn.BatchNorm2d(amount_channels_output),
                                              nn.ReLU())
    
    def forward(self, x):
        return self.layers(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, amount_channels_input, amount_channels_output):
        super(EncoderBlock, self).__init__()
        self.conv_block = Conv2dBlock(amount_channels_input, amount_channels_output)
        self.downsampling_layer = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.relu(x)
        skip_values = x
        x = self.downsampling_layer(x)
        x = self.relu(x)
        return x, skip_values

    
class DecoderBlock(nn.Module):
    def __init__(self, amount_channels_input, amount_channels_output):
        super(DecoderBlock, self).__init__()
        self.conv_block = Conv2dBlock(amount_channels_output * 2, amount_channels_output)
        self.upsampling_layer = nn.ConvTranspose2d(amount_channels_input, amount_channels_output, 2, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x, skip_values):
        x = self.upsampling_layer(x)
        x = self.relu(x)
        x = torch.cat([x, skip_values], axis=1)
        x = self.conv_block(x)
        x = self.relu(x)
        return x
    

class NoisePredictionUnet(nn.Module):
    def __init__(self, amounts_channels):
        super(NoisePredictionUnet, self).__init__()
        encoder_blocks = []
        decoder_blocks = []
        layers_between = []
        final_layers = []
        self.divisor = 2 ** (len(amounts_channels) - 2)
        for i, current_amount_channels in enumerate(amounts_channels):
            next_amount_channels = amounts_channels[i + 1]
            if i == 0:
                final_layers.append(nn.Conv2d(next_amount_channels, current_amount_channels, 1))
                final_layers.append(nn.Sigmoid())
                
                encoding_amount_channels_input = current_amount_channels + 1  # Due to the embedded timestep.
                encoding_amount_channels_output = next_amount_channels
                
                assert encoding_amount_channels_input <= encoding_amount_channels_output
                
                encoder_block = EncoderBlock(encoding_amount_channels_input, encoding_amount_channels_output)
                encoder_blocks.append(encoder_block)
                
            elif i == len(amounts_channels) - 2:
                layers_between.append(Conv2dBlock(current_amount_channels, next_amount_channels))
                
                decoding_amount_channels_input = next_amount_channels
                decoding_amount_channels_output = current_amount_channels
                
                assert decoding_amount_channels_input >= decoding_amount_channels_output
                
                decoder_block = DecoderBlock(decoding_amount_channels_input, decoding_amount_channels_output)
                decoder_blocks.insert(0, decoder_block)
                break
            else:                
                encoding_amount_channels_input = current_amount_channels
                encoding_amount_channels_output = next_amount_channels
                
                decoding_amount_channels_input = next_amount_channels
                decoding_amount_channels_output = current_amount_channels
                
                if i == 0:
                    encoding_amount_channels_input += 1
                
                assert encoding_amount_channels_input <= encoding_amount_channels_output
                assert decoding_amount_channels_input >= decoding_amount_channels_output
                
                encoder_block = EncoderBlock(encoding_amount_channels_input, encoding_amount_channels_output)
                encoder_blocks.append(encoder_block)
                
                decoder_block = DecoderBlock(decoding_amount_channels_input, decoding_amount_channels_output)
                decoder_blocks.insert(0, decoder_block)
                
        layers = encoder_blocks + layers_between + decoder_blocks + final_layers
        
        self.module_list = nn.ModuleList(layers)
    
    def apply_padding(self, x):
        pad_width = self.divisor - x.shape[3] % self.divisor
        pad_height = self.divisor - x.shape[2] % self.divisor
        
        pad_left = int(pad_width / 2)
        pad_right = pad_width - pad_left
        
        pad_up = int(pad_height / 2)
        pad_down = pad_height - pad_up
        padding = (pad_left, pad_right, pad_up, pad_down)
        x = torch.nn.functional.pad(x, padding, mode='constant', value=0)
        return x, padding
        
    def undo_padding(self, x, padding):
        (pad_left, pad_right, pad_up, pad_down) = padding
        return x[:, :, pad_left:-pad_right, pad_up:-pad_down]
    
    def forward(self, x):
        skip_values = []
        x, padding = self.apply_padding(x)
        for layer in self.module_list:
            if isinstance(layer, EncoderBlock):
                x, skip_value = layer(x)
                skip_values.append(skip_value)
            elif isinstance(layer, DecoderBlock):
                skip_value = skip_values.pop()
                x = layer(x, skip_value)
            else:
                x = layer(x)
                
        x = self.undo_padding(x, padding)
        assert len(skip_values) == 0
        return x


# -

def get_loss(predicted_noise):
    mse_loss = nn.MSELoss()
    expected_noise = torch.normal(torch.zeros(predicted_noise.shape), 1).to(device)
    return mse_loss(predicted_noise, expected_noise).to(device)


# # Training

batch_size = 60
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = NoisePredictionUnet([1, 16, 32, 64, 128, 256]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)


def get_X_train(images, variances):
    batch_size = images.shape[0]
    timesteps = torch.Tensor(range(variances.shape[0])).int().to(device)
    alphas_cum = torch.cumprod(variances, dim=0).to(device)
    noisy_images = torch.zeros(images.shape).to(device)
    noisy_images = torch.repeat_interleave(noisy_images, timesteps.shape[0], dim=0).to(device)
    timesteps_embedding = torch.zeros(noisy_images.shape).to(device)
    for timestep in timesteps:
        alpha_cum = alphas_cum[timestep]
        noisy_images[timestep*batch_size:(timestep + 1) * batch_size] = apply_noise(images, alpha_cum)
        timesteps_embedding[timestep*batch_size:(timestep + 1) * batch_size] = timestep
    noisy_images_with_timesteps = torch.cat([noisy_images, timesteps_embedding], dim=1).to(device)
    return noisy_images_with_timesteps


# +
variances = torch.ones(10).to(device) * 0.125 # Low amount of variances to verify training implementation.
timesteps = torch.Tensor(range(variances.shape[0])).to(device)

epochs = 100
epoch_mean_train_losses = []

min_epoch_train_loss = 0
amount_learning_rate_decays = 3
max_amount_epochs_cooldown = 2
amount_epochs_cooldown = 0

for epoch_index in range(epochs):
    batch_train_losses = []
    for batch_index, train_data_batch in enumerate(train_loader):
        images, _ = train_data_batch
        images = images.to(device)
        X_train = get_X_train(images, variances)
        predicted_noise = model(X_train)
        optimizer.zero_grad()
        loss = get_loss(predicted_noise)
        batch_train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
        if batch_index % 10 == 0:
            print("\r", end=f"Batch {batch_index + 1} - Training Loss: {batch_train_losses[-1]}")
        
    mean_loss = np.mean(batch_train_losses)
    epoch_mean_train_losses.append(mean_loss)
    print(f"\rEpoch {epoch_index + 1} - Average Training Loss: {epoch_mean_train_losses[-1]}")
    
    if amount_epochs_cooldown > 0:
        amount_epochs_cooldown -= 1
    if epoch_index == 0:
        min_epoch_train_loss = epoch_mean_train_losses[-1]
    elif amount_epochs_cooldown == 0:
        if min_epoch_train_loss >= epoch_mean_train_losses[-1]:
            min_epoch_train_loss = epoch_mean_train_losses[-1]
        else:
            if amount_learning_rate_decays == 0:
                print(f"Stopping Training.")
                break
            else:
                print(f"Loss did not decrease - decaying learning rate")
                amount_learning_rate_decays -= 1
                amount_epochs_cooldown = max_amount_epochs_cooldown
                scheduler.step()

# -

# # Evaluation

plt.plot(range(len(epoch_mean_train_losses)), epoch_mean_train_losses, linestyle="dashed")
plt.xlabel("Mean Training Loss")
plt.ylabel("Epoch")
plt.show()

# +
data_batch, _ = next(iter(train_loader))
timestep = 3
timestep_embedding = torch.ones(data_batch.shape) * timestep
x = torch.cat([data_batch, timestep_embedding], dim=1).to(device)
predicted_noises = model(x)

assert predicted_noises.shape == data_batch.shape

predicted_noise = predicted_noises[0].detach().cpu().numpy()[0]
plt.imshow(predicted_noise, cmap="gray")
plt.show()

expected_noise = torch.normal(torch.zeros(predicted_noise.shape), 1).to(device).cpu().numpy()
plt.imshow(expected_noise, cmap="gray")
plt.show()
# -


