# The module jupytext is used to treat this .py file as a jupyter notebook file. To keep the output after every session, go to "File" -> "Jupytext" -> "Pair Notebook with ipynb document". This generates a file PY_FILENAME.ipynb.

# # Imports

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch

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

# Plots the first 9 images.
amount_images_to_show = 9
grid_shape = np.array([3, 3])
for image_index in range(amount_images_to_show):
    plt.subplot(grid_shape[0], grid_shape[1], image_index + 1)
    plt.xticks([])
    plt.yticks([])
    image_to_show = train_image_data[image_index]
    plt.imshow(image_to_show, cmap="gray")
plt.show()

assert (train_labels[:9] == np.array([5, 0, 4, 1, 9, 2, 1, 3, 1])).all()

# # Training

batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

epochs = 10
for epoch_index in range(epochs):
    for batch_index, train_data_batch in enumerate(train_loader):
        ...

# # Evaluation



# # Forward Noise Process

def add_noise(image, beta):
    alpha = 1 - beta
    alpha_cum = torch.prod(alpha)
    random_array = torch.randn(image.shape)
    noisy_image = random_array * (1 - alpha_cum) + torch.sqrt(alpha_cum) * image
    return noisy_image
