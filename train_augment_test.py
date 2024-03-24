import torch
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
import os

all_images = 'all_images'

IMAGE_TRANSFORM = transforms.Compose([
    transforms.CenterCrop((520, 120)),
    # transforms.Grayscale(),
    # transforms.Resize((26, 12), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
])

IMAGE_TRANSFORM_TRAIN = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.3, hue=0.05),
    transforms.RandomRotation(degrees=5),
    # transforms.CenterCrop((520, 120)),
    # transforms.Grayscale(),
    # transforms.Resize((26, 12), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
    # transforms.Normalize(mean=[0.3068], std=[0.1754])
])

# show 1 origina images in RGB with train_transform images next to them
import matplotlib.pyplot as plt
import random

n_images = len(os.listdir(all_images))
for i in range(10):
    image_number = random.randint(0, n_images-1)
    # combine image folder with path of image
    img_path = os.path.join(all_images, os.listdir(all_images)[image_number])
    img = convert_image_dtype(read_image(img_path), torch.float)
    img_train = IMAGE_TRANSFORM_TRAIN(img)
    fig, axs = plt.subplots(1, 2, figsize=(10, 20))
    axs[0].imshow(img.permute(1, 2, 0))
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(img_train.permute(1, 2, 0))
    axs[1].set_title('Train Transformed Image')
    axs[1].axis('off')
    plt.show()