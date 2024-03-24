import torch
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
import os

# to try: bilinear, bicubic or nearest exact
IMAGE_TRANSFORM = transforms.Compose([
    transforms.CenterCrop((520, 120)),
    transforms.Grayscale(),
    transforms.Resize((26, 12), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
])

all_images = 'all_images'

# calculate mean and std of entire dataset
mean = 0
std = 0

for filename in os.listdir(all_images):
    img_path = os.path.join(all_images, filename)
    img = convert_image_dtype(read_image(img_path), torch.float)
    img = IMAGE_TRANSFORM(img)
    mean += img.mean()
    std += img.std()

mean /= len(os.listdir(all_images))
std /= len(os.listdir(all_images))

print(mean, std)