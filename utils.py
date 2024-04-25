import cv2
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype, to_tensor, hflip
import matplotlib.pyplot as plt

def generate_labels(images_dir, images_final_dir, labeled_images, n_regions=3, top_bottom_crop=0, realistic=False):
    """ Generate labels for images in a folder. The labels are based on the depth image and are saved in a dataframe.

    Args:
        images_dir (str): image folder to generate labels for
        images_final_dir (str): folder to save images to
        labeled_images (pd.Dataframe): dataframe to save labels to
        n_regions (int, optional): Number of regions to split image into. Defaults to 3.
        top_bottom_crop (int, optional): Fraction to crop from top and bottom of image. Defaults to 0.
        realistic (bool, optional): Make every image more realistic (only use for simulator images). Defaults to False.

    Returns:
        pd.Dataframe: dataframe with labels for images
    """
    # Loop through all images in folder and calculate labels
    print(f"Generating labels for images in {images_dir}...")
    for filename in tqdm(os.listdir(images_dir)):
        f = os.path.join(images_dir, filename)
        f_depth = os.path.join(images_dir + "_depth/", filename.strip(".jpg") + "_depth.png")

        # Read normal and depth image
        img_normal = cv2.imread(f)
        img_depth = cv2.imread(f_depth)

        # Check if files are valid
        if img_normal is None or img_depth is None:
            assert False, f"Error reading {f} or {f_depth}"

        # Make realistic and or mirror image
        if realistic:
            img_normal = make_realistic(img_normal)

        # Save images
        cv2.imwrite(os.path.join(images_final_dir, filename), img_normal)

        # Get y dimension
        img_y = img_depth.shape[1]

        # Split image into n regions
        img_regions = np.array_split(img_depth, n_regions, axis=0)
        img_regions = [img_region[:,int(img_y*top_bottom_crop):int(img_y*(1-top_bottom_crop))] for img_region in img_regions]
        mean_depths = [img_region.mean() for img_region in img_regions]

        # Get control action and one-hot encode
        action = min(mean_depths)
        action_list = [1 if mean_depth == action else 0 for mean_depth in mean_depths]
        data_row = [os.path.join(images_final_dir, filename)]
        data_row.extend(action_list)
        labeled_images.loc[len(labeled_images)] = data_row
        
    return labeled_images

def make_realistic(img):
    """ Make image realistic by blurring and darkening

    Args:
        img (np.array): image to make realistic

    Returns:
        np.array: realistic image
    """
    # Blur image with random kernel
    kernel = np.random.choice([1, 3, 5])
    img = cv2.GaussianBlur(img, (kernel, kernel), 0)

    # Darken image with random factor
    darken = np.random.uniform(0.8, 1.2)
    img = cv2.addWeighted(img, darken, img, 0, 0)

    return img

def plot_images(real_images_folder, sim_images_folder):
    """ Plot real, simulator and realistic simulator image side by side

    Args:
        real_images_folder (str): folder with real images
        sim_images_folder (str): folder with simulator images
    """
    # Load the images
    real_image_example = plt.imread(os.path.join(real_images_folder, os.listdir(real_images_folder)[0]))
    sim_image_example = plt.imread(os.path.join(sim_images_folder, os.listdir(sim_images_folder)[0]))

    # Augment the simulator image to make it more realistic
    realistic_sim_image_example = make_realistic(sim_image_example)

    # Show the images
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    ax[0].imshow(real_image_example)
    ax[0].set_title('Real Image')
    ax[1].imshow(sim_image_example)
    ax[1].set_title('Sim Image')
    ax[2].imshow(realistic_sim_image_example)
    ax[2].set_title('Realistic Sim Image')
    plt.show()

def calc_mean_std_dataset(train_dataset, IMAGE_TRANSFORM, own_dataset=False):
    """ Calculate the mean and standard deviation of the dataset.

    Args:
        train_dataset (-): dataset to calculate mean and std for
        IMAGE_TRANSFORM (transforms.Compose): transformation to apply to all images
        own_dataset (bool, optional): whether the dataset is a custom dataset or not. Defaults to False.

    Returns:
        float, float: mean and standard deviation of the dataset
    """
    # Set mean and std to 0
    mean = 0
    std = 0

    # When using own dataset, calculate mean and std by looping through image folder
    # When using dataset from huggingface, calculate mean and std by looping through dataset
    if own_dataset:
        filenames = pd.read_csv(train_dataset, skiprows=1, header=None).iloc[:, 0]
        n_images = len(filenames)
        for filename in filenames:
            f = filename.replace("\\", os.sep)
            image = convert_image_dtype(read_image(f), torch.float)
            image = IMAGE_TRANSFORM(image)
            mean += torch.mean(image)
            std += torch.std(image)
    else:
        n_images = len(train_dataset)
        for i in range(len(train_dataset)):
            image = to_tensor(train_dataset[i]["image"])
            image = IMAGE_TRANSFORM(image)
            mean += torch.mean(image)
            std += torch.std(image)

    # Calculate average mean and std
    mean /= n_images
    std /= n_images

    return mean, std

def generate_dataloaders(val_ratio, test_ratio, standard_transform, train_transform, train_dataset, test_dataset, batch_size, own_dataset=False):
    """ Create DataLoaders for the training, validation, and test sets. If own_dataset is True, this dataset will be split into a training,
    validation, and test set following the provided ratios. If own_dataset is False, the train_dataset will be split into a training and
    validation set, while the test_dataset will be used as the test set.

    Args:
        val_ratio (float): ratio of the dataset to use for the validation set
        test_ratio (float): ratio of the dataset to use for the test set
        standard_transform (transforms.Compose): transformation to apply to all images
        train_transform (transforms.Compose): transformation to optionally apply to training images
        train_dataset (-): dataset to use for training
        test_dataset (-): dataset to use for testing
        batch_size (float): batch size to use for the DataLoaders
        own_dataset (bool, optional): whether the dataset is a custom dataset or not. Defaults to False.

    Returns:
        DataLoader, DataLoader, DataLoader: DataLoaders for the training, validation, and test sets
    """

    # Create the datasets
    train_dataset_full = DroneImagesDataset(train_dataset, own_dataset, transform=train_transform)
    val_dataset_full = DroneImagesDataset(train_dataset, own_dataset, transform=standard_transform)
    test_dataset_full = DroneImagesDataset(test_dataset, own_dataset, transform=standard_transform)

    # Get indices for splitting the datasets
    indices = torch.randperm(len(train_dataset_full)).tolist()

    if own_dataset:
        # Split the datasets
        train_dataset = torch.utils.data.Subset(train_dataset_full, indices[:-int(len(indices)*(val_ratio+test_ratio))])
        val_dataset = torch.utils.data.Subset(val_dataset_full, indices[-int(len(indices)*(val_ratio+test_ratio)):-int(len(indices)*test_ratio)])
        test_dataset = torch.utils.data.Subset(test_dataset_full, indices[-int(len(indices)*test_ratio):])
    else:
        # Create the datasets
        train_dataset = torch.utils.data.Subset(train_dataset_full, indices[:-int(len(indices)*(val_ratio))])
        val_dataset = torch.utils.data.Subset(val_dataset_full, indices[-int(len(indices)*(val_ratio)):])
        test_dataset = test_dataset_full

    # Create DataLoaders for the training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=2, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=2, persistent_workers=True)

    return train_loader, val_loader, test_loader

# Create dataset class
class DroneImagesDataset(Dataset):
    """ Dataset class for drone images

    Args:
        Dataset (.csv or Hugging Face dataset): dataset with images and labels
    """
    def __init__(self, dataset, own_dataset=False, transform=None, mirror=True):
        self.own_dataset = own_dataset
        if self.own_dataset:
            self.annotations = pd.read_csv(dataset, skiprows=1, header=None)
        else:
            self.annotations = dataset
        self.transform = transform
        self.mirror = mirror

    def __getitem__(self, index):
        m_index = index // 2 if self.mirror else index
        if self.own_dataset:
            img_path = self.annotations.iloc[m_index, 0]
            img_path = img_path.replace("\\", os.sep)
            image = convert_image_dtype(read_image(img_path), torch.float)
            y_label = torch.tensor(list(self.annotations.iloc[m_index, 1:]), dtype=torch.float32)
        else:
            image = to_tensor(self.annotations[m_index]["image"])
            # TODO: update to make parametric with differing label length
            y_label = torch.tensor([self.annotations[m_index]["left"], self.annotations[m_index]["forward"], self.annotations[m_index]["right"]], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)

        if self.mirror and index % 2 == 0:
            image = hflip(image)
            y_label = torch.flip(y_label, [0])

        return (image, y_label)
    
    def __len__(self):
        if self.mirror:
            return len(self.annotations) * 2
        return len(self.annotations)