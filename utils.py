import cv2
import os
import numpy as np
from tqdm import tqdm
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
import matplotlib.pyplot as plt

def generate_labels(images_dir, images_final_dir, labeled_images, n_regions=3, top_bottom_crop=0, mirror=True, realistic=False):
    """ Generate labels for images in a folder. The labels are based on the depth image and are saved in a dataframe.

    Args:
        images_dir (str): image folder to generate labels for
        images_final_dir (str): folder to save images to
        labeled_images (pd.Dataframe): dataframe to save labels to
        n_regions (int, optional): Number of regions to split image into. Defaults to 3.
        top_bottom_crop (int, optional): Fraction to crop from top and bottom of image. Defaults to 0.
        mirror (bool, optional): Also save mirrored versions of every image. Defaults to True.
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
        if mirror:
            img_normal_mirror = cv2.flip(img_normal, 0)

        # Save images
        cv2.imwrite(os.path.join(images_final_dir, filename), img_normal)
        if mirror:
            cv2.imwrite(os.path.join(images_final_dir, filename.strip(".jpg") + "_mirrored.jpg"), img_normal_mirror)

        # Get dimensions
        img_x = img_depth.shape[0]
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
        
        if mirror:
            action_list_mirror = action_list[::-1]
            data_row_mirror = [os.path.join(images_final_dir, filename.strip(".jpg") + "_mirrored.jpg")]
            data_row_mirror.extend(action_list_mirror)
            labeled_images.loc[len(labeled_images)] = data_row_mirror

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

def calc_mean_std_dataset(image_folder, IMAGE_TRANSFORM):
    """ Calculate mean and std of dataset

    Args:
        image_folder (str): folder with images
        IMAGE_TRANSFORM (torchvision.transforms): image transformation used

    Returns:
        (float, float): mean and std of dataset
    """
    # Set mean and std to 0
    mean = 0
    std = 0

    # Loop through all images in folder and calculate mean and std
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = convert_image_dtype(read_image(img_path), torch.float)
        img = IMAGE_TRANSFORM(img)
        mean += img.mean()
        std += img.std()

    # Calculate average mean and std
    mean /= len(os.listdir(image_folder))
    std /= len(os.listdir(image_folder))

    return mean, std