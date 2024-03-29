import cv2
import os
import numpy as np
from tqdm import tqdm
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype

def generate_labels(images_dir, images_final_dir, labeled_images, n_regions=3, top_bottom_crop=0, mirror=True, realistic=False):
    """
    Generate control actions for each image in the folder

    :param images_dir: folder with images to label
    :param images_final_dir: folder to save images
    :param labeled_images: pd.DataFrame to save control actions
    :param n_regions: number of regions to split the image in
    :param top_bottom_crop: percentage of top and bottom of image to crop
    :param mirror: if True, mirror images
    :param realistic: if True, make images realistic (for sim images only)

    :return: pd.DataFrame with control actions
    """
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
    """
    Make image realistic

    :param img: image

    :return: realistic image
    """
    # blur image with random kernel
    kernel = np.random.choice([1, 3, 5])
    img = cv2.GaussianBlur(img, (kernel, kernel), 0)

    # darken image with random factor
    darken = np.random.uniform(0.8, 1.0)
    img = cv2.addWeighted(img, darken, img, 0, 0)

    return img

def calc_mean_std_dataset(image_folder, IMAGE_TRANSFORM):
    """
    Calculate mean and std of dataset

    :param image_folder: folder with images
    :param IMAGE_TRANSFORM: image transformation used

    :return: mean and std of dataset
    """
    mean = 0
    std = 0

    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = convert_image_dtype(read_image(img_path), torch.float)
        img = IMAGE_TRANSFORM(img)
        mean += img.mean()
        std += img.std()

    mean /= len(os.listdir(image_folder))
    std /= len(os.listdir(image_folder))

    return mean, std