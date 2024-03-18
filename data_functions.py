import cv2
import os
import pandas as pd
from tqdm import tqdm

def generate_labels(images_dir, images_final_dir, images_with_label, mirror=True, realistic=False, print_output=False):
    """
    Generate control actions for each image in the folder

    :param images_dir: folder with images
    :param images_final_dir: folder to save images
    :param images_with_label: pd.DataFrame to save control actions
    :param mirror: if True, mirror images
    :param realistic: if True, make images realistic (for sim images only)
    :param print_output: if True, print control actions

    :return: pd.DataFrame with control actions
    """
    for filename in tqdm(os.listdir(images_dir)):
        f = os.path.join(images_dir, filename)
        f_depth = os.path.join(images_dir + "_depth/", filename.strip(".jpg") + "_depth.png")

        # read depth image
        img_depth = cv2.imread(f_depth)

        # read normal image and mirror
        img_normal = cv2.imread(f)
        if realistic:
            img_normal = make_realistic(img_normal)
        if mirror:
            img_normal_mirror = cv2.flip(img_normal, 0)

        # save images
        cv2.imwrite(os.path.join(images_final_dir, filename), img_normal)
        if mirror:
            cv2.imwrite(os.path.join(images_final_dir, filename.strip(".jpg") + "_mirror.jpg"), img_normal_mirror)

        # get dimensions
        img_x = img_depth.shape[0]
        img_y = img_depth.shape[1]

        # split image into 3 regions
        img_left = img_depth[0:img_x//3, img_y//3:img_y]
        img_center = img_depth[img_x//3:2*img_x//3, img_y//3:img_y]
        img_right = img_depth[2*img_x//3:img_x, img_y//3:img_y]

        # calculate average depth for each region
        dirs = [img_left.mean(), img_center.mean(), img_right.mean()]
        
        # print control action
        action = dirs.index(min(dirs)) # CHECK THISSSSS !!!!!!! A?SDASI!!!!
        if action == 0:
            images_with_label.loc[len(images_with_label)] = [os.path.join(images_final_dir, filename), 0, 0, 1]
            images_with_label.loc[len(images_with_label)] = [os.path.join(images_final_dir, filename.strip(".jpg") + "_mirror.jpg"), 1, 0, 0]
        elif action == 1:
            images_with_label.loc[len(images_with_label)] = [os.path.join(images_final_dir, filename), 0, 1, 0]
            images_with_label.loc[len(images_with_label)] = [os.path.join(images_final_dir, filename.strip(".jpg") + "_mirror.jpg"), 0, 1, 0]
        else:
            images_with_label.loc[len(images_with_label)] = [os.path.join(images_final_dir, filename), 1, 0, 0]
            images_with_label.loc[len(images_with_label)] = [os.path.join(images_final_dir, filename.strip(".jpg") + "_mirror.jpg"), 0, 0, 1]

        if print_output:
            print(action)

    return images_with_label

def make_realistic(img):
    """
    Make image realistic

    :param img: image

    :return: realistic image
    """
    # blur image with a 5x5 kernel
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # darken image
    img = cv2.addWeighted(img, 0.6, img, 0, 0)

    return img
