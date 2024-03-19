import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def generate_labels(images_dir, images_final_dir, labeled_images, mirror=True, realistic=False, print_output=False):
    """
    Generate control actions for each image in the folder

    :param images_dir: folder with images to label
    :param images_final_dir: folder to save images
    :param labeled_images: pd.DataFrame to save control actions
    :param mirror: if True, mirror images
    :param realistic: if True, make images realistic (for sim images only)
    :param print_output: if True, print control actions

    :return: pd.DataFrame with control actions
    """
    print(f"Generating labels for images in {images_dir}...")
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
        mean_depths = [img_left.mean(), img_center.mean(), img_right.mean()]
        
        # print control action
        action = mean_depths.index(min(mean_depths))
        if action == 0:
            labeled_images.loc[len(labeled_images)] = [os.path.join(images_final_dir, filename), 1, 0, 0]
            if mirror:
                labeled_images.loc[len(labeled_images)] = [os.path.join(images_final_dir, filename.strip(".jpg") + "_mirror.jpg"), 0, 0, 1]
        elif action == 1:
            labeled_images.loc[len(labeled_images)] = [os.path.join(images_final_dir, filename), 0, 1, 0]
            if mirror:
                labeled_images.loc[len(labeled_images)] = [os.path.join(images_final_dir, filename.strip(".jpg") + "_mirror.jpg"), 0, 1, 0]
        else:
            labeled_images.loc[len(labeled_images)] = [os.path.join(images_final_dir, filename), 0, 0, 1]
            if mirror:
                labeled_images.loc[len(labeled_images)] = [os.path.join(images_final_dir, filename.strip(".jpg") + "_mirror.jpg"), 1, 0, 0]

        if print_output:
            print(action)

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
    darken = np.random.uniform(0.6, 1.0)
    img = cv2.addWeighted(img, darken, img, 0, 0)

    return img

# import numpy as np
# import torch
# import torch.nn.functional as F
# from torchvision.transforms import Compose

# # import sys
# # sys.path.append("/DepthAnything")

# from DepthAnything.depth_anything.dpt import DepthAnything
# from DepthAnything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# # from depth_anything.dpt import DepthAnything
# # from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# def generate_depth_images(encoder, img_path):
#     """
#     Generate depth images for each image in the folder

#     :param encoder: encoder to use
#     :param outdir: folder to save depth images
#     :param img_path: folder with images

#     :return: None
#     """
#     encoder = "vitb"
#     img_path = "../sim_images"
#     outdir = img_path + "_depth"
    
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()
    
#     total_params = sum(param.numel() for param in depth_anything.parameters())
#     print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
#     transform = Compose([
#         Resize(
#             width=518,
#             height=518,
#             resize_target=False,
#             keep_aspect_ratio=True,
#             ensure_multiple_of=14,
#             resize_method='lower_bound',
#             image_interpolation_method=cv2.INTER_CUBIC,
#         ),
#         NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         PrepareForNet(),
#     ])

#     print("Seems to work")

#     if True:
#         return 0
    
#     filenames = os.listdir(img_path)
#     filenames = [os.path.join(img_path, filename) for filename in filenames if not filename.startswith('.')]
#     filenames.sort()
    
#     for filename in tqdm(filenames):
#         raw_image = cv2.imread(filename)
#         image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
#         h, w = image.shape[:2]
        
#         image = transform({'image': image})['image']
#         image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
#         with torch.no_grad():
#             depth = depth_anything(image)
        
#         depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
#         depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
#         depth = depth.cpu().numpy().astype(np.uint8)
        
#         depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        
#         filename = os.path.basename(filename)

#         cv2.imwrite(os.path.join(outdir, filename[:filename.rfind('.')] + '_depth.png'), depth)

# from transformers import pipeline
# from PIL import Image

# def generate_depth_images():
#     """
#     Generate depth images for each image in the folder

#     :return: None
#     """
#     image_folder = "sim_images"
#     for filename in os.listdir(image_folder):
#         print(filename)
#         f = os.path.join(image_folder, filename)
#         image = Image.open(f)
#         pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf")
#         depth = pipe(image)["depth"]
#         depth.save(os.path.join(image_folder + "_depth4576", filename.strip(".jpg") + "_depth.png"))

#     # image = Image.open('real_images/11549407.jpg')
#     # pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf")
#     # depth = pipe(image)["depth"]

#     # # show the depth map
#     # depth.show()

        