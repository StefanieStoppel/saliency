import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure


def create_saliency_map(input_img, fixation_img, output_img):
    mask = cv2.imread(fixation_img)
    img = cv2.imread(input_img)
    im = mask + img  # random image
    masked = np.ma.masked_where(mask != 0, im)


    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(img, interpolation='none')
    plt.imshow(masked, 'jet', interpolation='none', alpha=0.5)
    plt.show()


def create_heatmap_overlay(color_path, fixation_path, display=False):
    # original source: https://stackoverflow.com/questions/46020894/how-to-superimpose-heatmap-on-a-base-image
    fixation = cv2.imread(fixation_path)
    color = cv2.imread(color_path)

    # get color map
    map_img = exposure.rescale_intensity(fixation, out_range=(0, 255))
    map_img = np.uint8(map_img)
    heatmap_img = cv2.applyColorMap(map_img, cv2.COLORMAP_HOT)

    # merge map and color image
    target = cv2.addWeighted(heatmap_img, 0.8, color, 1.0, 0)

    # show result
    if display:
        cv2.imshow('color_heatmap', target)
        cv2.waitKey()
    return target


def _get_image_path_list(data_path, file):
    file_path = os.path.join(data_path, file)
    with open(file_path) as f:
        content = f.readlines()
    content_sanitized = [os.path.abspath(f"{data_path}/{line.strip()}") for line in content]
    return content_sanitized


def _create_heatmap_path_list(image_path_list):
    return [line.replace("images", "maps").replace("image", "map") for line in image_path_list]


def _create_path_lists(data_path, image_txt_file, fixation_txt_file):
    train_img_paths = _get_image_path_list(data_path, file=image_txt_file)
    train_fixation_paths = _get_image_path_list(data_path, file=fixation_txt_file)
    train_map_pathlist = _create_heatmap_path_list(train_img_paths)
    return train_img_paths, train_fixation_paths, train_map_pathlist


def _create_heatmap_overlays(img_path_list, fixation_path_list, target_map_path_list):
    print("Creating heatmap overlays from color images and fixations.")
    for img_path, fix_path, target_map_path in zip(img_path_list, fixation_path_list, target_map_path_list):
        heatmap_overlay = create_heatmap_overlay(img_path, fix_path)
        cv2.imwrite(target_map_path, heatmap_overlay)


def create_heatmap_overlays_from_data(data_path="/home/steffi/dev/CV2/data"):
    train_img_paths, train_fixation_paths, train_map_pathlist = _create_path_lists(data_path,
                                                                                   image_txt_file="train_images.txt",
                                                                                   fixation_txt_file="train_fixations"
                                                                                                     ".txt")
    val_img_paths, val_fixation_paths, val_map_pathlist = _create_path_lists(data_path,
                                                                             image_txt_file="val_images.txt",
                                                                             fixation_txt_file="val_fixations.txt")

    _create_heatmap_overlays(val_img_paths, val_fixation_paths, val_map_pathlist)
    _create_heatmap_overlays(train_img_paths, train_fixation_paths, train_map_pathlist)


if __name__ == "__main__":
    create_heatmap_overlays_from_data()
