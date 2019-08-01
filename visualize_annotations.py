import os
import argparse
import cv2
import numpy as np


def read_image(file_src, file_type, file_name, mode=1):
    return cv2.imread(os.path.join(file_src, file_type, file_name), mode)


def visualize_segmentation_labels(src, dest, alpha_val=0.4):
    dest = os.path.join(dest, 'label_visualizations')

    if not os.path.exists(dest):
        os.makedirs(dest)

    for dir_path, dir_names, file_names in os.walk(src):
        continue

    for file in file_names:
        image = read_image(src, 'images', file)
        label = read_image(src, 'labels', file, 0)

        label = label * 255
        label = np.dstack((label, label, label))
        label[:, :, :2] = 0

        image_masked = cv2.addWeighted(image, 1.0, label, alpha_val, 0)
        cv2.imwrite(os.path.join(dest, file), image_masked)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", "--SourceDirectoryPath", required=True,
                    help="Source directory path for images and annotations. The images should be place under ./images,"
                         "and the semantic segmentation annotations should be placed under ./labels.")
    ap.add_argument("-dest", "--DestinationDirectoryPath", required=True,
                    help="Destination directory path for visualized annotations")
    args = vars(ap.parse_args())

    src = args["SourceDirectoryPath"]
    dest = args["DestinationDirectoryPath"]

    visualize_segmentation_labels(src, dest)

