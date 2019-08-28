import os
import sys
import cv2
import argparse
import numpy as np


def read_image(file_src, file_type, file_name, mode=1):
    return cv2.imread(os.path.join(file_src, file_type, file_name), mode)


def visualize_segmentation_labels(src, dest, alpha_val_image=0.7, alpha_val_label=0.4):
    dest = os.path.join(dest, 'label_visualizations')

    if not os.path.exists(dest):
        os.makedirs(dest)

    file_names = [f for f in os.listdir(os.path.join(src, 'images')) if f.endswith('.png')]
    file_names = sorted(file_names, key=lambda x: int(os.path.splitext(x)[0]))
    total_images = len(file_names)

    for i, file in enumerate(file_names):
        image = read_image(src, 'images', file)
        label = read_image(src, 'labels', file, 0)

        label = label * 255
        label = np.dstack((label, label, label))
        label[:, :, :2] = 0

        image_masked = cv2.addWeighted(image, alpha_val_image, label, alpha_val_label, 0)
        cv2.imwrite(os.path.join(dest, file), image_masked)
        # cv2.imwrite(os.path.join(dest, file), label)

        if (i+1) % 5 == 0:
            progress_txt = 'Processing and creating visualizations: [%06d/%06d]' % (i+1, total_images)
            sys.stdout.write("\r" + progress_txt)
            sys.stdout.flush()

    print('Created all visualizations in given destination: [%06d/%06d]' % (i+1, total_images))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", "--SourceDirectoryPath", default='./',
                    help="Source directory path for images and annotations. The images should be place under ./images,"
                         "and the semantic segmentation annotations should be placed under ./labels.")
    ap.add_argument("-dest", "--DestinationDirectoryPath", default='./',
                    help="Destination directory path for visualized annotations")
    args = vars(ap.parse_args())

    src = args["SourceDirectoryPath"]
    dest = args["DestinationDirectoryPath"]

    visualize_segmentation_labels(src, dest)
