import tensorflow as tf
import io
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import os

import json
import os
import shutil
import datetime

class JSON2COCO:
    def __init__(self, src, dest, part):
        if os.path.exists(src):
            self.json_path = src
        else:
            raise Exception('Given .json annotation input file doesn\'t exist.')

        self.image_src = os.path.abspath(os.path.join(src, '..', 'images'))

        self.src = src
        self.dest = dest
        # self.dest = os.path.join(dest, part)
        if not os.path.exists(self.dest):
            os.makedirs(self.dest)

        self.category_dict = dict()
        self.image_id = 0
        self.annotation_id = 0
        self.coco = dict()
        # self.coco['categories'] = []
        self.get_coco_dset()

    def get_coco_dset(self):
        self.coco = dict()
        self.coco['images'] = []
        self.coco['type'] = 'instances'
        self.coco['annotations'] = []
        self.coco['categories'] = []

    def convert_2_coco(self):
        self.recordReader(self.src)
        print('[PROGRESS] File conversion is successful.')

	# The print parameters below are not set correctly. Revise them.
        print('-'*80)
        print('[INFO] Dataset has been created. Annotations and images have been saved.')
        print(' - Number of images placed in created dataset: {}'.format(len(self.coco['images'])))
        print(' - Images have been saved under {}'.format(self.dest))
        print(' - The annotation file has been saved as {}'.format(self.coco_file))
        print('-'*80)
        print('[INFO] Categories have been created.')
        print(' - Number of categories created: {}'.format(len(self.coco['categories'])))
        print(' - Names and IDs of created categories: ' + ''.join(['{0}:{1} '.format(k, v) for k, v in self.category_dict.items()]))
        print('-' * 80)

    def add_category_item(self, name, id):
        category_item = dict()
        category_item['supercategory'] = 'none'
        # self.category_item_id += 1
        category_item['id'] = id
        category_item['name'] = name
        self.coco['categories'].append(category_item)
        self.category_dict[name] = id
        # return id

    def get_id(self, id):
        now = datetime.datetime.now()
        return int(now.strftime('%Y%m%d%H%M%S') + str(id))

    def add_image_item(self, name, size):
        file_name = os.path.join(self.image_name_root, name)
        print(file_name)
        if file_name is None:
            raise Exception('Could not find filename info in .json file.')
        if size['width'] is None:
            raise Exception('Could not find width info in .json file.')
        if size['height'] is None:
            raise Exception('Could not find height info in .json file.')
        image_item = dict()
        self.image_id += 1
        image_item['id'] = self.get_id(self.image_id) #int('1%07d' % self.image_id)
        print(image_item['id'])
        image_item['file_name'] = file_name
        image_item['width'] = size['width']
        image_item['height'] = size['height']
        self.coco['images'].append(image_item)
        return image_item['id']

    def add_annotation_item(self, image_id, category_id, bbox):
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x,y,w,h
        # left_top
        seg.append(bbox[0])
        seg.append(bbox[1])
        # left_bottom
        seg.append(bbox[0])
        seg.append(bbox[1] + bbox[3])
        # right_bottom
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1] + bbox[3])
        # right_top
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1])

        annotation_item['segmentation'].append(seg)

        annotation_item['area'] = bbox[2] * bbox[3]
        annotation_item['iscrowd'] = 0
        annotation_item['ignore'] = 0
        annotation_item['image_id'] = image_id
        annotation_item['bbox'] = bbox
        annotation_item['category_id'] = category_id
        # annotation_item['type'] = type_name
        self.annotation_id += 1
        annotation_item['id'] = self.get_id(self.annotation_id)
        print('annotation_id: ', annotation_item['id'])
        self.coco['annotations'].append(annotation_item)

    def recordReader(self, recordDirPath):
        for tfr in os.listdir(recordDirPath):
            if tfr.endswith(".record"):
                # counter = 0
                self.image_name_root = os.path.join(tfr.replace('.record',''), 'images')
                self.image_dest = os.path.join(self.dest, self.image_name_root)
                annot_dest = os.path.join(self.dest, tfr.replace('.record',''))

                if not os.path.exists(self.image_dest):
                    os.makedirs(self.image_dest)

                self.coco_file = os.path.join(annot_dest, '{}.json'.format(tfr.replace('.record','')))

                record_iterator = tf.python_io.tf_record_iterator(path=os.path.join(recordDirPath,tfr))

                for string_record in record_iterator:
                    example = tf.train.Example()
                    example.ParseFromString(string_record)

                    height = int(example.features.feature['image/height']
                                 .int64_list
                                 .value[0])

                    width = int(example.features.feature['image/width']
                                .int64_list
                                .value[0])

                    name = str(example.features.feature['image/filename'].bytes_list.value[0])
                    name = name[2:-1]

                    # decode image
                    image_encoded = example.features.feature['image/encoded'].bytes_list.value[0]
                    encoded_jpg_io = io.BytesIO(image_encoded)
                    image = PIL.Image.open(encoded_jpg_io)
                    pix = np.array(image.getdata())
                    img = np.reshape(pix, (height, width, 3))[:,:,[2,1,0]]

                    # read attributes
                    try:
                        # get the class labels and bounding box values in current image
                        current_img_id = self.add_image_item(name, {'width':width, 'height':height})
			# read the bounding box values for each object
                        for i, cls_name in enumerate(example.features.feature['image/object/class/text'].bytes_list.value):
                            xmin = int(example.features.feature['image/object/bbox/xmin'].float_list.value[i] * width)
                            ymin = int(example.features.feature['image/object/bbox/ymin'].float_list.value[i] * height)
                            xmax = int(example.features.feature['image/object/bbox/xmax'].float_list.value[i] * width)
                            ymax = int(example.features.feature['image/object/bbox/ymax'].float_list.value[i] * height)

                            bbox_height = ymax - ymin
                            bbox_width = xmax - xmin
                            coco_bbox = [xmin, ymin, bbox_width, bbox_height]

                            # pt1 = (int(example.features.feature['image/object/bbox/xmin'].float_list.value[i] * width),
                            #       int(example.features.feature['image/object/bbox/ymin'].float_list.value[i] * height))
                            # pt2 = (int(example.features.feature['image/object/bbox/xmax'].float_list.value[i] * width),
                            #       int(example.features.feature['image/object/bbox/ymax'].float_list.value[i] * height))

                            category_name = str(cls_name)[2:-1].lower()
                            category_id = example.features.feature['image/object/class/label'].int64_list.value[i]
			    # there is an incorrectly added category called 'fo'; correct them
                            if category_name == 'fo':
                                category_name = 'ufo'

                            print('category_name: ', category_name)
                            print('category_id: ', category_id)
                            print('categort_dict: ', self.category_dict)
                            print('coco_bbox: {} {} {} {}' .format(coco_bbox[0], coco_bbox[1], coco_bbox[2], coco_bbox[3]))
                            print('bbox:      {} {} {} {}' .format(xmin, ymin, xmax, ymax))

                            if category_name not in self.category_dict:
                                self.add_category_item(category_name, category_id)

                            # Add annotation-item dict to uav_coco_dset['annotations'] list
                            self.add_annotation_item(current_img_id, category_id, coco_bbox)
                            # img = cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)

                        # img = cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
                        print(os.path.join(self.image_dest, name))
                        cv2.imwrite(os.path.join(self.image_dest, name), img)
                        json.dump(self.coco, open(self.coco_file, 'w'))
                        # counter += 1
                        # if counter > 10:
                        #     break
                    except:
                        print('No label found for this image')
                        pass

                json.dump(self.coco, open(self.coco_file, 'w'))
                self.get_coco_dset()
                self.image_id = 0
                self.annotation_id = 0
                self.category_dict = dict()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-rd", "--RecordDirectoryPath", required=True,
                    help="Directory Path for record Files")
    ap.add_argument("-des", "--DestinationDirectoryPath", required=True,
                    help="Destination Directory Path for Generated Files")
    ap.add_argument("-pt", "--TypeOfDataset", required=True,
                    help="train or validation")
    args = vars(ap.parse_args())
    # Add FOR loop for all the tfrecords files in the folder
    example = JSON2COCO(args["RecordDirectoryPath"], args["DestinationDirectoryPath"], args["TypeOfDataset"])
    example.convert_2_coco()





