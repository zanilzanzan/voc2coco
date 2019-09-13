import json
import os
import shutil


class VOCJSON2COCO:
    def __init__(self, src, dest, val_part):
        if os.path.exists(src):
            self.json_path = src
        else:
            raise Exception('Given .json annotation input file doesn\'t exist.')

        self.image_src = os.path.abspath(os.path.join(src, '..', 'images'))
        # if not os.path.exists(self.image_src):
        #     raise Exception('Given images input directory doesn\'t exist.')

        self.dest = dest
        self.train_dest = os.path.join(dest, 'coco_annotations')
        if not os.path.exists(self.train_dest):
            os.makedirs(self.train_dest)

        self.coco_file_train = os.path.join(self.train_dest, 'coco_file.json')

        self.category_dict = dict()
        self.category_item_id = 19
        self.annotation_id = 7081902000000
        self.coco = dict()
        self.categories_dict = {1: 'uav', 2: 'airplane', 3: 'bicycle', 4: 'bird', 5: 'boat',
                                6: 'bus', 7: 'car', 8: 'cat', 9: 'cow', 10: 'dog', 11: 'horse',
                                12: 'motorcycle', 13: 'person', 14: 'traffic_light', 15: 'train',
                                16: 'truck', 17: 'ufo', 18: 'helicopter'}
        self.cat_names_dict = {'uav': 1, 'airplane': 2, 'bicycle': 3, 'bird': 4, 'boat': 5,
                                'bus': 6, 'car': 7, 'cat': 8, 'cow': 9, 'dog': 10, 'horse': 11,
                                'motorcycle': 12, 'person': 13, 'traffic_light': 14, 'train': 15,
                                'truck': 16, 'ufo': 17, 'helicopter': 18}
        self.coco['categories'] = []
        self.get_coco_dset('train')

        self.val_part = int(val_part/10)
        if self.val_part:
            self.coco_file_val = os.path.join(dest, 'val', 'uav_instances_coco_val.json')
            self.val_dest = os.path.join(dest, 'coco_annotations_val')
            if not os.path.exists(self.val_dest):
                os.makedirs(self.val_dest)
            self.get_coco_dset('val')

    def get_coco_dset(self, part):
        self.coco[part] = dict()
        self.coco[part]['images'] = []
        self.coco[part]['type'] = 'instances'
        self.coco[part]['annotations'] = []
        self.coco[part]['categories'] = []

    def convert_2_coco(self):
        self.convert_2_coco_from_json(self.json_path)
        print('[PROGRESS] File conversion is successful.')

        self.coco['train']['categories'] = self.coco['categories']
        json.dump(self.coco['train'], open(self.coco_file_train, 'w'))

        print('-'*80)
        print('[INFO] COCO annotations have been created.')
        print(' - Number of images in the  dataset: {}'.format(len(self.coco['train']['images'])))
        print(' - COCO annotation file has been saved as {}'.format(self.coco_file_train))
        print('-'*80)

        if self.val_part:
            self.coco['val']['categories'] = self.coco['categories']
            json.dump(self.coco['val'], open(self.coco_file_val, 'w'))
            print('[INFO] Validation set has been created. Annotations and images have been saved.')
            print(' - Number of images placed in validation set: {}'.format(len(self.coco['val']['images'])))
            print(' - Validation images have been saved under {}'.format(self.val_dest))
            print(' - Validation-set annotation file has been saved as {}'.format(self.coco_file_val))
            print('-' * 80)

        print('[INFO] Categories have been created.')
        print(' - Number of categories created: {}'.format(len(self.coco['train']['categories'])))
        print(' - Names and IDs of created categories: ' + ''.join(['{0}:{1} '.format(k, v) for k,
                                                                        v in self.cat_names_dict.items()]))

        print('-' * 80)

    def create_cat_dict(self):
        for key, value in self.cat_names_dict.items():
            category_item = dict()
            category_item['supercategory'] = 'none'
            category_item['name'] = key
            category_item['id'] = value
            self.coco['categories'].append(category_item)
        # print(self.coco['categories'])

    def add_category_item(self, name):
        category_item = dict()
        category_item['supercategory'] = 'none'
        category_item['id'] = self.cat_names_dict[name]
        print(category_item['id'])
        category_item['name'] = name
        self.coco['categories'].append(category_item)
        self.cat_names_dict[name] = self.cat_names_dict[name]
        self.category_item_id += 1
        return category_item['id']

    def get_image_id(self, file_name):
        img_id_pt2 = '0708' + file_name.rsplit('.', 1)[0][-10:]
        img_id_pt1 = '1902'  # '1' for 'testsite_uav'
        img_id = int(img_id_pt1 + img_id_pt2)
        return img_id

    def add_image_item(self, file_name, size, part):
        if file_name is None:
            raise Exception('Could not find filename info in .json file.')
        if size['width'] is None:
            raise Exception('Could not find width info in .json file.')
        if size['height'] is None:
            raise Exception('Could not find height info in .json file.')
        image_id = self.get_image_id(file_name)
        image_item = dict()
        image_item['id'] = image_id
        image_item['file_name'] = file_name
        image_item['width'] = size['width']
        image_item['height'] = size['height']
        self.coco[part]['images'].append(image_item)
        return image_id

    def add_annotation_item(self, image_id, category_id, bbox, part):
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
        self.annotation_id += 1
        annotation_item['id'] = self.annotation_id
        self.coco[part]['annotations'].append(annotation_item)

    def convert_2_coco_from_json(self, json_path):
        with open(json_path) as f:
            obj = json.load(f)

        self.create_cat_dict()

        for name in obj:
            if len(obj[name]['labels']) == 0:
                continue

            img_path_list = obj[name]['image_path'].split('/')
            img_name = os.path.join(img_path_list[-2], img_path_list[-1])

            # Use the part below to generate a smaller annotation file with fewer images.
            # Don't forget to use corresponding images!
            # if int(img_id_pt2[-5:]) < 2800:
            #    continue
            # print(img_id_pt2[-5:])

            # Assign the dict that contains bounding box list and category id to label_element and create COCO bbox list
            label_element = obj[name]['labels'][0]
            bbox = label_element['bbox']
            category_name = label_element['category_name'].lower()
            # print(category_name)
            bbox_height = bbox[2] - bbox[0]
            bbox_width = bbox[3] - bbox[1]
            coco_bbox = [bbox[1], bbox[0], bbox_width, bbox_height]

            if (self.annotation_id % 10) < self.val_part and self.val_part:
                part = 'val'
            else:
                part = 'train'

            # shutil.copyfile(os.path.join(self.image_src, name), os.path.join(self.dest, part, 'images',name))

            # Add image-item dict to uav_coco_dset['images'] list
            current_img_id = self.add_image_item(img_name, obj[name], part)

            # Correct category_name conflicts and check if the category name and id is already registered
            if category_name == 'duck':
                category_name = 'bird'
            elif category_name == 'auv' or category_name == 'drone':
                category_name = 'uav'
            elif category_name == 'fo':
                category_name = 'ufo'
            # print(category_name)

            if category_name not in self.cat_names_dict:
                current_category_id = self.add_category_item(category_name)
            else:
                current_category_id = self.cat_names_dict[category_name]
            # print(current_category_id)

            # Add annotation-item dict to uav_coco_dset['annotations'] list
            self.add_annotation_item(current_img_id, current_category_id, coco_bbox, part)
