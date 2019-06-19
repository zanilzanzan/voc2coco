import json
import os
import shutil


class JSON2COCO:
    def __init__(self, src, dest, val_part):
        if os.path.exists(src):
            self.json_path = src
        else:
            raise Exception('Given .json annotation input file doesn\'t exist.')

        self.image_src = os.path.abspath(os.path.join(src, '..', 'images'))
        # if not os.path.exists(self.image_src):
        #     raise Exception('Given images input directory doesn\'t exist.')

        self.dest = dest
        self.train_dest = os.path.join(dest, 'train', 'images')
        if not os.path.exists(self.train_dest):
            os.makedirs(self.train_dest)

        self.coco_file_train = os.path.join(dest, 'train', 'uav_instances_coco_train.json')

        self.category_dict = dict()
        self.category_item_id = 0
        self.annotation_id = 0
        self.coco = dict()
        self.coco['categories'] = []
        self.get_coco_dset('train')

        self.val_part = int(val_part/10)
        if self.val_part:
            self.coco_file_val = os.path.join(dest, 'val', 'uav_instances_coco_val.json')
            self.val_dest = os.path.join(dest, 'val', 'images')
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
        print('[INFO] Training set has been created. Annotations and images have been saved.')
        print(' - Number of images placed in training set: {}'.format(len(self.coco['train']['images'])))
        print(' - Training images have been saved under {}'.format(self.train_dest))
        print(' - Training-set annotation file has been saved as {}'.format(self.coco_file_train))
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
                                                                        v in self.category_dict.items()]))

        print('-' * 80)

    def add_category_item(self, name):
        category_item = dict()
        category_item['supercategory'] = 'none'
        self.category_item_id += 1
        category_item['id'] = self.category_item_id
        category_item['name'] = name
        self.coco['categories'].append(category_item)
        self.category_dict[name] = self.category_item_id
        return self.category_item_id

    def get_image_id(self, file_name):
        img_id_pt2 = '00' + file_name.rsplit('.', 1)[0][-10:]
        img_id_pt1 = '1'  # '1' for 'testsite_uav'
        img_id = int(img_id_pt1 + img_id_pt2)
        return img_id

    def add_image_item(self, name, size, part):
        file_name = name['reference_id']
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
        if 'conditions' in name:
            image_item['conditions'] = dict()
            image_item['conditions']['snowing'] = name['conditions']['snowing']
            image_item['conditions']['fog']	= name['conditions']['fog']
            image_item['conditions']['sky']	= name['conditions']['sky']
            image_item['conditions']['raining'] = name['conditions']['raining']
            image_item['conditions']['daytime']	= name['conditions']['daytime']
        self.coco[part]['images'].append(image_item)
        return image_id

    def add_annotation_item(self, image_id, category_id, bbox, part, type_name):
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
        annotation_item['type'] = type_name
        self.annotation_id += 1
        annotation_item['id'] = self.annotation_id
        self.coco[part]['annotations'].append(annotation_item)

    def convert_2_coco_from_json(self, json_path):
        with open(json_path) as f:
            obj = json.load(f)

        for name in obj:
            if len(name['final_output']) == 0:
                continue

                # Use the part below to generate a smaller annotation file with fewer images.
                # Don't forget to use corresponding images!
                # if int(img_id_pt2[-5:]) < 2800:
                #    continue
                # print(img_id_pt2[-5:])
            if (self.annotation_id % 10) < self.val_part and self.val_part:
                part = 'val'
            else:
                part = 'train'

            current_img_id = self.add_image_item(name, {'width':1920, 'height':1080}, part)

            for obj_crd in name['final_output']:
                print(obj_crd['label'])
                print(obj_crd['coordinates'])

                # Assign the dict that contains bounding box list and category id to label_element and create COCO bbox list
                # label_element = obj[name]['labels'][0]
                # bbox = label_element['bbox']
                # category_name = label_element['category_name'].lower()
                category_name = obj_crd['label'].lower()
                type_name = None
                if 'type' in obj_crd:
                    type_name = obj_crd['type'].lower()
                # bbox_height = bbox[2] - bbox[0]
                # bbox_width = bbox[3] - bbox[1]
                bbox_height = obj_crd['coordinates']['xmax'] - obj_crd['coordinates']['xmin']
                bbox_width = obj_crd['coordinates']['ymax'] - obj_crd['coordinates']['ymin']
                coco_bbox = [obj_crd['coordinates']['ymin'], obj_crd['coordinates']['xmin'], bbox_width, bbox_height]

                # shutil.copyfile(os.path.join(self.image_src, name), os.path.join(self.dest, part, 'images', name))
                # Add image-item dict to uav_coco_dset['images'] list
                # current_img_id = self.add_image_item(name, obj[name], part)

                # Correct category_name conflicts and check if the category name and id is already registered
                if category_name == 'fo':
                    category_name = 'ufo'
                if category_name not in self.category_dict:
                    current_category_id = self.add_category_item(category_name)
                else:
                    current_category_id = self.category_dict[category_name]

                # Add annotation-item dict to uav_coco_dset['annotations'] list
                self.add_annotation_item(current_img_id, current_category_id, coco_bbox, part, type_name)
            # break
