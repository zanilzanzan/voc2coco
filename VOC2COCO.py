import os
import sys
import json
import fnmatch
import xml.etree.ElementTree as ET


class V2CC:
    def __init__(self, src, dest):
        if os.path.exists(src):
            self.xml_path = src
        else:
            raise Exception('Given input directory doesn\'t exist.')
        if not os.path.exists(dest):
            os.makedirs(os.path.join(dest))
        # tail = os.path.basename(os.path.normpath(src))
        tail = self.splitall(self.xml_path)[-2]
        self.json_file = os.path.join(dest, tail + '.json')
        # print(str(os.path.basename(dest)))

        self.coco = dict()
        self.coco['images'] = []
        self.coco['type'] = 'instances'
        self.coco['annotations'] = []
        self.coco['categories'] = [{"supercategory": "none", "id": 17, "name": "ufo"},
                                   {"supercategory": "none", "id": 1, "name": "uav"}]

        self.image_set = set()
        self.category_dict = {'uav': 1, 'ufo': 17}

        self.category_item_id = 3
        self.image_id = 20190000000
        self.annotation_id = 0

    def splitall(self, path):
        allparts = []
        while 1:
            parts = os.path.split(path)
            if parts[0] == path:  # sentinel for absolute paths
                allparts.insert(0, parts[0])
                break
            elif parts[1] == path: # sentinel for relative paths
                allparts.insert(0, parts[1])
                break
            else:
                path = parts[0]
                allparts.insert(0, parts[1])
        return allparts

    def convert_2_coco(self):
        self.parse_xml_files(self.xml_path)
        json.dump(self.coco, open(self.json_file, 'w'))
        return self.json_file

    def add_category_item(self, name):
        category_item = dict()
        category_item['supercategory'] = 'none'
        self.category_item_id += 1
        category_item['id'] = self.category_item_id
        category_item['name'] = name
        self.coco['categories'].append(category_item)
        self.category_dict[name] = self.category_item_id
        return self.category_item_id

    def add_image_item(self, file_name, size):
        if file_name is None:
            raise Exception('Could not find filename tag in xml file.')
        if size['width'] is None:
            raise Exception('Could not find width tag in xml file.')
        if size['height'] is None:
            raise Exception('Could not find height tag in xml file.')
        self.image_id += 1
        image_item = dict()
        image_item['id'] = self.image_id
        src_name = self.splitall(self.xml_path)[-2]
        image_item['file_name'] = os.path.join(src_name, 'images', file_name)
        image_item['width'] = size['width']
        image_item['height'] = size['height']
        self.coco['images'].append(image_item)
        self.image_set.add(file_name)
        return self.image_id

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
        self.annotation_id += 1
        annotation_item['id'] = self.annotation_id
        self.coco['annotations'].append(annotation_item)

    def parse_xml_files(self, xml_path):
        xml_files = fnmatch.filter(os.listdir(xml_path), '*.xml')
        total_xml_files = len(xml_files)
        # print(total_xml_files)

        for i, f in enumerate(xml_files):
            progress_txt = '[PROGRESS] Processing XML files: [%06d/%06d]\r' % (i+1, total_xml_files)
            if (i+1) % 10 == 0:
                sys.stdout.write("\r" + progress_txt)
                sys.stdout.flush()

            if not f.endswith('.xml'):
                continue

            bndbox = dict()
            size = dict()
            current_image_id = None
            current_category_id = None
            file_name = None
            size['width'] = None
            size['height'] = None
            size['depth'] = None

            xml_file = os.path.join(xml_path, f)
            # print(xml_file)

            tree = ET.parse(xml_file)
            root = tree.getroot()
            if root.tag != 'annotation':
                raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

            # elem is <folder>, <filename>, <size>, <object>
            for elem in root:
                current_parent = elem.tag
                current_sub = None
                object_name = None

                if elem.tag == 'folder':
                    continue

                if elem.tag == 'filename':
                    file_name = elem.text
                    if file_name in self.category_dict:
                        raise Exception('file_name duplicated')

                # add img item only after parse <size> tag
                elif current_image_id is None and file_name is not None and size['width'] is not None:
                    if file_name not in self.image_set:
                        current_image_id = self.add_image_item(file_name, size)
                        # print('add image with {} and {}'.format(file_name, size))
                    else:
                        raise Exception('duplicated image: {}'.format(file_name))
                        # subelem is <width>, <height>, <depth>, <name>, <bndbox>

                for subelem in elem:
                    bndbox['xmin'] = None
                    bndbox['xmax'] = None
                    bndbox['ymin'] = None
                    bndbox['ymax'] = None

                    current_sub = subelem.tag
                    if current_parent == 'object' and subelem.tag == 'name':
                        object_name = subelem.text
                        if object_name.lower() == 'fo':
                            object_name = 'ufo'
                        if object_name.lower() not in self.category_dict:
                            current_category_id = self.add_category_item(object_name.lower())
                        else:
                            current_category_id = self.category_dict[object_name.lower()]

                    elif current_parent == 'size':
                        if size[subelem.tag] is not None:
                            raise Exception('xml structure broken at size tag.')
                        size[subelem.tag] = int(subelem.text)

                    # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                    for option in subelem:
                        if current_sub == 'bndbox':
                            if bndbox[option.tag] is not None:
                                raise Exception('xml structure corrupted at bndbox tag.')
                            bndbox[option.tag] = int(option.text)

                    # only after parse the <object> tag
                    if bndbox['xmin'] is not None:
                        if object_name is None:
                            raise Exception('xml structure broken at bndbox tag')
                        if current_image_id is None:
                            raise Exception('xml structure broken at bndbox tag')
                        if current_category_id is None:
                            raise Exception('xml structure broken at bndbox tag')
                        bbox = []
                        # x
                        bbox.append(bndbox['xmin'])
                        # y
                        bbox.append(bndbox['ymin'])
                        # w
                        bbox.append(bndbox['xmax'] - bndbox['xmin'])
                        # h
                        bbox.append(bndbox['ymax'] - bndbox['ymin'])
                        # print('add annotation with {},{},{},{}'.format(object_name, current_image_id,
                        # current_category_id, bbox))
                        self.add_annotation_item(current_image_id, current_category_id, bbox)
        print('[END] Processed all XML files in given directory: [%06d/%06d]\r' % (i+1, total_xml_files))
