import json
import os
import shutil


def merge_json_files(src, dest):
    if not os.path.exists(src):
        raise Exception('Given .json annotation input file doesn\'t exist.')

    annot_merged = dict()
    annot_merged['type'] = 'instances'
    annot_merged['images'] = list()
    annot_merged['annotations'] = list()
    annot_merged['categories'] = list()

    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                with open(json_path) as f:
                    annot = json.load(f)

                for key in annot.keys():
                    if not key == 'type':
                        print(key)
                        annot_merged[key] += (annot[key])

                annot_merged['categories'] = list({v['id']:v for v in annot_merged['categories']}.values())

    coco_file = os.path.join(dest, 'merged.json')
    json.dump(annot_merged, open(coco_file, 'w'))

if __name__ == '__main__':
    src = '/media/anil/data_4TB/datasets/tfrecord_files/train_images/'
    merge_json_files(src, src)

