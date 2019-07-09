import json
import os
import argparse


def merge_json_files(src, dest):
    if not os.path.exists(src):
        raise Exception('Given .json annotation input file doesn\'t exist.')

    if not os.path.exists(dest):
        os.makedirs(dest)

    annot_merged = dict()
    annot_merged['type'] = 'instances'
    annot_merged['info'] = list()
    annot_merged['licenses'] = list()
    annot_merged['images'] = list()
    annot_merged['annotations'] = list()
    annot_merged['categories'] = list()

    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                print(json_path)
                with open(json_path) as f:
                    annot = json.load(f)

                for key in annot.keys():
                    if not key == 'type':
                        print(key)
                        annot_merged[key] += (annot[key])

                print(len(annot_merged['images']))
    annot_merged['categories'] = list({v['id']: v for v in annot_merged['categories']}.values())

    json_save_name = os.path.basename(os.path.normpath(src)) + '.json'
    coco_file = os.path.join(dest, json_save_name)
    json.dump(annot_merged, open(coco_file, 'w'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", "--SourceDirectoryPath", required=True,
                    help="Source directory path for .json files")
    ap.add_argument("-dest", "--DestinationDirectoryPath", required=True,
                    help="Destination directory path for merged .json file")
    args = vars(ap.parse_args())

    src = args["SourceDirectoryPath"]
    dest = args["DestinationDirectoryPath"]

    merge_json_files(src, dest)

