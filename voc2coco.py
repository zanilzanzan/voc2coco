import argparse
import os

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Convert annotations from VOC format to COCO format')
    parser.add_argument('--annot_type', type=str, required=True,
                        help='The file format of the source annotation file: json or xml.')
    if parser.parse_known_args()[0].annot_type.lower() == 'json':
        option_parser = argparse.ArgumentParser()
        option_parser.add_argument('--json_input_file', type=str, required=True,
                                   help='.json annotation file path in VOC format in path/to/file.json form.')
        option_parser.add_argument('--output_dir', required=True,
                                   help='.json annotation file path in VOC format in path/to/file.json form.')
        option_parser.add_argument('--val_percent', type=int, required=False, default=0,
                                   help='Enter the percentage of validation partition of the dataset. Between 0-100.')
    elif parser.parse_known_args()[0].annot_type.lower() == 'xml':
        option_parser = argparse.ArgumentParser()
        option_parser.add_argument('--xml_input_dir', type=str, required=True,
                                   help='directory path for .xml annotation files in VOC format ')
        option_parser.add_argument('--output_dir', required=True,
                                   help='.json annotation file path in VOC format in path/to/file.json form.')
        option_parser.add_argument('--val_percent', type=int, required=False, default=0,
                                   help='Enter the percentage of validation partition of the dataset. Between 0-100.')
    else:
        raise Exception('--annot_type argument was given wrong. Either json or xml, it is supposed to be.')

    return option_parser.parse_known_args()


if __name__ == "__main__":
    opt = parse_args()
    dest = opt[0].output_dir
    val_part = opt[0].val_percent
    if not os.path.exists(dest):
        os.makedirs(os.path.join(dest))
        if val_part:
             os.makedirs(os.path.join(dest, 'val'))
    print('-'*80)

    if val_part > 0:
        if not val_part % 10 == 0 or val_part > 90:
            raise Exception('--val_percent must be multiples of 10, and should be smaller than 100.')
        else:
            print('[INFO] {}% of the data will be used as validation set.'.format(val_part))
    else:
        print('[INFO] Converted data will not contain a validation set.')

    print('[START] Format conversion starts.')

    if opt[1][1] == 'json':
        from VOCJSON2COCO import VOCJSON2COCO
        src = opt[0].json_input_file
        coco_annotations = VOCJSON2COCO(src, dest, val_part)
        coco_annotations.convert_2_coco()

    # if opt[1][1] == 'json':
    #     from JSON2COCO import JSON2COCO
    #     src = opt[0].json_input_file
    #     coco_annotations = JSON2COCO(src, dest, val_part)
    #     coco_annotations.convert_2_coco()
    else:
        from VOCXML2COCO import VOCXML2COCO
        src = opt[0].xml_input_dir
        coco_annotations = VOCXML2COCO(os.path.join(src, 'images'), dest)
        file_name = coco_annotations.convert_2_coco()
        # print(opt[0].xml_input_dir)

