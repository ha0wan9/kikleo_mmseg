import os
import numpy as np
from PIL import Image
import random
import os.path as osp
from pathlib import Path
import argparse
import sys
from shutil import copyfile

from KFW_dataset import KikleoFoodWasteDataset

def main(args):
    data = args.data_dir
    output = args.output
    ann_list = [str(path) for path in Path(data).rglob('*.png')]
    filename_list = [Path(ann).stem for ann in ann_list]
    random.shuffle(filename_list)
    print(f'Data files: {filename_list[:5]}...')
    print('Begin dataset creation.')
    move_sample(data, output, ann_list)
    print('Train validation split...')
    # train/val split
    split_dir = osp.join(output, 'splits')
    if not osp.isdir(split_dir):
        os.mkdir(split_dir)
    with open(osp.join(split_dir, 'train.txt'), 'w') as f:
        # select first 4/5 as train set
        train_length = int(len(filename_list) * 4 / 5)
        f.writelines(line + '\n' for line in filename_list[:train_length])
    with open(osp.join(split_dir, 'val.txt'), 'w') as f:
        # select last 1/5 as val set
        f.writelines(line + '\n' for line in filename_list[train_length:])
    print('done')
    sys.exit()

def move_sample(data, output, ann_list):
    ann_output = osp.join(output, 'labels')
    img_output = osp.join(output, 'images')
    if not osp.isdir(output):
        os.mkdir(output)
    if not osp.isdir(ann_output):
        os.mkdir(ann_output)
    if not osp.isdir(img_output):
        os.mkdir(img_output)

    conv_list = [str(path) for path in Path(data).rglob('*.jpg')]
    conv_list += [str(path) for path in Path(data).rglob('*.JPG')]
    for img_path in conv_list:
        img = Image.open(img_path)
        img.save(img_path[:-3]+'jpeg')
    img_list = [str(path) for path in Path(data).rglob('*.jpeg')]
    for file in ann_list:
        ann = np.asarray(Image.open(file))
        ann = np.squeeze(ann[..., 0])
        ann_img = Image.fromarray(ann)
        ann_img.save(osp.join(ann_output, Path(file).name))
 #       copyfile(file, osp.join(ann_output, Path(file).name))
    for file in img_list:
        copyfile(file, osp.join(img_output, Path(file).name))
    print(f'Dataset built at {output}')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str,
                        help='The path of the json files.', default=None)
    parser.add_argument('-o', '--output', type=str,
                        help='The path of the dataset to be built.', default=None)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))