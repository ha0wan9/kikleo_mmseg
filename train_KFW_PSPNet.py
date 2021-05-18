import argparse
import sys
import os.path as osp

import mmcv
from mmcv import Config
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor, set_random_seed

import KFW_dataset
from config_KFW import config_KFW_PSPNet, config_KFW_OCRNet

NB_CLASSES = 101
BATCH_SIZE = 8
NB_GPU = 1

def main(args):
    #cfg = config_KFW_PSPNet(data_root='data/Kikleo_FW', img_dir='images', ann_dir='labels', split_dir='splits')
    cfg = Config.fromfile('configs/ocrnet/ocrnet_hr48_512x512_80k_kfw.py')

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    model.PALETTE = datasets[0].PALETTE
    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--checkpoint', type=str,
                        help='The path of the checkpoint file.', default=None)
    parser.add_argument('-ti', '--test_images', type=str,
                        help='The path of the images to be tested.', default=None)
    parser.add_argument('-sd', '--save_dir', type=str,
                        help='The path for saving results.', default=None)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
