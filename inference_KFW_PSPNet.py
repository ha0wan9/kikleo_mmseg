import sys
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os.path as osp

import mmcv
from mmcv.runner import load_checkpoint
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette, get_classes

import KFW_dataset
from config_KFW import config_KFW_PSPNet, config_KFW_OCRNet


def main(args):

    test_data = [str(img) for img in Path(args.test_images).rglob('*.jpeg')]

    cfg = config_KFW_PSPNet(data_root='data/Kikleo_FW', img_dir='images', ann_dir='labels', split_dir='splits')
    datasets = [build_dataset(cfg.data.train)]
    # Build the detector
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    #model = init_segmentor(cfg, args.checkpoint, device='cuda:0')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = datasets[0].PALETTE
    model.cfg = cfg
    model.to('cuda:0')
    model.eval()

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    for data in test_data:
        img = mmcv.imread(data)
        result = inference_segmentor(model, img)
        #plt.figure(figsize=(8, 6))
        #show_result_pyplot(model, img, result, model.PALETTE, path=data)
        img = model.show_result(
            img, result, palette=model.PALETTE, show=False, opacity=0.8)
        img_result = Image.fromarray(mmcv.bgr2rgb(img))
        img_result.save(data+'_infer.png')
        sys.exit()


def show_result_pyplot(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.5,
                       path=None):
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.savefig(Path(path).name+'_infer.png')


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