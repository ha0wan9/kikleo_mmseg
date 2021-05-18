import json
import os.path as osp

meta_filepath = 'masks_meta.json'
data_root = 'data/Kikleo_FW'
img_dir = 'images'
ann_dir = 'labels'
split_dir = 'splits'

with open(meta_filepath, 'r') as file:
    meta = json.load(file)
    gt_human_color = meta['settings']['gt_human_color']

classes = list(gt_human_color.keys())
classes.insert(0, 'BG')
print(f'classes: {classes}')
palette = [gt_human_color[key] for key in gt_human_color.keys()]
palette.insert(0, [0, 0, 0])
print(f'palette: {palette}')
print(f'nb_classes: {len(classes)}')
assert len(palette) == len(classes), f'Inequality number between classes and palette colors: {len(palette)}'

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class KikleoFoodWasteDataset(CustomDataset):

  CLASSES = tuple(classes)
  PALETTE = palette
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.jpeg', seg_map_suffix='.png',
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None