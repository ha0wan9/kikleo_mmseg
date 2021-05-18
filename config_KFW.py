from mmcv import Config
from mmseg.apis import set_random_seed

NB_CLASSES = 101
BATCH_SIZE = 8
NB_GPU = 1

def config_KFW_PSPNet(data_root='data/Kikleo_FW', img_dir='images', ann_dir='labels', split_dir='splits'):

    cfg = Config.fromfile('configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py')
    # Since we use ony one GPU, BN is used instead of SyncBN
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    # modify num classes of the model in decode/auxiliary head
    cfg.model.decode_head.num_classes = NB_CLASSES
    cfg.model.auxiliary_head.num_classes = NB_CLASSES

    # Modify dataset type and path
    cfg.dataset_type = 'KikleoFoodWasteDataset'
    cfg.data_root = data_root

    cfg.data.samples_per_gpu = BATCH_SIZE
    cfg.data.workers_per_gpu = NB_GPU

    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    cfg.crop_size = (256, 256)
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(512, 256), ratio_range=(0.5, 2.0)),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **cfg.img_norm_cfg),
        dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(512, 256),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.img_dir = img_dir
    cfg.data.train.ann_dir = ann_dir
    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.train.split = split_dir+'/train.txt'

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = img_dir
    cfg.data.val.ann_dir = ann_dir
    cfg.data.val.pipeline = cfg.test_pipeline
    cfg.data.val.split = split_dir+'/val.txt'

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = img_dir
    cfg.data.test.ann_dir = ann_dir
    cfg.data.test.pipeline = cfg.test_pipeline
    cfg.data.test.split = split_dir+'/val.txt'

    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    cfg.load_from = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = 'train_KFW'

    cfg.total_iters = 4000
    cfg.log_config.interval = 100
    cfg.evaluation.interval = 4000
    cfg.checkpoint_config.interval = 4000

    # Set seed to facitate reproducing the result
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # Let's have a look at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')

    return cfg


def config_KFW_OCRNet(data_root='data/Kikleo_FW', img_dir='images', ann_dir='labels', split_dir='splits'):
    cfg = Config.fromfile('configs/ocrnet/ocrnet_hr48_512x512_80k_kfw.py')
    # Since we use ony one GPU, BN is used instead of SyncBN
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    # modify num classes of the model in decode/auxiliary head
    cfg.model.decode_head.num_classes = NB_CLASSES
    cfg.model.auxiliary_head.num_classes = NB_CLASSES

    # Modify dataset type and path
    cfg.dataset_type = 'KikleoFoodWasteDataset'
    cfg.data_root = data_root

    cfg.data.samples_per_gpu = BATCH_SIZE
    cfg.data.workers_per_gpu = NB_GPU

    # Set up working dir to save files and logs.
    cfg.work_dir = 'train_KFW'

    cfg.total_iters = 4000
    cfg.log_config.interval = 100
    cfg.evaluation.interval = 4000
    cfg.checkpoint_config.interval = 4000

    # Set seed to facitate reproducing the result
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # Let's have a look at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')
    return cfg