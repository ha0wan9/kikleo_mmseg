_base_ = './deeplabv3plus_r50-d8_512x1024_80k_kfw.py'
seed = 0
total_iters = 8000
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(pretrained='open-mmlab://resnet101_v1c',
             backbone=dict(depth=101,
                           norm_cfg=norm_cfg),
             decode_head=dict(
                 num_classes=101,
                 norm_cfg=norm_cfg),
             auxiliary_head=dict(
                 num_classes=101,
                 norm_cfg=norm_cfg))
