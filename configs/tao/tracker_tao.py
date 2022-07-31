# model settings
_base_ = './cem_lvis.py'
model = dict(
    freeze_detector=True,
    freeze_cem=True,
    method='teter',
    roi_head=dict(bbox_head=dict(num_classes=1230),
                  track_head=dict(
                      type='QuasiDenseEmbedHead',
                      num_convs=4,
                      num_fcs=1,
                      embed_channels=256,
                      norm_cfg=dict(type='GN', num_groups=32),
                      loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25),
                      loss_track_aux=dict(
                          type='L2Loss',
                          neg_pos_ub=3,
                          pos_margin=0,
                          neg_margin=0.1,
                          hard_mining=True,
                          loss_weight=1.0))
                  ),

    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.5, class_agnostic=True, split_thr=100000),
            max_per_img=50)
            )
)

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    # dict(
    #     type='LoadMultiImagesFromFile',
    #     file_client_args=dict(
    #         img_db_path='data/tao/tao_train_imgs.hdf5',
    #         backend='hdf5',
    #         type='tao')),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_ins_id=True),
    dict(
        type='SeqResize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        share_params=True,
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='SeqDefaultFormatBundle'),
    dict(
        type='SeqCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices'],
        ref_prefix='ref'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadImageFromFile',
    #      file_client_args=dict(
    #          img_db_path='data/tao/tao_val_imgs.hdf5',
    #          backend='hdf5',
    #          type='tao')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
dataset_type = 'TaoDataset'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            classes='data/lvis/annotations/lvis_classes.txt',
            ann_file='data/tao/annotations/train_ours.json',
            img_prefix='data/tao/frames/',
            key_img_sampler=dict(interval=1),
            ref_img_sampler=dict(num_ref_imgs=1, scope=1, method='uniform'),
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        classes='data/lvis/annotations/lvis_classes.txt',
        ann_file='data/tao/annotations/validation_ours.json',
        img_prefix='data/tao/frames/',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes='data/lvis/annotations/lvis_classes.txt',
        ann_file='data/tao/annotations/validation_ours.json',
        img_prefix='data/tao/frames/',
        ref_img_sampler=None,
        pipeline=test_pipeline)
)
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[8, 11])
total_epochs = 12
load_from = None
evaluation = dict(metric=['track'], start=8, interval=1, resfile_path='/scratch/tmp/')
work_dir = './saved_models/teter_swinT/'
