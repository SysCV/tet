# model settings
_base_ = '../_base_/qdtrack_faster_rcnn_r50_fpn.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    type='TETer',
    freeze_detector=False,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        type='TETerRoIHead',
        bbox_head=dict(num_classes=1230),
        cem_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        cem_head=dict(
            type='ClsExemplarHead',
            num_convs=4,
            num_fcs=3,
            embed_channels=1230,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='UnbiasedSupConLoss',
                            temperature=0.07,
                            contrast_mode='all',
                            pos_normalize=True,
                            loss_weight=0.25)
            , softmax_temp=-1),
        track_head=dict(
            type='QuasiDenseEmbedHead',
            num_convs=4,
            num_fcs=1,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='MultiPosCrossEntropyLoss',
                            loss_weight=0.25,
                            version='unbiased'),
            loss_track_aux=dict(
                type='L2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.1,
                hard_mining=True,
                loss_weight=1.0))
    ),
    tracker=dict(
        type='TETerTAO',
        init_score_thr=0.0001,
        obj_score_thr=0.0001,
        match_score_thr=0.5,
        memo_frames=10,
        momentum_embed=0.8,
        momentum_obj_score=0.5,
        match_metric='bisoftmax',
        match_with_cosine=True,
        contrastive_thr=0.5,
    ),
    train_cfg=dict(
        cem=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='CombinedSampler',
                num=256,
                pos_fraction=1,
                neg_pos_ub=0,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(type='RandomSampler'))
        )
    ),

    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300)
    )
)
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    # comment above line and comment out the lines below if use hdf5 file.
    # dict(
    #     type='LoadMultiImagesFromFile',
    #     file_client_args=dict(
    #         img_db_path='data/lvis/train_imgs.hdf5',
    #         backend='hdf5',
    #         type='lvis')),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_ins_id=True),
    dict(
        type='SeqResize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        share_params=False,
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=False, flip_ratio=0.5),
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

## dataset settings
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
            load_as_video=False,
            ann_file='data/lvis/annotations/lvisv0.5+coco_train.json',
            img_prefix='data/lvis/train2017/',
            key_img_sampler=dict(interval=1),
            ref_img_sampler=dict(num_ref_imgs=1, scope=1, method='uniform'),
            pipeline=train_pipeline)
    ),
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
# optimizer
optimizer = dict(
    # _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)


# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
# runtime settings
total_epochs = 36
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(metric=['bbox'], start=2, interval=2)
