# model settings
_base_ = '../_base_/qdtrack_faster_rcnn_r50_fpn.py'
model = dict(
    type='TETer',
    freeze_detector=True,
    freeze_qd = True,
    method='teter',
    roi_head=dict(
        type='TETerRoIHead',
        finetune_cem=True,
        bbox_head=dict(num_classes=8),
        cem_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        cem_head=dict(
            type='ClsExemplarHead',
            num_convs=4,
            num_fcs=3,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='UnbiasedSupConLoss', temperature=0.07, contrast_mode='all',
                            pos_normalize=True,
                            loss_weight=0.25)
            , softmax_temp=-1),

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
    tracker=dict(
        type='TETerBDD',
        init_score_thr=0.7,
        obj_score_thr=0.3,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        contrastive_thr = 0.5,
        match_metric='bisoftmax'),

    # model training and testing settings
    train_cfg=dict(
        embed=dict(
            sampler=dict(
                type='CombinedSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=3,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(
                    type='IoUBalancedNegSampler',
                    floor_thr=-1,
                    floor_fraction=0,
                    num_bins=3)))))
# dataset settings
dataset_type = 'BDDVideoDataset'
data_root = 'data/bdd/bdd100k/'
ann_root = 'data/bdd/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    # comment above line and comment out the lines below if use hdf5 file.
    # dict(type='LoadMultiImagesFromFile',
    #      file_client_args=dict(
    #          img_db_path= 'data/bdd/hdf5s/100k_train.hdf5',
    #          # vid_db_path='data/bdd/hdf5s/track_train.hdf5',
    #          backend='hdf5',
    #          type='bdd')),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_ins_id=True),
    dict(
        type='SeqResize',
        img_scale=[(1296, 640), (1296, 672), (1296, 704), (1296, 736),
                   (1296, 768), (1296, 800), (1296, 720)],
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
    # comment above line and comment out the lines below if use hdf5 file.
    # dict(type='LoadImageFromFile',
    #      file_client_args=dict(
    #          vid_db_path='data/bdd/hdf5s/track_val.hdf5',
    #          backend='hdf5',
    #          type='bdd')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1296, 720),
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
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=[
        dict(
            type=dataset_type,
            load_as_video=False,
            ann_file=ann_root +
                     'annotations/det_20/det_train_cocofmt.json',
            img_prefix=data_root + 'images/100k/train/',
            pipeline=train_pipeline)
    ],
    val=dict(
        type=dataset_type,
        ann_file=ann_root +
                 'annotations/box_track_20/box_track_val_cocofmt.json',
        scalabel_gt = ann_root + 'annotations/scalabel_gt/box_track_20/val/',
        img_prefix=data_root + 'images/track/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_root +
                 'annotations/box_track_20/box_track_val_cocofmt.json',
        scalabel_gt=ann_root + 'annotations/scalabel_gt/box_track_20/val/',
        img_prefix=data_root + 'images/track/val/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[8, 11])
# checkpoint savingp
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(metric=['bbox', 'track'], interval=1)
