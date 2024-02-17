_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=200, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=200,
        milestones=[80, 150],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best='PCK', rule='greater'))

# base dataset settings
dataset_type = 'JhmdbFlowDataset'
data_mode = 'topdown'
data_root = '/scratch/PI/cqf/har_data/jhmdb'

# codec settings
codec = dict(
    type='PoseSegmentationMask', input_size=(256, 256), mask_size=(64, 64), dataset_type=dataset_type, sigma=3, use_flow=True)

# model settings
model = dict(
    type='TopdownPoseEstimatorPSM',
    data_preprocessor=dict(
        type='PoseFlowDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    flownet=dict(
        type='RAFT',
        args_dict=dict(
            flow_model_path = '/home/zpengac/RAFT/models/raft-sintel.pth',
            global_flow = True,
            dataset = 'sintel',
            small = False
        )
    ),
    backbone_flow=dict(
        type='ResNet',
        depth=18,
        in_channels=2,
    ),
    backbone=dict(
        type='ResNet',
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
    ),
    head=dict(
        type='PointHead',
        in_channels=512,
        out_channels=15,
        num_layers=3,
        hid_channels=256,
        train_num_points=256,
        subdivision_steps=3,
        scale=1/8,
        use_flow=True,
        loss=dict(type='MultipleLossWrapper', losses=[
             dict(type='BodySegTrainLoss', loss_weight=1, use_target_weight=True),
             dict(type='JointSegTrainLoss', loss_weight=2, neg_weight=0.95, use_target_weight=True),
             dict(type='BodySegTrainLoss', use_target_weight=True)
             ]),
        decoder=codec))

find_unused_parameters = True

# pipelines
train_pipeline = [
    dict(type='LoadImagePair'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        rotate_factor=60,
        scale_factor=(0.75, 1.25)),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImagePair'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/Sub1_train.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/Sub1_test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(type='PSMMetricWrapper', use_flow=True, metric_config=dict(type='JhmdbPCKAccuracy', thr=0.2, norm_item=['bbox', 'torso']), outfile_prefix='logs/jhmdb24/td-hm_res50_8xb64-20e_jhmdb-sub1-256x256'),
]
test_evaluator = val_evaluator
