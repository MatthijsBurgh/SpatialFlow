# model settings
model = dict(
    type='SpatialFlow',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='SpatialFlowHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        stacked_mask_convs=1,
        stacked_stuff_convs=4,
        dcn_cls_convs_idx=None,
        dcn_reg_convs_idx=None,
        dcn_mask_convs_idx=None,
        dcn_stuff_convs_idx=None,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
        out_channels=256,
        featmap_strides=[8, 16, 32]),
    mask_head=dict(
        type='FCNMaskHead',
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=80,
        loss_mask=dict(
            type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
    stuff_head=dict(
        type='StuffHead',
        stuff_num_classes=54,
        in_channels=256,
        feat_channels=128,
        feat_indexes=[0, 1, 2],
        feat_strides=[8, 16, 32],
        out_stride=4,
        conv_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        loss_stuff=dict(
            type='CrossEntropyLoss', ignore_index=255, loss_weight=0.25)))
# training and testing settings
# training and testing settings
train_cfg = dict(
    single_stage=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    single_stage_nms=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.4),
        max_per_img=100),
    single_stage_mask=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(type='PseudoSampler', add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    single_stage=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.4),
        max_per_img=100),
    single_stage_mask=dict(mask_thr_binary=0.5))
# panoptic settings for model
confidence_thr = 0.37
overlap_thr = 0.37
stuff_area_limit = 4900
using_bbox = True
bbox_overlap_thr = 0.5
