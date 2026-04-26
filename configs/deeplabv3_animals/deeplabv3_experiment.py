_base_ = [
    'deeplabv3plus_r50-d8_4xb4-40k_voc12aug-512x512.py'
]
train_cfg = dict(
    _delete_=True,                
    type='EpochBasedTrainLoop',
    max_epochs=100,
)# ========== ПАРАМЕТРЫ МОДЕЛИ ==========
classes = ['background', 'cat', 'dog']  
num_classes = len(classes)

palette=[[0, 0, 0], [1, 1, 1], [2, 2, 2]] 
metainfo = dict(classes=classes, palette=palette)

model = dict(
    data_preprocessor=dict(
        _delete_=True,  
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        size=(256, 256),
    ),
    decode_head=dict(
        num_classes=num_classes,
        loss_decode=dict(
                type='FocalLoss',           # поменяем Focal Loss вместо CrossEntropy
                use_sigmoid=True,
                gamma=2.0,                  
                alpha=0.25,                 
        ),
    ),
    auxiliary_head=dict(
        num_classes=num_classes,
        loss_decode=            dict(
                type='FocalLoss',           
                use_sigmoid=True,
                gamma=2.0,                  
                alpha=0.25,                 
                loss_weight=0.3
            ),
    ),
)

# ========== ДАТАСЕТЫ ==========
dataset_type = 'mmseg.datasets.basesegdataset.BaseSegDataset'
data_root = '/home/ubuntu/animal_segmentation/animal_dataset'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='RandomRotate', prob=0.5, degree=10),  # Вращение
    dict(type='RandomFlip', prob=0.5),  # Отражение
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img_path='images', seg_map_path='masks'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img_path='images', seg_map_path='masks'),
        pipeline=test_pipeline
    )
)

# ========== МЕТРИКИ ==========
val_evaluator = dict(type='IoUMetric', iou_metrics=[ 'mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator

# ========== ОБУЧЕНИЕ ==========

optim_wrapper = dict(
    _delete_=True, 
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=5),
    dict(type='PolyLR', power=0.9, by_epoch=True)
]

# ========== LOGGING & CLEARML ==========
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend', save_dir='vis_data'),  # ClearML это подхватит
]

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(type='ClearMLVisBackend',  # <-- Добавляем ClearML
             init_kwargs=dict(
                 project_name='Animals_Segmentation_Experiment',
                 task_name='DeepLabV3_ResNet50',
             )),
    ],
    name='visualizer'
)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(type='CheckpointHook', interval=100, save_best='mDice'),
    visualization=dict(type='SegVisualizationHook', draw=False)
)

# ========== РАБОЧАЯ ДИРЕКТОРИЯ ==========
work_dir = '/home/ubuntu/animal_segmentation/work_dir_experiment'
load_from = "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x512_20k_voc12aug/deeplabv3_r50-d8_512x512_20k_voc12aug_20200617_010906-596905ef.pth"