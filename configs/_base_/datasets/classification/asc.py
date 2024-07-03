dataset_type = 'AscClsDataset'
data_root = '/mnt/dat1/wit/xxwu/medlab/code/data041721+80-skull+cmc+cere'

# in_channels = 3  # 数据集的图片通道数, 彩色为3, 灰度为1
num_classes = 2  # 分类类别（含背景）

spatial_size = (224, 224)

train_transform = [dict(type='LoadImaged', keys=['images', 'cerebellum', 'cisterns'], ensure_channel_first=True, image_only=True),
                   dict(type='RandFlipd', keys=['images', 'cerebellum', 'cisterns'], prob=0.5, spatial_axis=0),
                   dict(type='ScaleIntensityd', keys=['images', 'cerebellum', 'cisterns']),
                   dict(type='RandZoomd', keys=['images', 'cerebellum', 'cisterns'], prob=0.5, min_zoom=0.8, max_zoom=1.2, mode=('bilinear', 'nearest', 'nearest')),
                   dict(type='RandRotated', keys=['images', 'cerebellum', 'cisterns'], prob=0.5, range_x=0.3, mode=('bilinear', 'nearest', 'nearest')),
                   dict(type='RandAxisFlipd', keys=['images', 'cerebellum', 'cisterns'], prob=0.5),
                   dict(type='Resized', keys=['images', 'cerebellum', 'cisterns'], spatial_size=spatial_size, mode=('bilinear', 'nearest', 'nearest'))]

val_transform = [dict(type='LoadImaged', keys=['images', 'cerebellum', 'cisterns'], ensure_channel_first=True),
                 dict(type='ScaleIntensityd', keys=['images', 'cerebellum', 'cisterns']),
                 dict(type='Resized', keys=['images', 'cerebellum', 'cisterns'], spatial_size=spatial_size, mode=('bilinear', 'nearest', 'nearest'))]

batch_size = 16

train_dataloader = dict(batch_size=batch_size, num_workers=12,
                        dataset=dict(type=dataset_type, data_root=data_root, subset='train',
                                     transforms=train_transform))

val_dataloader = dict(batch_size=1, num_workers=12,
                      dataset=dict(type=dataset_type, data_root=data_root, subset='val',
                                   transforms=val_transform))

test_dataloader = dict(batch_size=1, num_workers=12,
                      dataset=dict(type=dataset_type, data_root=data_root, subset='test',
                                   transforms=val_transform))

inferer = dict(type='SimpleInferer')

train_cfg = dict()  # 训练时的配置
val_cfg = dict(inferer=inferer)  # 验证时的配置
test_cfg = dict(inferer=inferer) # 测试时的配置