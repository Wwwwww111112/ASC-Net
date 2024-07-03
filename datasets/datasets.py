from monai.data import CacheDataset, Dataset, PersistentDataset

from medlab.registry import DATASETS

DATASETS.register_module(name='Dataset', module=Dataset)
DATASETS.register_module(name='CacheDataset', module=CacheDataset)
DATASETS.register_module(name='PersistentDataset', module=PersistentDataset)
