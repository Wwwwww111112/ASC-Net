import glob
import os.path as osp

import pandas as pd
from monai.data import Dataset
from sklearn.model_selection import train_test_split

from medlab.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class BaseSegDataset(Dataset):
    def __init__(
            self,
            data_root: str,
            img_dir: str,
            label_dir: str,
            img_suffix: str,
            label_suffix: str,
            subset: str = None,
            test_size: float = 0.2,
            seed: int = 1234,
            transforms: dict = None,
            annotation_file: str = None,
    ):
        """
        base segmentation dataset
        if annotation_file is not None, use annotation file to load data
        else, use subset and test_size to split data

        if subset is None, use all data
        if subset is 'train', use train data
        if subset is 'val', use val data
        :param data_root: dataset root path
        :param img_dir:  image dir
        :param label_dir: label dir
        :param img_suffix: image suffix
        :param label_suffix: label suffix
        :param subset: train or val or None
        :param test_size: test size
        :param seed: random seed
        :param transforms: MONAI data transforms cfg
        :param annotation_file: annotation file path
        """
        assert subset in ['train', 'val', None]

        if annotation_file is not None:
            dataframe = pd.read_csv(osp.join(data_root, annotation_file), header=None, sep=' ')
            images = [osp.join(data_root, i) for i in dataframe[0].to_list()]
            labels = [osp.join(data_root, i) for i in dataframe[1].to_list()]
        else:
            images = sorted(glob.glob(osp.join(data_root, img_dir, '*' + img_suffix)))
            labels = sorted(glob.glob(osp.join(data_root, label_dir, '*' + label_suffix)))

        assert len(images) > 0, 'No images found in {}'.format(osp.join(data_root, img_dir))
        assert len(images) == len(labels), 'The number of images:{} and labels:{} should be equal'.format(len(images),
                                                                                                          len(labels))

        if subset is not None:
            train_x, val_x, train_y, val_y = train_test_split(images, labels, test_size=test_size, random_state=seed)
            if subset == 'train':
                images = train_x
                labels = train_y
            else:
                images = val_x
                labels = val_y

        data = [{'images': i, 'labels': m} for i, m in zip(images, labels)]
        transforms = TRANSFORMS.build(transforms)
        super().__init__(data=data, transform=transforms)
