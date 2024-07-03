import os

from monai.data import Dataset
from sklearn.model_selection import train_test_split

from medlab.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class BaseClsDataset(Dataset):
    META_INFO = {'classes': list()}

    def __init__(
            self,
            data_root: str,
            data_suffix: str,
            subset: str = None,
            test_size: float = 0.2,
            seed: int = 1234,
            transforms: dict = None,
            annotation_file: str = None,
    ):
        """
        base classification dataset
        if annotation_file is not None, use annotation file to load data
        else, use subset and test_size to split data

        if subset is None, use all data
        if subset is 'train', use train data
        if subset is 'val', use val data

        :param data_root: data root path
        :param data_suffix: image suffix
        :param subset: train or val or None
        :param test_size: test size
        :param seed: random seed
        :param transforms: MONAI data transforms cfg
        :param annotation_file: annotation file path
        """
        assert subset in ['train', 'val', None]

        classes = sorted(os.listdir(data_root))
        self.META_INFO['classes'] = classes
        images = []
        labels = []
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith(data_suffix):
                    images.append(os.path.join(root, file))
                    labels.append(classes.index(os.path.basename(root)))

        assert len(images) > 0, 'No images found in {}'.format(data_root)
        assert len(images) == len(labels), 'The number of images:{} and labels:{} should be equal'.format(len(images),
                                                                                                          len(labels))

        if subset is not None:
            train_x, val_x, train_y, val_y = train_test_split(images, labels, test_size=test_size, random_state=seed,
                                                              stratify=labels)
            if subset == 'train':
                images = train_x
                labels = train_y
            else:
                images = val_x
                labels = val_y
        data = [{'images': i, 'labels': m} for i, m in zip(images, labels)]
        transforms = TRANSFORMS.build(transforms)
        super().__init__(data=data, transform=transforms)

    @property
    def classes(self):
        return self.META_INFO['classes']
