
import os
from glob import glob

from monai.data import Dataset

from medlab.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class AscClsDataset(Dataset):
    META_INFO = {'classes': list()}

    def __init__(
            self,
            data_root: str,
            data_suffix: str = '.png',
            subset: str = 'train',
            transforms: dict = None
    ):

        assert subset in ['train', 'val', 'test']

        classes = sorted(os.listdir(os.path.join(data_root, subset)))
        self.META_INFO['classes'] = classes
        images = sorted(glob(os.path.join(data_root, subset, '*', '*', '*{}'.format(data_suffix))))
        cisterns = [i.replace(subset, subset+'_mask') for i in images]
        cerebellum = [i.replace(subset, subset+'_mask_cere') for i in images]
        labels = [classes.index(i.split('/')[-3]) for i in images]

        assert len(images) > 0, 'No images found in {}'.format(data_root)
        assert len(images) == len(cerebellum) == len(cisterns) == len(labels), 'The number of images:{} cerebellum:{} cisterns:{} labels:{} should be equal'.format(
            len(images), len(cerebellum), len(cisterns),len(labels))

        data = [{'images': i, 'cerebellum': s, 'cisterns':c, 'labels': l} for i, s, c, l in zip(images, cerebellum, skulls, labels)]
        transforms = TRANSFORMS.build(transforms)
        # transforms = None
        super().__init__(data=data, transform=transforms)

    @property
    def classes(self):
        return self.META_INFO['classes']
