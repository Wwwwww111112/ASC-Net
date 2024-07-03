import os
from glob import glob

from monai.data import Dataset

from medlab.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class BmecClsDataset(Dataset):
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
        skulls = [i.replace(subset, subset+'_mask_skull') for i in images]
        cisterns = [i.replace(subset, subset+'_mask') for i in images]
        cerebellum = [i.replace(subset, subset+'_mask_cere') for i in images]
        labels = [classes.index(i.split('/')[-3]) for i in images]

        assert len(images) > 0, 'No images found in {}'.format(data_root)
        assert len(images) == len(skulls) == len(cisterns) == len(cerebellum) == len(labels), 'The number of images:{} skulls:{} cisterns:{} cerebellum:{} labels:{} should be equal'.format(
            len(images), len(skulls), len(cisterns), len(cerebellum),len(labels))

        data = [{'images': i, 'skulls': s, 'cisterns':c, 'cerebellum':r,'labels': l} for i, s, c,r, l in zip(images, skulls, cisterns,cerebellum, labels)]
        transforms = TRANSFORMS.build(transforms)
        # transforms = None
        super().__init__(data=data, transform=transforms)

    @property
    def classes(self):
        return self.META_INFO['classes']
#两个mask约束'skulls': s, 'cisterns':c, 'cerebellum':r,
# import os
# from glob import glob

# from monai.data import Dataset

# from medlab.registry import DATASETS, TRANSFORMS


# @DATASETS.register_module()
# class BmecClsDataset(Dataset):
#     META_INFO = {'classes': list()}

#     def __init__(
#             self,
#             data_root: str,
#             data_suffix: str = '.png',
#             subset: str = 'train',
#             transforms: dict = None
#     ):

#         assert subset in ['train', 'val', 'test']

#         classes = sorted(os.listdir(os.path.join(data_root, subset)))
#         self.META_INFO['classes'] = classes
#         images = sorted(glob(os.path.join(data_root, subset, '*', '*', '*{}'.format(data_suffix))))
#         skulls = [i.replace(subset, subset+'_mask_skull') for i in images]
#         # cisterns = [i.replace(subset, subset+'_mask') for i in images]
#         cerebellum = [i.replace(subset, subset+'_mask_cere') for i in images]
#         labels = [classes.index(i.split('/')[-3]) for i in images]

#         assert len(images) > 0, 'No images found in {}'.format(data_root)
#         assert len(images) == len(cerebellum) == len(skulls) == len(labels), 'The number of images:{} cerebellum:{} skulls:{} labels:{} should be equal'.format(
#             len(images), len(cerebellum), len(cisterns),len(labels))

#         data = [{'images': i, 'cerebellum': s, 'skulls':c, 'labels': l} for i, s, c, l in zip(images, cerebellum, skulls, labels)]
#         transforms = TRANSFORMS.build(transforms)
#         # transforms = None
#         super().__init__(data=data, transform=transforms)

#     @property
#     def classes(self):
#         return self.META_INFO['classes']
