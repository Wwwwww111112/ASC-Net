from monai.metrics import (ConfusionMatrixMetric, DiceMetric,
                           HausdorffDistanceMetric, MeanIoU, ROCAUCMetric)

from medlab.registry import METRICS

METRICS.register_module(name='DiceMetric', module=DiceMetric)
METRICS.register_module(name='MeanIoU', module=MeanIoU)
METRICS.register_module(name='ConfusionMatrixMetric', module=ConfusionMatrixMetric)
METRICS.register_module(name='HausdorffDistanceMetric', module=HausdorffDistanceMetric)
METRICS.register_module(name='ROCAUCMetric', module=ROCAUCMetric)
