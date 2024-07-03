from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
from monai.data import decollate_batch
from monai.utils.enums import PostFix
from sklearn.metrics import roc_curve, auc
from medlab.registry import INFERERS, TASKS, TRANSFORMS
from sklearn.metrics import confusion_matrix
from .base_task import BaseTask
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
@TASKS.register_module()
class AscClsTask(BaseTask):
    def __init__(self, *args, **kwargs):
        """
        base classification task, one input, one output
        """
        super().__init__(*args, **kwargs)
        self.val_inferer = INFERERS.build(self.val_cfg.get('inferer', dict(type='SimpleInferer')))
        self.test_inferer = INFERERS.build(self.test_cfg.get('inferer', dict(type='SimpleInferer')))
        num_classes = kwargs.get('num_classes', None)
        assert num_classes is not None, "num_classes must be specified in model"
        if num_classes == 1:
            self.post_pred_act = TRANSFORMS.build([
                dict(type='ToDevice', device='cpu'),
                dict(type='Activations', sigmoid=True)
            ])
            self.post_pred_cls = TRANSFORMS.build([
                dict(type='AsDiscrete', threshold=0.5)
            ])
            self.post_label = TRANSFORMS.build([dict(type='ToDevice', device='cpu')])

            post_save = []
        else:
            self.post_pred_act = TRANSFORMS.build([
                dict(type='ToDevice', device='cpu'),
                dict(type='Activations', softmax=True)
            ])

            self.post_pred_cls = TRANSFORMS.build([
                dict(type='AsDiscrete', argmax=True, to_onehot=num_classes),
            ])
            self.post_label = TRANSFORMS.build([
                dict(type='AsDiscrete', to_onehot=num_classes),
                dict(type='ToDevice', device='cpu')
            ])
            post_save = [
                dict(type='AsDiscreted', keys='preds', argmax=True)
            ]
        save_dir = self.test_cfg.get('save_dir', None)
        if save_dir is not None:
            post_save.extend([
                dict(
                    type='CopyItemsd',
                    keys=PostFix.meta("images"),
                    times=1,
                    names=PostFix.meta("preds")
                ),
                dict(
                    type='SaveClassificationd',
                    keys='preds',
                    saver=None,
                    meta_keys=None,
                    output_dir=save_dir,
                    filename='predictions.csv',
                    delimiter=",",
                    overwrite=True)
            ])
        self.post_save = TRANSFORMS.build(post_save)

        if 'ROCAUCMetric' in self.metrics_key:
            index = self.metrics_key.index('ROCAUCMetric')
            self.metrics_key.pop(index)
            self.roc_auc_metric = self.metrics.pop(index)
        else:
            self.roc_auc_metric = None

        self.train_pred = []
        self.train_label = []
        self.val_pred = []
        self.val_label = []
        self.test_pred = []
        self.test_label = []

    def forward(self, images, cisterns,cerebellum):
        return self._model(images,cisterns,cerebellum)
    # def forward(self, images, cerebellum, skulls):
    #     return self._model(images, cerebellum, skulls)

    def training_step(self, batch, batch_idx):
        """
        training step for classification task
        :param batch: batch data
        :param batch_idx: batch index
        :return: loss
        """

        images = batch['images']
        cisterns = batch['cisterns']
        cerebellum=batch['cerebellum']
        labels = batch['labels']

        batch_size = images.shape[0]
        outputs = self.forward(images, cerebellum,cisterns)


        loss = self.loss_func(outputs, labels)

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        
        outputs = [self.post_pred_act(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        self.train_pred.extend(outputs)
        self.train_label.extend(labels)

        self.log("train_step_loss", loss.item(), sync_dist=True, batch_size=batch_size)
        return loss
    
    def on_train_epoch_end(self) -> None:
        if self.roc_auc_metric is not None:
            self.roc_auc_metric(self.train_pred, self.train_label)
        
        self.train_pred = [self.post_pred_cls(i) for i in self.train_pred]
        if len(self.metrics) > 0:
            for metric in self.metrics:
                metric(self.train_pred, self.train_label)
        
        self.train_pred.clear()
        self.train_label.clear()

        self.log_dict(self.parse_metrics(prefix='train_'), sync_dist=True)


    def validation_step(self, batch, batch_idx):
        """
        validation step for classification task
        :param batch: batch data
        :param batch_idx: batch index
        :return: None
        """
        images = batch['images']
        cisterns = batch['cisterns']
        cerebellum=batch['cerebellum']
        labels = batch['labels']

        batch_size = images.shape[0]
        outputs = self.forward(images, cerebellum,cisterns)

        loss = self.loss_func(outputs, labels)

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        outputs = [self.post_pred_act(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        self.val_pred.extend(outputs)
        self.val_label.extend(labels)

        self.log('val_step_loss', loss.item(), sync_dist=True, batch_size=batch_size)

    def on_validation_epoch_end(self):
        
        if self.roc_auc_metric is not None:
            self.roc_auc_metric(self.val_pred, self.val_label)
        
        self.val_pred = [self.post_pred_cls(i) for i in self.val_pred]

        if len(self.metrics) > 0:
            for metric in self.metrics:
                metric(self.val_pred, self.val_label)

        self.val_pred.clear()
        self.val_label.clear()

        self.log_dict(self.parse_metrics(), sync_dist=True)

    def test_step(self, batch, batch_idx):
        """
        test step for classification task, save predictions to csv file
        :param batch: batch data
        :param batch_idx: batch index
        :return:
        """
        images = batch["images"]
        cisterns = batch["cisterns"]
        cerebellum=batch['cerebellum']
        labels = batch["labels"]

        outputs = self.forward(images, cerebellum, cisterns)

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        batch["preds"] = outputs

        outputs = [self.post_pred_act(i) for i in decollate_batch(outputs)]
        self.test_pred.extend([i[1].item() for i in outputs])
        self.test_label.extend([i for i in decollate_batch(labels)])
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        if self.roc_auc_metric is not None:
            self.roc_auc_metric(outputs, labels)

        outputs = [self.post_pred_cls(i) for i in outputs]

        
        if len(self.metrics) > 0:
            for metric in self.metrics:
                metric(outputs, labels)

        for i in decollate_batch(batch):
            self.post_save(i)
            
    
    def on_test_epoch_end(self):
        """
        test epoch end hook, parse metrics
        """
        print(self.parse_metrics(prefix='test'))
        
    def parse_metrics(self, prefix=''):
        """
        parse metrics to dict
        :return: metrics dict
        """
        value_dict = {}
        values = []
        if self.roc_auc_metric is not None:
            value_dict['ROCAUCMetric'] = self.roc_auc_metric.aggregate()
            self.roc_auc_metric.reset()

        for metric in self.metrics:
            value = metric.aggregate()

            if isinstance(value, list):
                values.extend([v.item() for v in value])
            else:
                values.append(value.item())
        value_dict.update(dict(zip(self.metrics_key, values)))

        for metric in self.metrics:
            metric.reset()

        return {prefix + key: value for key, value in value_dict.items()}