import os.path as osp
from typing import Dict, OrderedDict

from monai.data import decollate_batch

from medlab.registry import INFERERS, TASKS, TRANSFORMS

from .base_task import BaseTask


@TASKS.register_module()
class BaseSegTask(BaseTask):
    def __init__(self, *args, **kwargs):
        """
        base segmentation task, one input, one output
        """
        super().__init__(*args, **kwargs)
        self.val_inferer = INFERERS.build(self.val_cfg.get('inferer', dict(type='SimpleInferer')))
        self.test_inferer = INFERERS.build(self.test_cfg.get('inferer', dict(type='SimpleInferer')))
        num_classes = kwargs.get('num_classes', None)
        assert num_classes is not None, "num_classes must be specified in model"
        if num_classes == 1:
            self.post_pred = TRANSFORMS.build([
                dict(type='Activations', sigmoid=True),
                dict(type='AsDiscrete', threshold=0.5)
            ])
            self.post_label = TRANSFORMS.build([])
            post_save = [
                dict(type='ToDevice', device='cpu'),
            ]
        else:
            self.post_pred = TRANSFORMS.build([dict(type='AsDiscrete', argmax=True, to_onehot=num_classes)])
            self.post_label = TRANSFORMS.build([dict(type='AsDiscrete', to_onehot=num_classes)])
            post_save = [
                dict(type='AsDiscrete', argmax=True),
                dict(type='ToDevice', device='cpu')
            ]
        save_dir = self.test_cfg.get('save_dir', None)
        if save_dir is not None:
            post_save.append(
                dict(
                    type='SaveImage',
                    output_dir=osp.join(self.test_cfg.get('save_dir', None), 'predict_result'),
                    resample=True,
                    output_postfix='pred',
                    output_ext=self.test_cfg.get('output_ext', '.png'),
                    scale=self.test_cfg.get('scale', 1.0),
                    squeeze_end_dims=True,
                    separate_folder=False
                )
            )
        self.post_save = TRANSFORMS.build(post_save)

    def forward(self, x):
        return self.parse_outputs(self._model(x))

    def training_step(self, batch, batch_idx):
        """
        training step for classification task
        :param batch: batch data
        :param batch_idx: batch index
        :return: loss
        """
        images = batch['images']
        labels = batch['labels']

        batch_size = images.shape[0]
        outputs = self.forward(images)
        loss = self.loss_func(outputs, labels)

        self.log("train_step_loss", loss, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        validation step for classification task
        :param batch: batch data
        :param batch_idx: batch index
        :return: None
        """
        images = batch['images']
        labels = batch['labels']

        batch_size = images.shape[0]
        # outputs = self.forward(images)
        outputs = self.val_inferer(inputs=images, network=self.forward)

        loss = self.loss_func(outputs, labels)

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        for metric in self.metrics:
            metric(outputs, labels)

        self.log('val_step_loss', loss.item(), sync_dist=True, batch_size=batch_size)

    def on_validation_epoch_end(self):
        """
        validation epoch end hook, parse and log metrics
        """
        self.log_dict(self.parse_metrics(), sync_dist=True)

    def test_step(self, batch, batch_idx):
        """
        test step for classification task, save predictions to disk
        :param batch: batch data
        :param batch_idx: batch index
        :return:
        """
        images = batch["images"]
        labels = batch["labels"]

        outputs = self.test_inferer(inputs=images, network=self.forward)

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        for metric in self.metrics:
            metric(outputs, labels)

        outputs[0].meta['filename_or_obj'] = batch['images_meta_dict']['filename_or_obj'][0]

        self.post_save(outputs[0])

    def on_test_epoch_end(self):
        """
        test epoch end hook, parse metrics
        """
        print(self.parse_metrics())

    def parse_metrics(self):
        """
        parse metrics to dict
        :return: metrics dict
        """
        values = []
        for metric in self.metrics:
            value = metric.aggregate()

            if isinstance(value, list):
                values.extend([v.item() for v in value])
            else:
                values.append(value.item())

        value_dict = dict(zip(self.metrics_key, values))
        for metric in self.metrics:
            metric.reset()

        return value_dict

    @staticmethod
    def parse_outputs(outputs):
        """
        parse outputs to tensor or list
        :return: outputs dict
        """
        if isinstance(outputs, (OrderedDict, Dict)):
            outputs = list(outputs.values())
            if len(outputs) == 1:
                outputs = outputs[0]
        return outputs
