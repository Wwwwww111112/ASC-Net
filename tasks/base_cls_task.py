from monai.data import decollate_batch
from monai.utils.enums import PostFix

from medlab.registry import INFERERS, TASKS, TRANSFORMS

from .base_task import BaseTask


@TASKS.register_module()
class BaseClsTask(BaseTask):
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

    def forward(self, x):
        return self._model(x)

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
        outputs = self.val_inferer(inputs=images, network=self.forward)

        loss = self.loss_func(outputs, labels)

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        outputs = [self.post_pred_act(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        if self.roc_auc_metric is not None:
            self.roc_auc_metric(outputs, labels)

        outputs = [self.post_pred_cls(i) for i in outputs]

        if len(self.metrics) > 0:
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
        test step for classification task, save predictions to csv file
        :param batch: batch data
        :param batch_idx: batch index
        :return:
        """
        images = batch["images"]
        labels = batch["labels"]

        outputs = self.test_inferer(inputs=images, network=self.forward)

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        batch["preds"] = outputs

        outputs = [self.post_pred_act(i) for i in decollate_batch(outputs)]
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
        print(self.parse_metrics())

    def parse_metrics(self):
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

        return value_dict
