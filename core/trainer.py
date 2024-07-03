import copy
import os.path as osp
import time
from typing import Dict, Union

from lightning.pytorch.trainer import Trainer as LightningTrainer
from mmengine.config import Config, ConfigDict

from medlab.registry import CALLBACKS, LOGGERS, TRAINERS

ConfigType = Union[Dict, Config, ConfigDict]


@TRAINERS.register_module()
class Trainer(LightningTrainer):
    def __init__(self, *args, **kwargs):
        """
        build loggers and callbacks, get time prefix, init lightning trainer
        :param args: same as lightning trainer
        :param kwargs: same as lightning trainer
        """
        time_prefix = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        logger = kwargs.get('logger', None)
        callbacks = kwargs.get('callbacks', None)
        if logger is not None:
            logger = self.build_loggers(logger, kwargs.get('default_root_dir', None), time_prefix)
        if callbacks is not None:
            callbacks = self.build_callbacks(kwargs.get('callbacks', None), kwargs.get('default_root_dir', None),
                                             time_prefix)
        kwargs.update(dict(logger=logger, callbacks=callbacks))
        super().__init__(*args, **kwargs)
        self._time_prefix = time_prefix

    @staticmethod
    def build_loggers(logger, work_dir, name):
        """
        build logger(s) from cfg
        :param logger: logger cfg
        :param work_dir: logger save dir
        :param name: logger name
        :return: lightning logger(s)
        """
        loggers_cfg = copy.deepcopy(logger)

        if isinstance(loggers_cfg, list):
            for i in loggers_cfg:
                i.update(dict(save_dir=work_dir, name=name))
        else:
            loggers_cfg.update(dict(save_dir=work_dir, name=name))

        return LOGGERS.build(loggers_cfg)

    @staticmethod
    def build_callbacks(callbacks, work_dir, name):
        """
        build callback(s) from cfg
        :param callbacks: callback cfg
        :param work_dir: if ModelCheckpoint in callbacks , save dir is work_dir/name/checkpoints
        :param name: checkpoint dir name
        :return:
        """
        callbacks_cfg = copy.deepcopy(callbacks)
        if isinstance(callbacks_cfg, list):
            for i in callbacks_cfg:
                if i.get('type', None) == 'ModelCheckpoint':
                    i.update(dict(dirpath=osp.join(work_dir, name, 'checkpoints')))
        else:
            if callbacks_cfg.get('type', None) == 'ModelCheckpoint':
                callbacks_cfg.update(dict(dirpath=osp.join(work_dir, name, 'checkpoints')))
        return CALLBACKS.build(callbacks_cfg)

    @classmethod
    def from_cfg(cls, cfg: ConfigType):
        """
        build trainer from cfg
        :param cfg:
        :return: trainer
        """
        cfg = copy.deepcopy(cfg)
        trainer = cls(
            accelerator=cfg.get('accelerator', 'auto'),
            strategy=cfg.get('strategy', 'auto'),
            devices=cfg.get('devices', 'auto'),
            num_nodes=cfg.get('num_nodes', 1),
            precision=cfg.get('precision', "32-true"),
            logger=cfg.get('logger', None),
            callbacks=cfg.get('callbacks', None),
            fast_dev_run=cfg.get('fast_dev_run', False),
            max_epochs=cfg.get('max_epochs', None),
            min_epochs=cfg.get('min_epochs', None),
            max_steps=cfg.get('max_steps', -1),
            min_steps=cfg.get('min_steps', None),
            max_time=cfg.get('max_time', None),
            limit_train_batches=cfg.get('limit_train_batches', None),
            limit_val_batches=cfg.get('limit_val_batches', None),
            limit_test_batches=cfg.get('limit_test_batches', None),
            limit_predict_batches=cfg.get('limit_predict_batches', None),
            overfit_batches=cfg.get('overfit_batches', 0.0),
            val_check_interval=cfg.get('val_check_interval', None),
            check_val_every_n_epoch=cfg.get('check_val_every_n_epoch', 1),
            num_sanity_val_steps=cfg.get('num_sanity_val_steps', None),
            log_every_n_steps=cfg.get('log_every_n_steps', None),
            enable_checkpointing=cfg.get('enable_checkpointing', None),
            enable_progress_bar=cfg.get('enable_progress_bar', None),
            enable_model_summary=cfg.get('enable_model_summary', None),
            accumulate_grad_batches=cfg.get('accumulate_grad_batches', 1),
            gradient_clip_val=cfg.get('gradient_clip_val', None),
            gradient_clip_algorithm=cfg.get('gradient_clip_algorithm', None),
            deterministic=cfg.get('deterministic', None),
            benchmark=cfg.get('benchmark', None),
            inference_mode=cfg.get('inference_mode', True),
            use_distributed_sampler=cfg.get('use_distributed_sampler', True),
            profiler=cfg.get('profiler', None),
            detect_anomaly=cfg.get('detect_anomaly', False),
            barebones=cfg.get('barebones', False),
            plugins=cfg.get('plugins', None),
            sync_batchnorm=cfg.get('sync_batchnorm', False),
            reload_dataloaders_every_n_epochs=cfg.get('reload_dataloaders_every_n_epochs', 0),
            default_root_dir=cfg.get('default_root_dir', None)
        )
        return trainer

    @property
    def time_prefix(self):
        """
        get time prefix
        :return: string time prefix
        """
        return self._time_prefix
