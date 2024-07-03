import copy
from typing import Any, Dict, Union

import lightning.pytorch as pl
from mmengine.config import Config, ConfigDict
from monai.data import DataLoader
from torch import nn

from medlab.registry import (DATASETS, LOSSES, LR_SCHEDULERS, METRICS, MODELS,
                             OPTIMIZERS)


class BaseTask(pl.LightningModule):
    def __init__(
            self,
            model: Union[nn.Module, Dict],
            train_dataloader: Union[DataLoader, Dict] = None,
            val_dataloader: Union[DataLoader, Dict] = None,
            test_dataloader: Union[DataLoader, Dict] = None,
            loss_func: Union[nn.Module, Dict] = None,
            optims: Any = None,
            metrics: Any = None,
            train_cfg: Dict = None,
            val_cfg: Dict = None,
            test_cfg: Dict = None,
            **kwargs
    ):
        """
        BaseTask
        :param model: nn.Module or model cfg
        :param train_dataloader: train dataloader or dataloader cfg
        :param val_dataloader: val dataloader or dataloader cfg
        :param test_dataloader: test dataloader or dataloader cfg
        :param loss_func: loss function or loss function cfg
        :param optims: PyTorch-lightning configure_optimizers
        :param metrics: metrics cfg
        :param train_cfg: train cfg
        :param val_cfg: val cfg
        :param test_cfg: test cfg
        :param kwargs: reserved parameters
        """
        super().__init__()

        self._model = self.build_model(model)
        self._train_dataloader = self.build_dataloader(train_dataloader)
        self._val_dataloader = self.build_dataloader(val_dataloader)
        self._test_dataloader = self.build_dataloader(test_dataloader)
        self.loss_func = self.build_loss_func(loss_func)
        self.optims = optims
        self.metrics_key, self.metrics = self.build_metric(metrics)
        self.train_cfg = train_cfg if train_cfg is not None else dict()
        self.val_cfg = val_cfg if val_cfg is not None else dict()
        self.test_cfg = test_cfg if test_cfg is not None else dict()

    def train_dataloader(self):
        """
        :return: train dataloader (PyTorch-lightning needs)
        """
        return self._train_dataloader

    def val_dataloader(self):
        """
        :return: val dataloader (PyTorch-lightning needs)
        """
        return self._val_dataloader

    def test_dataloader(self):
        """
        :return: test dataloader (PyTorch-lightning needs)
        """
        return self._test_dataloader

    @staticmethod
    def build_dataloader(dataloader: Union[DataLoader, Dict]) -> DataLoader:
        """
        build dataloader from cfg
        :param dataloader: dataloader or dataloader cfg
        :return: MONAI dataloader
        """
        if isinstance(dataloader, (DataLoader, type(None))):
            return dataloader

        dataloader_cfg = copy.deepcopy(dataloader)
        dataset_cfg = dataloader_cfg.pop('dataset')
        if isinstance(dataset_cfg, Dict):
            dataset = DATASETS.build(dataset_cfg)
        else:
            dataset = dataset_cfg

        data_loader = DataLoader(
            dataset=dataset,
            **dataloader_cfg
        )

        return data_loader

    @staticmethod
    def build_model(model: Union[nn.Module, Dict]) -> nn.Module:
        """
        build model from cfg
        :param model: nn.Module or model cfg
        :return: nn.Module
        """
        if isinstance(model, nn.Module):
            return model
        return MODELS.build(model)

    @staticmethod
    def build_loss_func(loss_func: Union[nn.Module, Dict]) -> nn.Module:
        """
        build loss function from cfg
        :param loss_func: nn.Module or loss cfg
        :return: nn.Module loss
        """
        if isinstance(loss_func, nn.Module):
            return loss_func

        loss_cfg = copy.deepcopy(loss_func)
        if loss_cfg.get('type', None) == 'DeepSupervisionLoss':
            loss = loss_cfg.get('loss')
            if isinstance(loss, (dict, ConfigDict, Config)):
                loss = LOSSES.build(loss)
                loss_cfg.update(dict(loss=loss))
        return LOSSES.build(loss_func)

    @staticmethod
    def build_optims(optims: Any, params: Any) -> Any:
        """
        build optimizer and lr schedules from cfg
        :param optims: optimizer and lr scheduler cfg
        :param params: model parameters to be optimized
        :return: optimizer and lr scheduler dict (PyTorch-lightning needs)
        """
        optims_cfg = copy.deepcopy(optims)
        optimizer = optims_cfg.get('optimizer', None)
        lr_scheduler = optims_cfg.get('lr_scheduler', None)
        assert optimizer is not None, 'optimizer must be provided'
        optimizer.update(dict(params=params))
        optimizer = OPTIMIZERS.build(optimizer)

        if lr_scheduler is not None:
            scheduler = lr_scheduler.pop('scheduler', None)
            scheduler.update(dict(optimizer=optimizer))
            scheduler = LR_SCHEDULERS.build(scheduler)
            lr_scheduler.update(dict(scheduler=scheduler))
            optims_cfg.update(dict(optimizer=optimizer, lr_scheduler=lr_scheduler))
            return optims_cfg
        return optimizer

    @staticmethod
    def build_metric(metrics: Any) -> Any:
        """
        build metrics from cfg
        :param metrics: metrics cfg
        :return: MONAI metrics
        """
        return METRICS.build(metrics)

    def configure_optimizers(self):
        """
        :return: optimizer and lr scheduler dict (PyTorch-lightning needs)
        """
        return self.build_optims(self.optims, self.parameters())
