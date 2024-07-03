import argparse
import logging
import os
import os.path as osp
import warnings

from lightning.pytorch import seed_everything
from mmengine.config import Config

from medlab.registry import TASKS, TRAINERS


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', help='model checkpoint file path', default=None)
    parser.add_argument('--work-dir', help='the dir to save logs and models')

    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')

    args = parser.parse_args()

    return args


def main():
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    seed = cfg.get('seed', 1)
    ckpt_path = cfg.get('ckpt_path', None)
    trainer_cfg = cfg.get('trainer', dict())
    seed_everything(seed)

    if args.checkpoint is not None:
        ckpt_path = args.checkpoint

    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        work_dir = args.work_dir
    elif trainer_cfg.get('default_root_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    if args.amp is True:
        trainer_cfg.update(dict(precision='16'))

    trainer_cfg['default_root_dir'] = work_dir
    trainer = TRAINERS.build(trainer_cfg)

    task = TASKS.build(
        dict(
            type=cfg.get('task_type', None),
            model=cfg.get('model', None),
            train_dataloader=cfg.get('train_dataloader', None),
            val_dataloader=cfg.get('val_dataloader', None),
            test_dataloader=None,
            loss_func=cfg.get('loss_func', None),
            optims=cfg.get('optims', None),
            metrics=cfg.get('metrics', None),
            train_cfg=cfg.get('train_cfg', dict()),
            val_cfg=cfg.get('val_cfg', dict()),
            test_cfg=dict(),
            num_classes=cfg.get('num_classes', None)
        )
    )
    os.makedirs(osp.join(work_dir, trainer.time_prefix), exist_ok=True)
    cfg.dump(osp.join(work_dir, trainer.time_prefix, 'config.py'))
    trainer.fit(task, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
