import argparse
import logging
import os.path as osp
import warnings

from lightning.pytorch import seed_everything
from mmengine.config import Config

from medlab.registry import TASKS, TRAINERS


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('checkpoint', help='model checkpoint file path')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--save-dir', help='the dir to save logs and models')

    args = parser.parse_args()

    return args


def main():
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    args = parse_args()

    # load config
    if args.config is not None:
        cfg_path = args.config
    else:
        cfg_path = osp.join(osp.dirname(osp.dirname(args.checkpoint)), 'config.py')

    assert osp.exists(cfg_path)

    cfg = Config.fromfile(cfg_path)

    seed = cfg.get('seed', 1)
    ckpt_path = cfg.get('ckpt_path', None)
    trainer_cfg = cfg.get('trainer', dict())
    seed_everything(seed)

    if args.checkpoint is not None:
        ckpt_path = args.checkpoint

    if args.save_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        save_dir = args.save_dir
    else:
        save_dir = cfg.get('save_dir', osp.join(osp.dirname(ckpt_path), osp.splitext(osp.basename(ckpt_path))[0]))

    trainer_cfg['default_root_dir'] = save_dir

    trainer = TRAINERS.build(trainer_cfg)

    test_cfg = cfg.get('test_cfg', dict())
    test_cfg.update(dict(save_dir=save_dir))

    task = TASKS.build(
        dict(
            type=cfg.get('task_type', None),
            model=cfg.get('model', None),
            train_dataloader=None,
            val_dataloader=None,
            test_dataloader=cfg.get('test_dataloader', None),
            loss_func=cfg.get('loss_func', None),
            optims=None,
            metrics=cfg.get('metrics', None),
            train_cfg=dict(),
            val_cfg=dict(),
            test_cfg=test_cfg,
            num_classes=cfg.get('num_classes', None)
        )
    )
    trainer.test(task, ckpt_path=ckpt_path)



if __name__ == '__main__':
    main()
