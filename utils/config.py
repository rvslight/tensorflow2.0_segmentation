import os
import shutil
import yaml
from easydict import EasyDict as edict


def _get_default_config():
    c = edict()

    c.TRAIN_DIR = ''
    c.DATA_DIR = ''
    c.FOLD_DF = ''

    c.PARALLEL = False
    c.PRINT_EVERY = 1
    c.DEBUG = False

    c.TRAIN = edict()
    c.TRAIN.BATCH_SIZE = 1
    c.TRAIN.NUM_WORKERS = 1
    c.TRAIN.NUM_EPOCHS = 1

    c.EVAL = edict()
    c.EVAL.BATCH_SIZE = 1
    c.EVAL.NUM_WORKERS = 1

    c.DATA = edict()
    c.DATA.IMG_H = 512
    c.DATA.IMG_W = 288

    c.MODEL = edict()
    c.MODEL.NAME = ''
    c.MODEL.ARCHITECTURE = ''
    c.MODEL.ENCODER = ''
    c.MODEL.MULTIPLE = 1.0

    c.MODEL.PARAMS = edict()
    c.MODEL.PARAMS.resnet = ''
    c.MODEL.PARAMS.se_block = True

    c.LOSS = edict()
    c.LOSS.NAME = 'bce'
    c.LOSS.FINETUNE_EPOCH = 10000
    c.LOSS.FINETUNE_LOSS = ''
    c.LOSS.LABEL_SMOOTHING = False
    c.LOSS.PARAMS = edict()

    c.OPTIMIZER = edict()
    c.OPTIMIZER.NAME = 'adam'
    c.OPTIMIZER.LR = 0.001
    c.OPTIMIZER.PARAMS = edict()

    c.SCHEDULER = edict()
    c.SCHEDULER.NAME = 'none'
    c.SCHEDULER.PARAMS = edict()

    return c


def _merge_config(src, dst):
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load(config_path):
    with open(config_path, 'r') as fid:
        yaml_config = edict(yaml.load(fid))

    config = _get_default_config()
    _merge_config(yaml_config, config)

    return config


def save_config(config_path, train_dir):
    configs = [config for config in os.listdir(train_dir)
               if config.startswith('config') and config.endswith('.yml')]
    if not configs:
        last_num = -1
    else:
        last_config = list(sorted(configs))[-1]  # ex) config5.yml
        last_num = int(last_config.split('.')[0].split('config')[-1])

    save_name = 'config%02d.yml' % (int(last_num) + 1)

    shutil.copy(config_path, os.path.join(train_dir, save_name))