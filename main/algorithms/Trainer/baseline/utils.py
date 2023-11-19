import os
import torch
import random
import numpy as np
import datetime
import logging
import json
import copy


def get_config() -> dict:
    file_path = 'config.json'
    with open(file_path) as f:
        config = json.load(f)
    config['now'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config['logger'] = None
    return config


def get_agent(name):
    from agent import CoLight, EcoLight, FixedTime, FRAP, MaxPressure, MPLight, SOTL, TinyLight, TinyLightQuan, RandomPath
    name_2_class = {
        'CoLight': CoLight,
        'EcoLight': EcoLight,
        'FixedTime': FixedTime,
        'FRAP': FRAP,
        'MaxPressure': MaxPressure,
        'MPLight': MPLight,
        'SOTL': SOTL,
        'TinyLight': TinyLight,
        'TinyLightQuan': TinyLightQuan,
        'RandomPath': RandomPath,
    }
    return name_2_class[name]


def set_logger(config):
    config['log_path'] = 'log/{}/{}'.format(config['inter_name'], config['cur_agent'])
    if config['save_result']:
        if config['flow_idx'] == 0:
            with open('{}/config.json'.format(config['log_path']), 'w') as fout:
                json.dump(config, fout)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s -%(message)s'))
    logger.addHandler(sh)
    if config['save_result']:
        fh = logging.FileHandler(
            os.path.join(
                config['log_path'],
                '{}_{}_{}.log'.format(config['cur_agent'], config['flow_idx'], config['now'])
            )
        )
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s -%(message)s'))
        logger.addHandler(fh)
    config['logger'] = logger


def release_logger(config):
    if config['logger'] is not None:
        for handler in config['logger'].handlers[:]:
            config['logger'].removeHandler(handler)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def set_thread(num_thread=None):
    if num_thread is not None:
        torch.set_num_threads(num_thread)
    else:
        import platform
        if platform.system() == 'Linux':
            torch.set_num_threads(1)
        elif platform.system() == 'Darwin':
            torch.set_num_threads(1)


def copy_model_params(source_model, target_model):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(source_param.clone())


def _is_dumpable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def get_dumpable_config(config: dict):
    dumpable_config = {}
    for k, v in config.items():
        if _is_dumpable({k: v}):
            dumpable_config[k] = copy.deepcopy(v)
    return dumpable_config


def list_with_unique_element(original_list):
    new_list = []
    for elem in original_list:
        if elem not in new_list:
            new_list.append(elem)
    return new_list


def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        trimmed_dir_name = dir_name if not dir_name.endswith('/') else dir_name[:-1]
        os.rename(dir_name, "{}_rename_at_{}".format(trimmed_dir_name, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        os.makedirs(dir_name)
