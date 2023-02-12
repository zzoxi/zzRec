from .trainer import *
from .trainer_cml import *
from config.configurator import configs

def build_trainer(data_handler, logger):
    if 'trainer' not in configs['train']:
        trainer = Trainer(data_handler, logger)
    elif configs['train']['trainer'] == 'cml_trainer':
        trainer = CMLTrainer(data_handler, logger)
    else:
        raise NotImplementedError
    return trainer