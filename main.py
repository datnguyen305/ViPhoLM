from builders.task_builder import build_task
from configs.utils import get_config
from argparse import ArgumentParser
import os
import numpy as np
import torch
import random

parser = ArgumentParser()
parser.add_argument("--config-file", type=str, required=True)
args = parser.parse_args()
config_file = args.config_file

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # (Tùy chọn) Ép PyTorch sử dụng các thuật toán deterministic hoàn toàn
    # torch.use_deterministic_algorithms(True)

if __name__ == "__main__":
    config = get_config(config_file)
    set_seed(config.training.seed)

    task = build_task(config)
    vocab = task.load_vocab(config.vocab)
    task.start()
    task.get_predictions()
    task.logger.info("Task done!")

