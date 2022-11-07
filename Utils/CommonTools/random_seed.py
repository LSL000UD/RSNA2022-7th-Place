import torch
import random
import numpy as np


def set_random_seed(seed):
    import torch.backends.cudnn as cudnn
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
