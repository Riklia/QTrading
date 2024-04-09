import numpy as np
import random
import torch
import matplotlib


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


matplotlib.use('agg')
set_seed()
