import torch

import cps_config as C


def get_device():
    if C.DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(C.DEVICE)
