import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pynvml
import random


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def search_free_cuda():
    pynvml.nvmlInit()
    id = 2
    for i in range(4):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if mem_info.used == 0:
            id = i
            break
    os.environ["CUDA_VISIBLE_DEVICES"] = str(id)


def cudafy(module):
    if torch.cuda.is_available():
        return module.cuda(), True
    else:
        return module.cpu(), False


def cuda_avaliable():
    if torch.cuda.is_available():
        return True, torch.device("cuda")
    else:
        return False, torch.device("cpu")


def show_parameters(model: nn.Module, if_show_parameters=False):
    for name, parameters in model.named_parameters():
        if parameters.requires_grad == False:
            continue
        print("name:{} ; size:{} ".format(name, parameters.shape))
        if if_show_parameters:
            print("parameters:", parameters)


def count_parameters(model: nn.Module):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def shuffle2list(a: list, b: list):
    # shuffle two list with same rule, you can also use sklearn.utils.shuffle package
    c = list(zip(a, b))
    random.shuffle(c)
    a[:], b[:] = zip(*c)
    return a, b


def gather(param, ids):
    # Take the line corresponding to IDS subscript from param and form a new tensor
    if param.is_cuda:
        mask = F.one_hot(ids, num_classes=param.shape[0]).float().cuda()
    else:
        mask = F.one_hot(ids, num_classes=param.shape[0]).float()
    ans = torch.mm(mask, param)
    return ans


if __name__ == "__main__":
    # p = torch.rand(3, 3)
    # ids = torch.from_numpy(np.arange(3))
    # ans = gather(p, ids)
    # print("p:", p)
    # print("ans", ans)
    centroid_ids = torch.tensor([0, 2, 3])
    pd = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], [2, 3, 7, 6, 4], [2, 9, 7, 0, 4], [1, 8, 2, 3, 0]])
    for i in range(5):
        pd[i][i] = 0
    ans = gather(pd, centroid_ids)
    print("ans:", ans)
    w = torch.where(ans == 0)
    print(w)
