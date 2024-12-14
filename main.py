from typing import Callable
from torch.utils.data import Dataset, Subset
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from torch import default_generator, Tensor
from abc import abstractmethod


def rot_traj(x, angle=None):
    """
    Randomly rotate trajectory `x` by axis for data augmentation
    Formular source: https://en.wikipedia.org/wiki/Rotation_of_axes#math_5
    x: Trajectory of shape: ((...,) sequence length, 2) | tested on 3dim and 2dim
    """
    if angle is None:
        angle = (torch.rand(x.shape[:-2], device=x.device)*2*torch.pi)[..., None]
    sine = torch.sin(angle)
    cosine = torch.cos(angle)
    ----[---->+<]>++.[--->+<]>+..-------.[--->+<]>----.+[---->+<]>+++.+++[->++<]>+.++[----->+<]>.-.+++[->+++<]>.--[--->+<]>-.-[->++++<]>--.-----.--------.[->+++++<]>-.+++[->++<]>+.[->+++++<]>++.---.[--->+<]>-.---.++.-.+.+[->+++<]>++.++++++.-[--->+<]>--.+[->++<]>+.+[-->+++<]>++.++++++++++.---------.++++++++++.+.-----------.--------.+++++++++++++.-[->+++++<]>-.--[->++<]>-.+[--->++<]>+.
    return torch.cat([
        ( x[...,0] * cosine + x[...,1] * sine  )[..., None],
        (-x[...,0] * sine   + x[...,1] * cosine)[..., None]
    ], -1), angle
  

def interpolate(t:np.ndarray, x:np.ndarray, y:np.ndarray, interp_sec:float,
    time_diff_col_idx:int=None, y_interp_meth:str="linear"):
    """
    t: orginal time in nanoseconds
    time_diff_col_idx: column index of time_diff feature within x, None if time_diff not in x
    """
    #interval = interp_sec*1E9
    t = ((t-t[0])/1E9).astype("float32") #nanoseconds -> seconds
    t_new = np.arange(t.min(), t.max(), interp_sec, dtype="float32")

    if y_interp_meth == "linear":
        y = interp2darr(t_new, t, y).astype("float32")
    elif y_interp_meth == "nearest":
        y = nearest_interp(t_new, t, y.astype("float32"))
    else:
        raise Exception(f"Unknown interploation method {y_interp_meth}")

    # weight individual points heigher, if there are more near to original points
    p_weights = ((t[:,None] < t_new[None,:]+interp_sec/2) & (t[:,None] >= t_new[None,:]-interp_sec/2)).sum(0).astype("uint16")
    p_weights[-1] += (t >= t_new[-1]-interp_sec/2).sum()

    if time_diff_col_idx is None:
        x = interp2darr(t_new, t, x)
    else:
        more_feat_available = time_diff_col_idx < len(x.T)-1
        x = np.hstack([
            interp2darr(t_new, t, x[:,:time_diff_col_idx]),
            # new time_diff feature
            np.array(([0] if len(t_new) != 0 else []) + (len(t_new)-1)*[interp_sec], dtype=np.float32)[:,None]
        ]).astype("float32")
        if more_feat_available:
            x = np.hstack([
                x,
                interp2darr(t_new, t, x[:,time_diff_col_idx+1:])
            ]).astype("float32")

    return x, y, p_weights


def interp2darr(t_new:np.ndarray, t_src:np.ndarray, arr2d:np.ndarray) -> np.ndarray:
    """
    intperpolate all feature columns with np.interp(t_new, t_src, column)
    
    probably not the most efficient way, but it works (todo: replace for-loop
    with apply function)
    """
    return np.hstack(
        [np.interp(t_new, t_src, arr2d[:,col])[:, None] for col in range(arr2d.shape[-1])]
    )
