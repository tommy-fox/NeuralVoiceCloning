try:
    from monotonic_align.core import maximum_path_c as _maximum_path_c
    _HAS_CYTHON_MA = True
except ImportError:
    _HAS_CYTHON_MA = False

import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
from munch import Munch


def maximum_path(neg_cent, mask):
  """ Cython optimized version (falls back to numpy if Cython unavailable).
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent = np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path = np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))

  if _HAS_CYTHON_MA:
      _maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  else:
      # Pure-numpy fallback (slower but works everywhere)
      for b in range(path.shape[0]):
          t_t = t_t_max[b]
          t_s = t_s_max[b]
          # Viterbi-style forward pass
          Q = np.full((t_t, t_s), fill_value=-np.inf, dtype=np.float32)
          Q[0, 0] = neg_cent[b, 0, 0]
          for j in range(1, t_s):
              Q[0, j] = Q[0, j - 1] + neg_cent[b, 0, j]
          for i in range(1, t_t):
              Q[i, i] = Q[i - 1, i - 1] + neg_cent[b, i, i]
              for j in range(i + 1, t_s):
                  Q[i, j] = max(Q[i - 1, j - 1], Q[i, j - 1]) + neg_cent[b, i, j]
          # Backtrace
          i, j = t_t - 1, t_s - 1
          path[b, i, j] = 1
          while i > 0 or j > 0:
              if i == 0:
                  j -= 1
              elif j == 0:
                  i -= 1
              elif Q[i - 1, j - 1] >= Q[i, j - 1]:
                  i -= 1
                  j -= 1
              else:
                  j -= 1
              path[b, i, j] = 1

  return torch.from_numpy(path).to(device=device, dtype=dtype)


def mask_from_lens(attn, input_lengths, mel_lengths):
    """Build a binary mask for monotonic alignment from input/mel lengths."""
    bs = attn.shape[0]
    t_t = attn.shape[1]
    t_s = attn.shape[2]
    mask = torch.zeros(bs, t_t, t_s, dtype=attn.dtype, device=attn.device)
    for b in range(bs):
        mask[b, :input_lengths[b], :mel_lengths[b]] = 1
    return mask

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths, max_len=None):
    if max_len is None:
        max_len = lengths.max()
    mask = torch.arange(max_len).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)
    