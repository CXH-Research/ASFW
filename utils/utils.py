import os
import random
from collections import OrderedDict

import numpy as np
import torch


def seed_everything(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, epoch, model_name, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    checkpoint_file = os.path.join(outdir, model_name + '_' + epoch + '.pth')
    torch.save(state, checkpoint_file)


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location=lambda storage, loc: storage.cuda(0))
    new_state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('module'):
            name = key[7:]
        else:
            name = key
        new_state_dict[name] = value
    model.load_state_dict(new_state_dict)


def mask_generator(shadow: torch.Tensor, shadow_free: torch.Tensor, grad: bool = False) -> torch.Tensor:
    # Convert to grayscale using RGB weights
    rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).to(shadow.device)
    
    # Convert to grayscale
    gray_shadow = (shadow * rgb_weights).sum(dim=1, keepdim=True)
    gray_shadow_free = (shadow_free * rgb_weights).sum(dim=1, keepdim=True)
    
    # Calculate difference
    diff = gray_shadow_free - gray_shadow
    
    # Implement Otsu's method in PyTorch
    hist = torch.histc(diff, bins=256, min=diff.min(), max=diff.max()).to(shadow.device)
    bin_edges = torch.linspace(diff.min(), diff.max(), 257).to(shadow.device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate probabilities
    total = hist.sum()
    w = torch.zeros_like(hist)
    mean = torch.zeros_like(hist)
    
    # Cumulative sum of probabilities
    cumsum = torch.cumsum(hist, dim=0)
    w = cumsum / total
    
    # Cumulative mean
    cumsum_mean = torch.cumsum(hist * bin_centers, dim=0)
    mean = cumsum_mean / (cumsum + 1e-10)
    
    # Global mean
    global_mean = cumsum_mean[-1] / total
    
    # Calculate between-class variance
    between_var = w * (1 - w) * (mean - global_mean) ** 2
    
    # Find threshold that maximizes between-class variance
    thresh_idx = torch.argmax(between_var)
    threshold = bin_centers[thresh_idx]
    
    # Generate mask
    mask = (diff >= threshold).float()
    
    if grad:
        mask.requires_grad = True
        
    return mask