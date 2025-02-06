from copy import deepcopy
from importlib import import_module

from types import SimpleNamespace

import torch
import torch.nn as nn

class ModelEMA(nn.Module):
    """
    EMA for model weights.
    Source: https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/475080#2641635
    
    Ex.
    def training_step(self, batch, batch_idx):
        self.ema_model.update(self.model)
        ...
    """
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def get_model(cfg: SimpleNamespace, inference_mode: bool = False):

    # Build
    m = import_module(f"src.models.{cfg.model_type}").Net(
        cfg=cfg, 
        inference_mode=inference_mode,
        )

    # Param count
    n_params= count_parameters(m)
    print(f"Model: {cfg.model_type}")
    print("n_param: {:_}".format(n_params))

    # Load weights
    f= cfg.weights_path
    if f != "":
        
        # Skip loading loss_fn params
        checkpoint = torch.load(f, map_location=cfg.device, weights_only=True)
        filtered_checkpoint = {}
        for k, v in checkpoint.items():
            if not k.startswith('loss_fn'):
                filtered_checkpoint[k] = v
        m.load_state_dict(filtered_checkpoint, strict=False)

        # Log missing params
        missing_params= [k for k in m.state_dict() if k not in filtered_checkpoint]
        no_params= [k for k in filtered_checkpoint if k not in m.state_dict()]
        if len(missing_params) > 0:
            print("MISSING_PARAMS:", missing_params)
        if len(no_params) > 0:
            print("NO_PARAMS:", no_params)
        print("LOADED WEIGHTS:", f)

    return m, n_params

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)