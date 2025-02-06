import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleModel(nn.Module):
    def __init__(self, models, softmax: bool = False):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.softmax = softmax

    def forward(self, x):
        output = None
        
        for m in self.models:
            logits = m(x)

            if self.softmax:
                logits = F.softmax(logits, dim=1)
            
            if output is None:
                output = logits
            else:
                output += logits
        
        output /= len(self.models)
        return output