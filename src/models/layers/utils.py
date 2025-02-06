import torch

def load_weights(model, wpath):
    state_dict = torch.load(wpath, map_location="cpu", weights_only=True)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"MISSING_KEYS: {missing_keys}")
    if unexpected_keys:
        print(f"UNEXPECTED_KEYS: {unexpected_keys}")
    
    return model