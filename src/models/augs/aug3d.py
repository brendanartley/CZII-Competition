import torch
import torch.nn.functional as F
import random

def rotate(x, mask= None, dims= ((-3,-2), (-3,-1), (-2,-1)), p= 1.0):
    """
    Rotate pixels.

    Same rotate for each sample in batch is 
    used for speed. This reduces batch 
    diversity.
    """
    bs= x.shape[0]
    for d in dims:
        if random.random() < p:
            k = random.randint(0,3)
            x = torch.rot90(x, k=k, dims=d)
            if mask is not None:
                mask = torch.rot90(mask, k=k, dims=d) 

    if mask is not None:
        return x, mask
    else:
        return x

def pixel_aug(x, mask= None, scale_range= (0.90, 1.25), shift_range= (-0.05, 0.05)):
    """
    Scale and shift pixels.
    """
    for idx in range(x.shape[0]):
        scale = random.uniform(*scale_range)
        shift = random.uniform(*shift_range)
        x[idx] = torch.clamp(x[idx] * scale + shift, -10.0, 10.0)

    if mask is not None:
        return x, mask
    else:
        return x

def flip_3d(x, mask= None, dims=(-3,-2,-1), p= 0.5):
    """
    Flip along axis.
    """
    axes = [i for i in dims if random.random() < p]
    if axes:
        x = torch.flip(x, dims=axes)
        if mask is not None:
            mask = torch.flip(mask, dims=axes)
        
    if mask is not None:
        return x, mask
    else:
        return x

def swap_dims(x, mask= None, p= 0.5, dims=(-2,-1)):
    """
    Randomly swap dims.
    """
    if random.random() < p:
        swap_dims= list(dims)
        random.shuffle(swap_dims)
        x = x.transpose(*swap_dims)
        if mask is not None:
            mask = mask.transpose(*swap_dims)

    if mask is not None:
        return x, mask
    else:
        return x

def cutmix_3d(x, mask= None, p= 1.0, skipz= False):
    """
    Cutmix.
    """

    # Shuffle
    x_mixed = x.roll(1, dims=0)
    if mask is not None:
        mask_mixed = mask.roll(1, dims=0)

    # Shapes
    pb, pc, pz, py, px= x.shape

    for idx in range(pb):
        prob= random.random()
        if prob < p:

            # Get bbox size
            if skipz: 
                z_size= pz
            else: 
                z_size= int(random.uniform(0.0, 1.0) * pz)
            y_size= int(random.uniform(0.0, 1.0) * py)
            x_size= int(random.uniform(0.0, 1.0) * px)

            # Get bbox positions
            z_start = random.randint(0, pz - z_size)
            y_start = random.randint(0, py - y_size)
            x_start = random.randint(0, px - x_size)
            z_end= z_start + z_size
            y_end= y_start + y_size
            x_end= x_start + x_size

            # Apply to box
            x[idx, :, z_start:z_end, y_start:y_end, x_start:x_end] = \
            x_mixed[idx, :, z_start:z_end, y_start:y_end, x_start:x_end]

            if mask is not None:
                mask[idx, :, z_start:z_end, y_start:y_end, x_start:x_end] = \
                mask_mixed[idx, :, z_start:z_end, y_start:y_end, x_start:x_end]

    if mask is not None:
        return x, mask
    else:
        return x


def mixup_3d(x, mask=None, p=1.0, alpha=10.0):
    """
    Mixup.
    """
    # Shuffle
    x_mixed = x.roll(1, dims=0)
    if mask is not None:
        mask_mixed = mask.roll(1, dims=0)

    # Proba
    lam = torch.distributions.Beta(alpha, alpha).sample()
    apply_mixup = (torch.rand(x.shape[0]) < p).to(x.device)

    # Mix
    x[apply_mixup] = lam * x[apply_mixup] + (1 - lam) * x_mixed[apply_mixup]
    if mask is not None:
        mask[apply_mixup] = lam * mask[apply_mixup] + (1 - lam) * mask_mixed[apply_mixup]
        return x, mask
    else:
        return x
        

if __name__ == "__main__":

    x= torch.ones(1,1,32,128,128)
    mask= torch.ones(1,6,32,128,128)
    x,mask= mixup_3d(x, mask)
    print(x.shape, mask.shape)