from typing import Iterable, Tuple
import os
import pickle
from tqdm import tqdm
from types import SimpleNamespace

import pandas as pd
import numpy as np
import zarr

import torch

import kornia as K

from src.models.augs import aug3d

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode

        self.foldmap= {
            -1: [],
            0: ["TS_5_4"],
            1: ["TS_69_2"],
            2: ["TS_6_4"],
            3: ["TS_6_6"],
            4: ["TS_73_6"],
            5: ["TS_86_3"],
            6: ["TS_99_9"],
            100: ["TS_5_4", "TS_69_2", "TS_6_4", "TS_6_6", "TS_73_6", "TS_86_3", "TS_99_9"],
        }

        self.class2label= {
            'apo-ferritin': 1,
            'beta-galactosidase': 2,
            'ribosome': 3,
            'thyroglobulin': 4,
            'virus-like-particle': 5,
            'beta-amylase': 6,
        }

        # NOTE: Not exact diameter. Used for cut/paste augmentation.
        self.class2diameter= {
            'apo-ferritin': 24,
            'beta-galactosidase': 30,
            'ribosome': 44,
            'thyroglobulin': 38,
            'virus-like-particle': 38,
            'beta-amylase': 26,
        }

        self.data, self.labels, self.idxs = self.load_metadata()
        
        # Kornia rotation params
        self.center= torch.tensor([self.cfg.patch_size[-2]/2, self.cfg.patch_size[-1]/2]).half().to(self.cfg.device)
        self.scale= torch.tensor([1.0, 1.0]).half().to(self.cfg.device)

    def _scale_tomo(self, x):
        x = x*30_000
        x = np.clip(x, -5, 5)
        return x

    def load_metadata(self, ):

        # Load datasets
        if self.mode == "test":
            data = sorted(os.listdir(self.cfg.test_fpath))
            labels= []
            idxs= data
            return data, labels, idxs
        else:
            from copick.impl.filesystem import CopickRootFSSpec
            roots= []

            # Simulated
            label_df= None
            particle_dict= {}
            if self.mode == "train" and self.cfg.pretrain == True:
                fpath= "./src/pre/copick_overlay_sim.json"
                root= CopickRootFSSpec.from_file(fpath)
                roots.append({
                    "root": root,
                    "filter": "wbp",
                    "scale_val": 1.5,
                    "clip_val": 5,
                })

            # Normal
            if self.cfg.pretrain == False:
                fpath= "./src/pre/copick_overlay.json"
                root= CopickRootFSSpec.from_file(fpath)
                roots.append({
                    "root": root,
                    "filter": "denoised",
                    "scale_val": 30_000,
                    "clip_val": 5,
                })

                label_df= pd.read_csv("./data/raw/labels.csv")

        # Load runs
        data= []
        labels= []
        idxs= []
        for row in roots:
            root= row["root"]
            filter_type= row["filter"]

            # Select fold
            if self.mode == "train":
                root._runs= [r for r in root.runs if r.name not in self.foldmap[self.cfg.fold]]
            elif self.mode == "val":
                root._runs= [r for r in root.runs if r.name in self.foldmap[self.cfg.fold]]
            else:
                raise ValueError()
            idxs += [_.name for _ in root._runs]

            # Load data
            for run in tqdm(root.runs):
                tomo = run.get_voxel_spacing(10) \
                    .get_tomogram(filter_type) \
                    .numpy()
                    
                # Load fpath
                fpath= "./data/processed/overlay/ExperimentRuns/{}/{}_segmentation.npy".format(
                    run.name, run.name,
                    )
                if not os.path.exists(fpath):
                    fpath= fpath.replace("processed", "processed_sim")
                label = np.load(fpath)

                # Scale
                tomo= self._scale_tomo(tomo)
                tomo= np.expand_dims(tomo, axis=0)

                data.append(tomo)
                labels.append(label)

                # Save particles for cut-paste augmentation
                if label_df is not None:
                    particle_dict[run.name]= {k:[] for k in self.class2label.keys()}
                    for _, row in label_df[label_df.experiment == run.name].iterrows():

                        particle_type= row["particle_type"]
                        diameter= self.class2diameter[particle_type]
                        radius= diameter // 2

                        z= int((row["z"] / 10.012444) + 0.5)
                        y= int((row["y"] / 10.012444) + 0.5)
                        x= int((row["x"] / 10.012444) + 0.5)

                        z_start, z_end= z-radius, z+radius
                        y_start, y_end= y-radius, y+radius
                        x_start, x_end= x-radius, x+radius
                        
                        # Check within bounds
                        if z_start > 0 and y_start > 0 and x_start > 0 and z_end < 184 - radius and y_end < 630 - radius and x_end < 630 - radius:

                            # Select particle
                            tmp_tomo= tomo[:, z_start:z_end, y_start:y_end, x_start:x_end]
                            tmp_label= label[:, z_start:z_end, y_start:y_end, x_start:x_end]

                            # Move to GPU
                            tmp_tomo= torch.from_numpy(tmp_tomo).to(self.cfg.device, dtype=torch.float16)
                            tmp_label= torch.from_numpy(tmp_label).to(self.cfg.device, dtype=torch.float16)

                            particle_dict[run.name][particle_type].append(
                                (tmp_tomo, tmp_label)
                            )        
        self.particle_dict= particle_dict

        # Move to GPU
        data = [torch.from_numpy(_).to(self.cfg.device, dtype=torch.float16) for _ in data]
        labels = [torch.from_numpy(_).to(self.cfg.device, dtype=torch.float16) for _ in labels]
        if self.mode != "train":
            print("val_idxs:", idxs)

        return data, labels, idxs

    def rotate_xy_3d(self, img, mask=None):
        """
        Hacky way to rotate on GPU over
        the last 2 axes of a volume.
        """
        _, T, H, W = img.shape  
        C, _, _, _ = mask.shape
        angle=(torch.rand(1) * 360).half().to(self.cfg.device)

        # Rotate img
        M = K.geometry.transform.get_rotation_matrix2d(
            center= self.center.repeat(T, 1), 
            angle= angle.repeat(T), 
            scale= self.scale.repeat(T, 1),
            )  
        img= K.geometry.transform.warp_affine(
            src = img.view(T, 1, H, W), 
            M = M, 
            dsize = (H, W),
            padding_mode = "zeros",
            mode = "bilinear"
        )    
        img= img.view(1, T, H, W)

        # Rotate mask
        if mask is not None:
            M = K.geometry.transform.get_rotation_matrix2d(
                center= self.center.repeat(C*T, 1), 
                angle= angle.repeat(C*T), 
                scale= self.scale.repeat(C*T, 1),
            )  

            mask= K.geometry.transform.warp_affine(
                src = mask.reshape(C*T, 1, H, W), 
                M = M, 
                dsize = (H, W),
                padding_mode = "zeros",
                mode = "nearest",
            )    
            mask= mask.view(C, T, H, W)
            return img, mask
        else:
            return img

    def __getitem__(self, idx):
        # Return
        if self.mode == "train":
            img_idx = idx % len(self.data)
            pz, py, px= self.cfg.patch_size

            # ========== Select patch ==========
            # Crop larger size so rotation has no edge artifacts
            extra_pad = 32
            py = py + extra_pad
            px = px + extra_pad

            z= np.random.randint(0, 184 - pz)
            y= np.random.randint(0, 630 - py)
            x= np.random.randint(0, 630 - px)

            x_out= self.data[img_idx][:, z:z+pz, y:y+py, x:x+px]
            y_out= self.labels[img_idx][:, z:z+pz, y:y+py, x:x+px]
            all_background= torch.all(y_out[1:, ...] < 0.5)

            # ========== Cutpaste Augmentation ==========
            if self.cfg.pretrain:
                pass
            elif all_background and np.random.random() < self.cfg.cutpaste_prob:  

                # Select particle
                experiment= self.idxs[img_idx]
                particle_type= np.random.choice(
                    a= ['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle', 'beta-amylase'],
                    )
                tmp_idx= np.random.randint(0, len(self.particle_dict[experiment][particle_type]))
                tmp_tomo, tmp_label= self.particle_dict[experiment][particle_type][tmp_idx]
                _, tz, ty, tx= tmp_tomo.shape
                
                # Trim if particle larger than patch size
                if tz > pz:
                    tz_start= np.random.randint(0, tz - pz)
                    tz_end= tz_start + pz
                else:
                    tz_start= 0
                    tz_end= tz

                # Augment particle
                tmp_tomo, tmp_label= aug3d.rotate(tmp_tomo, tmp_label, p= 1.0, dims=[(-2,-1)])
                tmp_tomo, tmp_label= aug3d.flip_3d(tmp_tomo, tmp_label)
                tmp_tomo, tmp_label= aug3d.swap_dims(tmp_tomo, tmp_label)

                # Get position
                z_start= np.random.randint(0, pz - tz) if tz < pz else 0
                z_end= z_start + tz
                y_start= np.random.randint(0, py - ty)
                y_end= y_start + ty
                x_start= np.random.randint(0, px - tx)
                x_end= x_start + tx

                # Insert
                x_out[:, z_start:z_end, y_start:y_end, x_start:x_end]= tmp_tomo[:, tz_start:tz_end, :, :]
                y_out[:, z_start:z_end, y_start:y_end, x_start:x_end]= tmp_label[:, tz_start:tz_end, :, :]

            # ========== Rotation Augmentation ==========
            if np.random.random() < self.cfg.rotate_prob:
                x_out, y_out= self.rotate_xy_3d(
                    img= x_out,
                    mask= y_out,
                )
            x_out = x_out[..., extra_pad//2:-extra_pad//2, extra_pad//2:-extra_pad//2]
            y_out = y_out[..., extra_pad//2:-extra_pad//2, extra_pad//2:-extra_pad//2]
            y_out[0, ...] = 1 - y_out[1:, ...].sum(dim=0, keepdims=True)

            return {
                'input': x_out, 
                'target': y_out,
                }
        elif self.mode == "val":
            return {
                'input': self.data[idx], 
                'target': self.labels[idx],
                }
        elif self.mode == "test":
            tomo= zarr.open(
                    "{}/{}/VoxelSpacing10.000/denoised.zarr".format(
                        self.cfg.test_fpath, self.data[idx],
                        ),
                    )[0]
            tomo= np.array(tomo)
            tomo= self._scale_tomo(tomo)
            tomo= np.expand_dims(tomo, axis=0)
            label= np.array([[0]])

            return {
                'input': tomo,
                'target': label,
                'experiment': self.data[idx],
                }

    
    def __len__(self,):
        if self.mode == "train":
            return self.cfg.batch_size * self.cfg.epoch_steps
        else:
            return len(self.data)


if __name__ == "__main__":
    pass
