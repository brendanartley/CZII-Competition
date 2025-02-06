import argparse
import glob
import os
import shutil
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp

import zarr
import copick

class Processor():
    def __init__(self, simulated: bool):
        self.simulated= simulated

        if simulated:
            self.in_dir= "./data/raw_sim/train/overlay"
            self.out_dir= "./data/processed_sim/overlay"
            self.filter_type= "wbp"
        else:
            self.in_dir= "./data/raw/train/overlay"
            self.out_dir= "./data/processed/overlay"
            self.filter_type= "denoised"

        self.class2label= {
            'apo-ferritin': 1,
            'beta-galactosidase': 2,
            'ribosome': 3,
            'thyroglobulin': 4,
            'virus-like-particle': 5,
            'beta-amylase': 6,
        }

        if not self.simulated:
            self.add_points= self._load_pseudo_points()

    def _closest_distance(self, group):
        """
        Gets the nearest particle label for each row.
        """
        coords = group[["z", "y", "x"]].values

        # Dist between all possible pairs    
        distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

        # Mask self-distance
        np.fill_diagonal(distances, np.inf)
        closest_distances = distances.min(axis=1)
        group['closest_distance'] = closest_distances
        return group

    def _load_pseudo_points(self, ) -> dict:
        """
        Creates a dictionary of manually
        picked particles (eg. pseudolabels).
        """
        d= {
            "TS_5_4": [],
            "TS_6_4": [],
            "TS_69_2": [],
            "TS_6_6": [],
            "TS_73_6": [],
            "TS_86_3": [],
            "TS_99_9": [],
        }

        df= pd.read_csv("./data/raw/labels_pseudo.csv")
        print("-"*10, " PSEUDO ", '-'*10)
        print(df["particle_type"].value_counts())
        print("-"*10)
        for i, row in df.iterrows():
            d[row["experiment"]].append((
                row["particle_type"],
                row["z"],
                row["y"],
                row["x"]
            ))
        return d

    def _create_label_df(self, ) -> None:
        """
        Creates labels.csv file.
        """
        fpath= "{}/ExperimentRuns/{}/Picks/curation_0_{}.json"
        idxs= os.listdir("{}/ExperimentRuns/".format(self.out_dir))

        # Manual skip points
        # Some are added into the pseudolabels w/ corrected centers
        skip_points= {
            "TS_5_4": [
                ('thyroglobulin', 279.908, 221.468, 4527.706),
            ],
            "TS_6_4": [
                ('virus-like-particle', 745.927, 2340.359, 6045.947),
                ('virus-like-particle', 671.21, 5638.402, 911.29),
                ('ribosome', 1144.294, 3282.821, 650.107),
                ('ribosome', 806.34, 1197.481, 4654.602),
            ],
            "TS_69_2": [],
            "TS_6_6": [
                ('ribosome', 933.065, 4506.473, 4333.457)
            ],
            "TS_73_6": [
                ],
            "TS_86_3": [
                ('ribosome', 562.929, 4594.434, 4327.26),
                ('ribosome', 893.643, 4321.102, 5689.451),
                ('ribosome', 1086.719, 3972.649, 1217.9),
            ],
            "TS_99_9": [
                ('ribosome', 1181.704, 4207.943, 271.44),
            ],
        }

        # Create dataframe
        all_rows= []
        for idx in idxs:
            for c in self.class2label.keys():
                
                # Load picks
                json_file= fpath.format(self.out_dir, idx, c)
                try:
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                except:
                    continue

                # Iterate picks
                for row in json_data["points"]:

                    # Skip
                    if idx in skip_points:
                        if (c, row["location"]["z"], row["location"]["y"], row["location"]["x"]) in skip_points[idx]:
                            print("Manual_skip:", idx, row["location"])
                            continue

                    all_rows.append({
                        'experiment': idx, 
                        'particle_type': c,
                        'z': row["location"]["z"],
                        'y': row["location"]["y"],
                        'x': row["location"]["x"],
                        })
            
            if not self.simulated:
                # Add picks
                for row in self.add_points[idx]:
                    particle_type, z, y, x= row
                    all_rows.append({
                        "experiment": idx,
                        'particle_type': particle_type,
                        'z': z * 10.012444,
                        'y': y * 10.012444,
                        'x': x * 10.012444,
                    })
        df= pd.DataFrame(all_rows)

        # Set overlapping points to beta-galactosidase
        if not self.simulated:

            # Select overlapping points
            df = df.groupby("experiment")[df.columns].apply(lambda x: self._closest_distance(x)).reset_index(drop=True)
            z= df[((df["closest_distance"] < 100) & (df["particle_type"].isin(["thyroglobulin", "beta-amylase", "beta-galactosidase"])))].copy()

            # Add to skip points 
            for _, row in z.iterrows():
                if row["particle_type"] != "beta-galactosidase":
                    skip_points[row["experiment"]].append(
                        (row["particle_type"], row["z"], row["y"], row["x"])
                    )

            # Adjust center
            z= z.groupby(['experiment', 'closest_distance'], as_index=False)[['z', 'y', 'x']].mean() 
            z["particle_type"]= "beta-galactosidase"

            # Add to labels
            df= df[~((df["closest_distance"] < 100) & (df["particle_type"].isin(["thyroglobulin", "beta-amylase", "beta-galactosidase"])))]
            df= pd.concat([df, z], axis=0)
            df= df.drop(columns=["closest_distance"])

        # Save
        if self.simulated:
            f= "./data/raw/labels_sim.csv"
        else:
            f= "./data/raw/labels.csv"
        df.to_csv(f, index=False)
        print("-"*10, " Updated Labels ", "-"*10)

        return skip_points

    def _adjust_steepness(self, x, steepness: float = 25, center: float = 0.5):
        return 1 / (1 + np.exp(-steepness * (x - center)))

    def _create_heatmap(
        self, 
        arr,
        idx,
        points, 
        radius,  
    ):
        """
        Creates 3D heatmap value by value.

        Note: Not efficient for large segmentation maps.
        """
        
        # Iterate center points
        for pz, py, px in points:

            # Add offset to center pixel
            pz= pz+0.5
            py= py+0.5
            px= px+0.5

            # Define bounds to compute kernel
            z_min = max(0, int(pz - 3 * radius))
            z_max = min(arr.shape[1], int(pz + 3 * radius + 1))
            y_min = max(0, int(py - 3 * radius))
            y_max = min(arr.shape[2], int(py + 3 * radius + 1))
            x_min = max(0, int(px - 3 * radius))
            x_max = min(arr.shape[3], int(px + 3 * radius + 1))

            # Iterate each pixel
            for z in range(z_min, z_max):
                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        dist = np.sqrt((z - pz)**2 + (y - py)**2 + (x - px)**2)
                        if dist <= 3*radius:

                            # Calc value
                            gaussian_value = np.exp(-dist**2 / (2 * radius**2))

                            # Adjust steepness
                            gaussian_value= self._adjust_steepness(gaussian_value)
                            arr[idx, z, y, x] = max(arr[idx, z, y, x], gaussian_value)

        return arr

    def _run_helper(self, args):
        """
        Process a single run/tomogram.
        """
        run, root, targets= args

        # Pixel scaling
        fpath= "{}/ExperimentRuns/{}/VoxelSpacing10.000/{}.zarr/.zattrs".format(
            self.in_dir.replace("/overlay", "/static"),
            run.name,
            self.filter_type,
            )
        with open(fpath, "r") as f:
            z= json.load(f)
        z_scale, y_scale, x_scale= z["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]

        # Tomogram
        tomo = run.get_voxel_spacing(10) \
                    .get_tomogram(self.filter_type) \
                    .numpy()
        label = np.zeros((len(self.class2label.keys())+1, *tomo.shape), dtype=np.float32)

        # Segmentation mask
        count_dict= {}
        for o in root.pickable_objects:
            pick = run.get_picks(o.name, user_id="curation")

            scaled_points= []
            if len(pick) == 1:

                # Scale points
                for p in pick[0].points:
                    
                    # Cleaned data points
                    if run.name in self.skip_points:
                        if (o.name, p.location.z, p.location.y, p.location.x) in self.skip_points[run.name]:
                            print("Skipping:", run.name, o.name, p.location)
                            continue

                    scaled_points.append((
                        (p.location.z / z_scale), 
                        (p.location.y / y_scale), 
                        (p.location.x / x_scale), 
                        ))
                        
                # Add manual annotations
                if not self.simulated:
                    for row in self.add_points[run.name]:
                        particle_type, z, y, x= row
                        if particle_type == o.name:
                            scaled_points.append((
                                z, 
                                y, 
                                x,
                                ))

                # Create heatmap
                idx= targets[o.name]["label"]
                radius= targets[o.name]["radius"] / (10 * 3.8)
                label= self._create_heatmap(
                    arr= label,
                    idx= idx,
                    points= scaled_points,
                    radius= radius,
                )
                count_dict[o.name]= len(scaled_points)

        # Normalize labels
        label= np.clip(label, 0.0, 1.0)
        lmax= label.sum(axis=0).max()
        if lmax > 1.0:
            print(run.name, "LabelOverlap.", lmax)
        label[0, ...]= 1 - np.sum(label[1:], axis=0) # set background class
        label= np.clip(label, 0.0, 1.0)
        label= label.astype(np.float16)

        # Save segmentation
        fpath= os.path.join(
            self.out_dir, 
            f"ExperimentRuns/{run.name}/{run.name}_segmentation.npy",
            )
        np.save(fpath, label)     

        return run

    def run(self) -> None:
        
        # Load cfg
        if self.simulated:
            f= "./src/pre/copick_overlay_sim.json"
        else:
            f= "./src/pre/copick_overlay.json"
        root = copick.from_file(f)

        # Get targets
        targets= {}
        for o in root.pickable_objects:
            if o.is_particle:
                targets[o.name] = {}
                targets[o.name]['label'] = self.class2label[o.name]
                targets[o.name]['radius'] = o.radius

        # Setup new overlay directory
        # Source: https://www.kaggle.com/code/kharrington/deepfindet-train?scriptVersionId=204524304&cellId=3
        fpaths= glob.glob(os.path.join(self.in_dir, "ExperimentRuns/*/Picks/*.json"))
        for f in fpaths:
            new_f= f.replace(self.in_dir, self.out_dir)
            new_f= new_f.replace("/Picks/", "/Picks/curation_0_")

            # Make new dir
            new_dir= "/".join(new_f.split("/")[:-1])
            os.makedirs(new_dir, exist_ok=True)

            # Copy file
            shutil.copy(f, new_f)

        if not self.simulated:
            self.skip_points= self._create_label_df()
        else:
            self.skip_points= {}

        # Run
        arr= [(run, root, targets) for run in root.runs]
        with mp.Pool() as p:
            results = list(tqdm(p.imap(self._run_helper, arr), total=len(arr)))    

        return


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--simulated", action="store_true", help="Simulated data or not.")
    args= parser.parse_args()
    return args

if __name__ == "__main__":
    args= parse_args()
    p = Processor(**vars(args))
    p.run()
