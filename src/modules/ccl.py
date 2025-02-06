import numpy as np
import pandas as pd

import cc3d

from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

def mask2centroids(
    preds, 
    experiment, 
    params= {}, 
    log: bool = False,
    ):  

    default_params= {
        'apo-ferritin': {
            "min_volume": 5, 
            "threshold": 0.05,
            "radius": 6,
            },
        'beta-galactosidase': {
            "min_volume": 5, 
            "threshold": 0.5,
            "radius": 9,
            },
        'ribosome': {
            "min_volume": 5, 
            "threshold": 0.25,
            "radius": 15,
            },
        'thyroglobulin': {
            "min_volume": 5, 
            "threshold": 0.25,
            "radius": 13,
            },
        'virus-like-particle': {
            "min_volume": 5, 
            "threshold": 0.50,
            "radius": 13.5,
        }
    }
    class2label= {
                'apo-ferritin': 1,
                'beta-galactosidase': 2,
                'ribosome': 3,
                'thyroglobulin': 4,
                'virus-like-particle': 5
            }
    label2class= {v:k for k,v in class2label.items()}

    # Use default if not specified
    for k in class2label.keys():
        if k not in params:
            params[k]= default_params[k]

    # Iterate classes
    all_preds= []
    for particle_type, idx in class2label.items():

        # Params
        thresh= params[particle_type]["threshold"]
        min_volume= params[particle_type]["min_volume"]
        max_volume= 1_000_000

        # Binarize preds
        cpreds= preds[class2label[particle_type]]
        cpreds= np.where(cpreds > thresh, 1, 0).astype(np.uint8)
        if log:
            print("particle_type: {}, pixels: {:_}".format(particle_type, cpreds.sum()))

        # Apply CCL
        components= cc3d.connected_components(cpreds)
        stats = cc3d.statistics(components)

        # Filter volumes
        for c, vc in zip(stats["centroids"], stats["voxel_counts"]):
            z= c[0] * 10.012444
            y= (c[1] - 0.5) * 10.012444
            x= (c[2] - 0.5) * 10.012444
            if vc > min_volume and vc < max_volume:
                all_preds.append({
                    'experiment': experiment, 
                    'particle_type': particle_type,
                    'z': z,
                    'y': y,
                    'x': x,
                    'vc': vc,
                    })
    df= pd.DataFrame(all_preds)

    # Merge overlapping particle centers
    group_cols= ["experiment", "particle_type", "z", "y", "x"]
    if len(df) > 0:
        df= df.groupby(["experiment", "particle_type"])[group_cols].apply(
            merge_particle_centers, 
            ).reset_index(drop=True)

    return df


def merge_particle_centers(
    group, 
    div_factor: float = 2,
    ):
    """
    Merges particles centers within radius / divfactor 
    of each other.

    Operates on pandas groupby.apply().
    """

    # Thresholds
    particle_radius = {
        'apo-ferritin': 60,
        'beta-amylase': 65,
        'beta-galactosidase': 90,
        'ribosome': 150,
        'thyroglobulin': 130,
        'virus-like-particle': 135,
    }
    threshold= particle_radius[group["particle_type"].iloc[0]] / div_factor

    # Pairwise distances
    coords = group[["z", "y", "x"]].to_numpy()
    distances = cdist(coords, coords, metric='euclidean')
    
    # Create adjacency matrix (1 if within threshold, 0 otherwise)
    adjacency_matrix = distances <= threshold
    graph = csr_matrix(adjacency_matrix)
    
    # Find connected points
    n_components, labels = connected_components(
        csgraph=graph, 
        directed=False,
        )
    
    # Set coords as mean of group
    group["cluster"] = labels
    group= group.groupby(["cluster", "experiment", "particle_type"]).mean().reset_index()
    group= group.drop(columns=["cluster"])

    return group
