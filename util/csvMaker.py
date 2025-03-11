import os
from pathlib import Path
import pandas as pd

image_names = os.listdir(Path("data\FSC147_384_V2\images_384_VarV2"))
density_map_names = [name.replace("jpg", "npy") for name in image_names]

df = pd.DataFrame(data={'jpg_name':image_names, 'dmap_name': density_map_names})
df.to_csv(Path("data\dataset.csv"))