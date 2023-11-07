import torch
import numpy as np

base_folder = 'D:/MachineLearning/datasets/VesuviusDS/'#"G:/VS_CODE/CV/Vesuvius Challenge/"
output_folder = base_folder + "3d_surface/"

#Run once for each fragment
i=1 #3 4
frag = torch.load(output_folder + "scan_{i}_0d64.pt").to('cpu')
frag = frag.detach().numpy()
shp = np.shape(frag)
np.save(base_folder + f'numpy/F{i}_{shp[0]}-{shp[1]}-{shp[2]}', frag)