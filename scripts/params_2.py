import sys
import os

# Paths and Parameters
input_path = "../input"
output_path = "../output"
# I/O files
pict_data_file = f"{input_path}/pict_data_cell.csv"      # data from PICT here
tag_repr_file = f"{input_path}/tags_representation.csv"  # data for tag representation
model_info = f"{output_path}/model.log"                  # output log file

# I/O directories
tmp_dir = f'{output_path}/tmp'                          # tmp dir to save model logs
pdbs_dir = f'{output_path}/pdbs'
best_pdbs_dir = f'{output_path}/best_pdbs'
rmsd_dir = f'{output_path}/rmsd'



# Modeling parameters
num_models = 10000
n_select = 200
tag_radius = 1
bb_dim = 500
cg_steps = 1000

# Analysis
cluster_threshold = 2000

# Options
verbose = False
out_pdb = True
