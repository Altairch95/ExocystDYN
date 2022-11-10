#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import glob
import sys
import os
import argparse

sys.path.append('/home/altair/PycharmProjects/UCSF_colab/PMI_analysis/pyext/src/')
from analysis_trajectories import AnalysisTrajectories

#########
# PARSER
#########
p = argparse.ArgumentParser(
            description="Extract selected models from RMF files.\n"
                        "Example of usage: "
						"extract_models.py -d  mc_tags --dir_head run_ -cl 2 -st 0"
                        )
p.add_argument('-d', '--directory', action="store", dest="dir_name",
			   help="directory name to process")
p.add_argument('-p', '--prefix', action="store", dest="prefix",
			   default="run_",
			   help="prefix for all directories containing structural "
					"sampling runs. Usually these are named as <prefix>1, <prefix>2, etc.")
p.add_argument('-n', '--nproc', action="store", dest="processors",
			   default="3",
			   help="Number of processors to use")
p.add_argument('-s', '--nskip', action="store", dest="threshold",
			   default="1",
			   help="Threshold for score-based clustering")
p.add_argument('-cl', '--cluster', action="store", dest="cluster",
               help="Specify cluster")
p.add_argument('-st', '--state', action="store", dest="state",
               help="Specify RMF state")
parsero = p.parse_args()


if __name__ == '__main__':
	dir_name = parsero.dir_name  # mc_tags_k1
	output_directory = f"../output/{dir_name}/"
	cl = parsero.cluster
	st = parsero.state
	dir_head = parsero.prefix  # 'run_tags_'
	print(f'\nExtracting models from {output_directory}/{dir_head}, cluster: {cl}, state: {st}\n')
	analys_dir = output_directory + 'analysis/'
	print('Analysis dir is ', analys_dir)
	nproc = parsero.processors
	nskip = int(parsero.threshold)  # threshold for the score-based clustering
	# the lower it is, the more clusters you obtain

	# Check if analysis dir exists
	if not os.path.isdir(analys_dir):
		os.makedirs(analys_dir)

	# How are the trajectories dir names
	out_dirs = glob.glob(output_directory + 'run_*/')
	print('out dir are ', out_dirs)

	################################
	# Get and organize fields for
	# analysis
	################################
	# Read the total score, plot
	# and check for score convengence

	# XLs_cutoffs = {'DSSO':30.0}
	# Load module
	AT = AnalysisTrajectories(out_dirs,
							  analysis_dir=analys_dir,
							  nproc=nproc,
							  nskip=nskip)

	HA = AT.get_models_to_extract(analys_dir + f'/selected_models_A_cluster{cl}_detailed.csv')
	HB = AT.get_models_to_extract(analys_dir + f'/selected_models_B_cluster{cl}_detailed.csv')

	rmf_file_out_A = f'A_models_clust{cl}_{st}.rmf3'
	rmf_file_out_B = f'B_models_clust{cl}_{st}.rmf3'

	# Arguments for do_extract_models_single_rmf:
	# HA :: Dataframe object from AT.get_models_to_extract()
	# file_out :: The output rmf file
	AT.do_extract_models_single_rmf(HA,
									rmf_file_out_A,    # RMF file outputted for Sample A
									output_directory,  # Top directory containing the PMI output folders
									analys_dir,        # Analysis directory to write RMF and score files
									scores_prefix=f'A_models_clust{cl}_{st}',
									sel_state=st)      # Prefix for the scores file

	AT.do_extract_models_single_rmf(HB,
									rmf_file_out_B,
									output_directory,
									analys_dir,
									scores_prefix=f'B_models_clust{cl}_{st}',
									sel_state=st)

	print('\nDONE!\n')
	sys.exit(0)
