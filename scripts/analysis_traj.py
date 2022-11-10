#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
import glob
import sys
import os
import argparse
import time

sys.path.append('/home/altair/PycharmProjects/UCSF_colab/PMI_analysis/pyext/src/')
from analysis_trajectories import AnalysisTrajectories

#########
# PARSER
#########
p = argparse.ArgumentParser(
	description="Analyse trajectories from RMF files.\n"
				"Example of usage: analysis_traj.py -d  mc_tags -p run_ "
				"-c 1 -n 1 - r DR,EV"
)
p.add_argument('-d, --directory', action="store", dest="dir_name",
			   help="directory name to process")
p.add_argument('-p, --prefix', action="store", dest="prefix",
			   default="run_",
			   help="prefix for all directories containing structural "
					"sampling runs. Usually these are named as <prefix>1, <prefix>2, etc.")
p.add_argument('-c, --nproc', action="store", dest="processors",
			   default="3",
			   help="Number of processors to use")
p.add_argument('-n, --nskip', action="store", dest="threshold",
			   default="20",
			   help="Threshold for score-based clustering")
p.add_argument('-r, --restraints', action="store", dest="restraint_list",
			   default="1",
			   help="Which restraints would you like to analyse? \n"
					"EV: Excluded volume,\n"
					"DR: Distance Restraint\n"
					"CR: Connectivity Restraint\n"
					"XLs: Cross-links restraints")
parsero = p.parse_args()

if __name__ == '__main__':
	dir_name = parsero.dir_name  # mc_tags_k1
	data_directory = "../input/data/"
	output_directory = f"../output/{dir_name}/"
	dir_head = parsero.prefix  # 'run_tags_'
	print(f'\nExtracting models from {output_directory}{dir_head}\n')

	analys_dir = output_directory + 'analysis/'
	print('Analysis dir is ', analys_dir)
	nproc = int(parsero.processors)
	nskip = int(parsero.threshold)  # threshold for the score-based clustering
	restraints = [r + '_sum' for r in parsero.restraint_list.split(',')]
	# the lower it is, the more clusters you obtain
	print('nproc:  ', nproc)
	print('nskip:  ', nskip)
	print('restraints: ', restraints)

	# Check if analysis dir exists
	if not os.path.isdir(analys_dir):
		os.makedirs(analys_dir)

	# How are the trajectories dir names
	out_dirs = glob.glob(output_directory + dir_head + '*/')
	print(f'Processing {len(out_dirs)} run directories ...\n')

	time.sleep(2)
	################################
	# Get and organize fields for
	# analysis
	################################
	# Read the total score, plot
	# and check for score convergence

	# Load module
	AT = AnalysisTrajectories(out_dirs,
							  dir_name=dir_head,
							  analysis_dir=analys_dir,
							  nproc=nproc,
							  nskip=nskip)
	# Define restraints to analyze
	if 'DR_sum' in restraints:
		AT.set_analyze_Distance_restraint()
	if 'XLs_sum' in restraints:
		XLs_cutoffs = {'DSSO': 30.0}
		AT.set_analyze_XLs_restraint(XLs_cutoffs=XLs_cutoffs)
	if 'CR_sum' in restraints:
		AT.set_analyze_Connectivity_restraint()
	if 'EV_sum' in restraints:
		AT.set_analyze_Excluded_volume_restraint()

	# Read stat files
	AT.read_stat_files()
	AT.write_models_info()
	if 'XLs' in restraints:
		AT.get_psi_stats()

	AT.hdbscan_clustering(restraints)  # 'EV_sum', 'DR_sum', 'CR_sum'
	if 'XLs' in restraints:
		AT.summarize_XLs_info()

	print('\nDONE!\n')
	sys.exit(0)
