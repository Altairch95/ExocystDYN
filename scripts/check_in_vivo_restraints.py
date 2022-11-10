#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function

import time

import IMP
import IMP.pmi
import IMP.pmi.analysis
import IMP.pmi.output
import IMP.atom
import RMF
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import scipy
import scipy.spatial
import scipy.spatial.distance
from scipy.spatial.distance import pdist

import matplotlib as mpl
from operator import add

mpl.rcParams.update({'font.size': 8})

sys.path.append('/home/altair/PycharmProjects/UCSF_colab/PMI_analysis/pyext/src/')

#########
# PARSER
#########
p = argparse.ArgumentParser(
	description="Check % of in vivo restraints fulfilled in each RMF frame "
				"and plot the results.\n"
				"Example of usage: "
				"check_in_vivo_restraints.py -d  mc_tags -cl 2 -st 0"
)
p.add_argument('-d', action="store", dest="dir_name",
			   help="directory name to process")
p.add_argument('-cl', action="store", dest="cluster",
			   help="Specify cluster")
p.add_argument('-st', action="store", dest="state",
			   help="Specify RMF state")
parsero = p.parse_args()


def get_particles_at_lowest_resolution(hierarchy):
	"""
        Read rmf3 file and return only coordinates of beads at
        lowest resolution
    """
	particles_dict = {}
	for mol in IMP.atom.get_by_type(hierarchy, IMP.atom.MOLECULE_TYPE):
		copy = IMP.atom.Copy(mol).get_copy_index()
		sel = IMP.atom.Selection(mol, resolution=1)
		d = IMP.core.XYZR(sel.get_selected_particles()[0])
		crd = np.array([d.get_x(), d.get_y(), d.get_z()])
		particles_dict[mol.get_name()] = crd
	# particles_dict[mol.get_name()] = sel.get_selected_particles()

	return particles_dict


def get_coords_array(particles_list):
	"""
        Get all beads coordinates and radii
    """
	coords = []
	radii = []
	for p in particles_list:
		residue_indexes = IMP.pmi.tools.get_residue_indexes(p)
		if len(residue_indexes) != 0:
			for res in range(min(residue_indexes), max(residue_indexes) + 1):
				d = IMP.core.XYZR(p)
				crd = np.array([d.get_x(), d.get_y(), d.get_z()])
				coords.append(crd)
				radii.append(d.get_radius())

	return np.array(coords), np.array(radii)


def get_min_distances(particles_1, particles_2):
	"""
    Given two proteins, computes the contact map
    """

	coords1, radii1 = get_coords_array(particles_1)
	coords2, radii2 = get_coords_array(particles_2)
	distances = scipy.spatial.distance.cdist(coords1, coords2)
	# distances = (distances - radii2).T - radii1
	# distances = np.array(distances)
	min_distances = np.apply_along_axis(min, 1, distances)
	return min_distances


def update_Distances(rmf_file, frames, selection_regions, min_distances):
	model = IMP.Model()
	for fr in frames:
		try:
			hier = IMP.pmi.analysis.get_hiers_from_rmf(model, fr, rmf_file)[0]
		except:
			print('Missing frame')
			continue
		prot_dictionary = get_particles_at_lowest_resolution(hier)
		for i, t in enumerate(selection_regions):
			particles1 = IMP.atom.Selection(hier,
											molecule=t[2],
											residue_indexes=list(range(t[0], t[1] + 1))).get_selected_particles()
			particles2 = IMP.atom.Selection(hier,
											molecule=t[5],
											residue_indexes=list(range(t[3], t[4] + 1))).get_selected_particles()
			min_d = get_min_distances(particles1, particles2)
			min_distances[i] = min_distances[i] + [np.min(min_d)]


def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]


def process_frames(rmf_file, frames, pict_df):
	"""
    Checking in vivo restraints in pict_df for each frame in frames
    on the rmf_file.

    Reading RMF files in chunks to speed up the process.
    :param rmf_file: rmf3 file
    :param frames: range of frames
    :param pict_df: dataframe with reference distances to check
    :return:
    """
	# Reading RMF files
	m = IMP.Model()
	frames_percent_fulfilled = list()
	# ranges = list(chunks(range(1, 1000), 500))
	for fr in frames[ran]:
		start = time.time()  # Tracking out time
		try:
			hier = IMP.pmi.analysis.get_hiers_from_rmf(m, fr, rmf_file)[0]  # Load hierarchy from RMF
		except:
			print('Missing frame')
			continue
		frame_particles = get_particles_at_lowest_resolution(hier)
		pair_distances = [np.round(pdist(np.array([frame_particles[pair[0]], frame_particles[pair[1]]]))[0], 3) for
						  pair in pict_df.to_numpy()]
		pict_df.loc[:, "frame_dist"] = pair_distances  # for debugging purposes
		check_fulfilled = np.where((pict_df.frame_dist >= pict_df.distance - (2 * pict_df.serr_mu)) &
								   (pict_df.frame_dist <= pict_df.distance + (2 * pict_df.serr_mu)), 1, 0)
		pict_df.fulfilled = list(map(add, pict_df.fulfilled, check_fulfilled))
		frames_percent_fulfilled.append(np.round(np.count_nonzero(check_fulfilled == 1) * 100 /
												 check_fulfilled.size, 3))
		print(f'Frame {fr} checked in {time.time() - start} s\n')
	print(f'Chunk {ran} done!\n')
	time.sleep(1)

	return frames_percent_fulfilled


def plot(df, percent_list, name):
	"""
    Plot the distribution of distance restraints fulfilled per bait-FRB.

    :param df: dataframe with values
    :param percent_list: list with percentages (y-axis)
    :param name: name of the plot
    :return:
    """
	sns.set(font_scale=3)
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 40))
	ax1.set_title(f'Sample_{name}', size=50, y=1.15, fontweight='bold')
	sns.histplot(data=df, x='fulfilled', hue='bait', stat='count', fill=True, ax=ax1)
	sns.histplot(x=percent_list, stat='density', fill=True, ax=ax2)
	sns.move_legend(ax1, 'upper center', bbox_to_anchor=(0.5, -0.1), ncol=5,
					title='Anchor Restraints fulfilled % ', frameon=False)
	ax1.set_xlabel('')
	ax1.set_ylabel('Count')
	ax2.set_xlabel('Fulfilled % ', labelpad=20)
	ax2.set_ylabel('Density', labelpad=20)
	plt.tight_layout(pad=3.0)
	# plt.show()
	plt.savefig(f'../output/analysis/pict_tags_{name}.png')


if __name__ == '__main__':
	###########
	# MAIN
	############
	# IMP.set_log_level(IMP.SILENT)
	dir_name = parsero.dir_name  # mc_tags_k1
	data_directory = "../input/data/"
	pict_distances = data_directory + "pict_restraints.csv"
	print(f'\nChecking in vivo restraints from file {pict_distances}\n')

	cluster = parsero.cl   # 2
	state = parsero.st     # 0
	aligned = input('\tAre the rmf files aligned? (y/n): ')
	if aligned == 'y':
		aligned_name = '_aligned'
		rmf_A = f'../output/{dir_name}/analysis/A_models_clust{cluster}_{state}_aligned.rmf3'
		rmf_B = f'../output/{dir_name}/analysis/B_models_clust{cluster}_{state}_aligned.rmf3'
	elif aligned == 'n':
		aligned_name = ''
		rmf_A = f'../output/{dir_name}/analysis/A_models_clust{cluster}_{state}.rmf3'
		rmf_B = f'../output/{dir_name}/analysis/B_models_clust{cluster}_{state}.rmf3'

	# Open RMF files in read only mode to get number of frames
	rhA = RMF.open_rmf_file_read_only(rmf_A)
	rhB = RMF.open_rmf_file_read_only(rmf_B)
	frames_A = np.arange(0, rhA.get_number_of_frames(), 1)
	frames_B = np.arange(0, rhB.get_number_of_frames(), 1)
	print('Frames A', len(frames_A))
	print('Frames B', len(frames_B))

	# Load in vivo (PICT) data into a Dataframe and modify nm to Angstroms
	data = pd.read_csv(pict_distances, float_precision='round_trip')
	data.loc[:, ['distance', 'serr_mu',
				 'sigma', 'serr_sigma']] = data.copy().loc[:, ['distance', 'serr_mu',
															   'sigma', 'serr_sigma']].apply(lambda x: round(x * 10, 3))
	data.loc[:, "fulfilled"] = np.zeros(data.shape[0], dtype=int)

	# Define chunks to make the process fast: when reading RMF frames, the longer we read, the slower it is.
	# A solution for that is to make chunks, so we open and close the RMF sequentially;
	# Another solution would be to use multiprocessing, but sometimes can be misleading...

	# First create chunks
	ranges_A = list(chunks(range(1, len(frames_A)), 250))
	df_A = data.copy()
	percent_A = list()
	# Parse RMF frames in chunks
	for ran in ranges_A:
		percent_A += process_frames(rmf_A, frames_A, df_A)
	df_A.loc[:, "fulfilled"] = df_A.loc[:, 'fulfilled'].apply(lambda i: round(i * 100 / len(percent_A), 3))
	# Plot results for all dataset
	plot(df_A, percent_A, f'A{aligned_name}')
	# Do the same with dataset B
	ranges_B = list(chunks(range(1, len(frames_B)), 250))
	df_B = data.copy()
	percent_B = list()
	for ran in ranges_B:
		percent_B += process_frames(rmf_B, frames_B, df_B)
	df_B.loc[:, "fulfilled"] = df_B.loc[:, 'fulfilled'].apply(lambda i: round(i * 100 / len(percent_B), 3))
	plot(df_B, percent_B, f'B{aligned_name}')

	print('\nDONE!\n')
	sys.exit(0)
