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
from statistics import mean, stdev
import numpy as np
import sys
import scipy
import scipy.spatial
import scipy.spatial.distance

import matplotlib as mpl

mpl.rcParams.update({'font.size': 8})
sys.path.append('/home/altair/PycharmProjects/UCSF_colab/PMI_analysis/pyext/src/')

#########
# PARSER
#########
p = argparse.ArgumentParser(
            description="Process already aligned rmf filed to calculate the RMSD against "
						"a reference rmf file. Plot the results as the dispersion of each "
						"element location (error bar plot)\n"
                        "Example of usage: "
						"process_aligned_rmf.py -d  mc_tags --dir_head run_ -cl 2 -st 0"
                        )
p.add_argument('-d', action="store", dest="dir_name",
               help="directory name to process")
p.add_argument('--dir_head', action="store", dest="dir_head",
               default="run_",
               help="Name of each MC run folder")
p.add_argument('-cl', action="store", dest="cluster",
               help="Specify cluster")
p.add_argument('-st', action="store", dest="state",
               help="Specify RMF state")
parsero = p.parse_args()


def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]


def get_coordinates_alignment(hier, selection=None):
	coord_dict = {}

	if selection:
		for k, v in selection.items():
			sel = IMP.atom.Selection(hier,
									 molecule=v[0],
									 residue_indexes=np.arange(v[1], v[2], 1),
									 resolution=IMP.atom.ALL_RESOLUTIONS,
									 copy_index=v[3]).get_selected_particles()
			coords = [np.array(IMP.core.XYZ(p).get_coordinates())
					  for p in sel]
			coord_dict[k] = coords

	else:
		mols = sorted(IMP.pmi.tools.get_molecules(hier), key=lambda k: k.get_name()[:4], reverse=True)
		# print(mols)
		for m in mols:
			sel = IMP.atom.Selection(hier,
									 molecule=m.get_name(),
									 copy_index=IMP.atom.Copy(m).get_copy_index(),
									 resolution=IMP.atom.ALL_RESOLUTIONS).get_selected_particles()

			coords = [np.array(IMP.core.XYZ(p).get_coordinates())
					  for p in sel]

			coord_dict[m.get_name()] = coords

	return coord_dict


def get_reference_coordinates(rmf_in, selection=None):
	m = IMP.Model()

	f = RMF.open_rmf_file_read_only(rmf_in)
	hier = IMP.rmf.create_hierarchies(f, m)[0]

	IMP.rmf.load_frame(f, RMF.FrameID(0))

	# Get coordinates from frame 1
	ref_coord = get_coordinates_alignment(hier, selection)
	del m, f
	return ref_coord


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


def check_distance_to_ref(ref_coords, rmf_mov, frames_mov):
	"""
	Method to measure the distance distribution from a set of solutions
	to its reference point.
	:param ref_coords: reference coordinates
	:param rmf_mov: mov rmf file
	:param frames_mov: range of frames mov
	:return: dictionary with distances
	"""
	# Reading RMF files
	m = IMP.Model()
	distances = dict()
	for k in ref_coords:
		distances[k] = list()
	for fr in frames_mov[ran]:
		start = time.time()  # Tracking out time
		try:
			hier = IMP.pmi.analysis.get_hiers_from_rmf(m, fr, rmf_mov)[0]  # Load hierarchy from RMF
		except:
			print('Missing frame')
			continue
		frame_particles = get_particles_at_lowest_resolution(hier)
		for k in distances:
			coords1 = ref_coords[k]
			coords2 = frame_particles[k]
			distances[k].append(np.round(scipy.spatial.distance.pdist(np.vstack((coords1[0], coords2)))[0], 3))
		end = time.time()
		print(f'Frame {fr} analyzed in {end - start} s\n')
	return distances


def plot_distance_to_ref_distribution(distance_dict, save_path, title):
	"""
	Plot the dispersion for each fluorophore location as an
	error bar plot with mean +- stdev
	:param distance_dict: dictionary with each tag data
	:param save_path: path to save plot
	:param title: title of the plot
	"""
	fig, ax = plt.subplots(figsize=(20, 8))
	ax.set_title(title, fontdict={'fontsize': 20}, loc='center', pad=15.0)
	ax.errorbar(x=[x * 0.1 for x in list(range(len(distance_dict.keys())))], y=[mean(y) for y in distance_dict.values()],
				yerr=[stdev(y) for y in distance_dict.values()], fmt='^', c='orange', capsize=5, markersize=7)
	ax.set_xticks([x * 0.1 for x in list(range(len(distance_dict.keys())))])
	ax.set_xticklabels(labels=[l for l in distance_dict.keys()], rotation=40, ha='right')
	ax.set_ylabel(ylabel='RMSD (Angstroms)', fontdict={'fontsize': 20}, labelpad=10.0)
	ax.set_xlabel(xlabel='Fluorophores \n(FRB & GFP)', fontdict={'fontsize': 20}, labelpad=10.0)
	plt.tight_layout()
	# plt.show()
	plt.savefig(save_path, dpi=100)


def write_rmsd(file, data, sample_name, reference_rmf, aligned_rmf):
	with open(file, 'w') as rmsd_file:
		rmsd_file.write(f'Sample {sample_name}: RMSD frames A from {aligned_rmf} to reference {reference_rmf}\n')
		rmsd_file.write('tag,mean_distance,stdev\n')
		for k, v in data.items():
			rmsd_file.write(f'{k},{mean(v)},{stdev(v)}\n')


if __name__ == '__main__':
	#####################
	# MAIN
	#####################
	# IO files
	dir_name = parsero.dir_name  # mc_tags_k1
	cl = parsero.cl              # 2
	st = parsero.st              # 0

	ref_rmf = f'../output/{dir_name}/analysis/rmsd/cluster.{cl}/cluster_center_model.rmf3'
	rmf_A_aligned = f'../output/{dir_name}/analysis/A_models_clust{cl}_{st}_aligned.rmf3'
	rmf_B_aligned = f'../output/{dir_name}/analysis/B_models_clust{cl}_{st}_aligned.rmf3'

	rhref = RMF.open_rmf_file_read_only(ref_rmf)
	rhA = RMF.open_rmf_file_read_only(rmf_A_aligned)
	rhB = RMF.open_rmf_file_read_only(rmf_B_aligned)

	frames_ref = rhref.get_number_of_frames()
	frames_A = np.arange(0, rhA.get_number_of_frames(), 1)
	frames_B = np.arange(0, rhB.get_number_of_frames(), 1)

	print('Frames ref', frames_ref)
	print('Frames A', len(frames_A))
	print('Frames B', len(frames_B))

	# Get reference coordinates
	reference_coordinates = get_reference_coordinates(ref_rmf)
	# Distances A to ref
	distances_A = dict()
	for ran in list(chunks(range(1, len(frames_A)), 250)):
		if len(distances_A) == 0:
			distances_A.update(check_distance_to_ref(reference_coordinates, rmf_A_aligned, frames_A))
		else:
			distances = check_distance_to_ref(reference_coordinates, rmf_A_aligned, frames_A)
			for k, v in distances.items():
				distances_A[k] += v

	plot_distance_to_ref_distribution(distances_A, f'../output/{dir_name}/analysis/rmsd/rmsd_A.png', 'Sample_A')
	write_rmsd(f'../output/{dir_name}/analysis/rmsd/rmsd_A.txt', distances_A, 'A', ref_rmf, rmf_A_aligned)

	# Distances B to ref
	distances_B = dict()
	for ran in list(chunks(range(1, len(frames_B)), 250)):
		if len(distances_B) == 0:
			distances_B.update(check_distance_to_ref(reference_coordinates, rmf_B_aligned, frames_B))
		else:
			distances = check_distance_to_ref(reference_coordinates, rmf_B_aligned, frames_B)
			for k, v in distances.items():
				distances_B[k] += v
	plot_distance_to_ref_distribution(distances_B, f'../output/{dir_name}/analysis/rmsd/rmsd_B.png', 'Sample_B')
	write_rmsd(f'../output/{dir_name}/analysis/rmsd/rmsd_B.txt', distances_B, 'B', ref_rmf, rmf_B_aligned)

