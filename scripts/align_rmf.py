import IMP
import IMP.pmi
import IMP.pmi.macros
import RMF
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
import argparse

#########
# PARSER
#########
p = argparse.ArgumentParser(
            description="Align selected RMF files. \n"
                        "Example of usage: align_rmf.py -d  mc_tags -cl 2 -st 0"
                        )
p.add_argument('-d', action="store", dest="dir_name",
               help="directory name to process")
p.add_argument('-cl', action="store", dest="cluster",
               help="Specify cluster")
p.add_argument('-st', action="store", dest="state",
               help="Specify RMF state")
parsero = p.parse_args()


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
		mols = IMP.pmi.tools.get_molecules(hier)
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


def transform_coordinates(hier, transformation):
	# Transform all coordinates
	rbs, beads = IMP.pmi.tools.get_rbs_and_beads(hier)
	for rb in rbs:
		IMP.core.transform(rb, transformation)
	for p in beads:
		temp_coord = IMP.core.XYZ(p)
		IMP.core.transform(temp_coord, transformation)


def get_reference_coordinates(rmf_in, selection=None):
	"""
		Get reference coordinates in reference rmf file
		:param rmf_in: reference rmf file
		:return: coordinates
	"""
	m = IMP.Model()

	f = RMF.open_rmf_file_read_only(rmf_in)
	hier = IMP.rmf.create_hierarchies(f, m)[0]

	IMP.rmf.load_frame(f, RMF.FrameID(0))

	# Get coordinates from frame 1
	ref_coord = get_coordinates_alignment(hier, selection)
	del m, f
	return ref_coord


def align_rmf(rmf_in, rmf_out, ref_coord, selection=None, frames=None):
	"""
	Align selected frames in rmf_in to ref_coordinates and
	calculate RMSD.
	:param rmf_in: input rmf
	:param rmf_out: output rmf
	:param selection: selection of particles
	:param ref_coord: reference coordinates after running Sampcon.py
	:param frames: passing selected frames
	:return:
	"""
	fh_out = RMF.create_rmf_file(rmf_out)

	m = IMP.Model()
	f = RMF.open_rmf_file_read_only(rmf_in)
	print('Number of frames', f.get_number_of_frames())

	if not frames:
		frames = np.arange(0, f.get_number_of_frames(), 100)

	hier = IMP.rmf.create_hierarchies(f, m)[0]
	states = IMP.atom.get_by_type(hier, IMP.atom.STATE_TYPE)
	for i, s in enumerate(states):
		if i == sel_state:
			p = IMP.Particle(m, 'System')
			hier_temp = IMP.atom.Hierarchy.setup_particle(p)
			hier_temp.add_child(s)
			IMP.rmf.add_hierarchy(fh_out, hier_temp)

	RMSD = []
	for i in frames:
		if i % 100 == 0: print('Frame:', i)
		IMP.rmf.load_frame(f, RMF.FrameID(i))

		temp_coord = get_coordinates_alignment(hier, selection)

		ali = IMP.pmi.analysis.Alignment(ref_coord, temp_coord)
		(rmsd, transformation) = ali.align()
		RMSD.append(rmsd)

		transform_coordinates(hier, transformation)
		IMP.rmf.save_frame(fh_out, str(i))

		del temp_coord

	del f

	print('Mean RMSD:', np.mean(np.array(RMSD)))
	return RMSD


if __name__ == '__main__':
	#####################
	# MAIN
	#####################
	# IO files
	dir_name = parsero.dir_name  # mc_tags_k1
	cl = parsero.cl  # 2
	st = parsero.st  # 0
	sel_state = 0

	ref_rmf = f'../output/{dir_name}/analysis/rmsd/cluster.{cl}/cluster_center_model.rmf3'
	rmf_in_A = f'../output/{dir_name}/analysis/A_models_clust{cl}_{st}.rmf3'
	rmf_out_A = f'../output/{dir_name}/analysis/A_models_clust{cl}_{st}_aligned.rmf3'
	rmf_in_B = f'../output/{dir_name}/analysis/B_models_clust{cl}_{st}.rmf3'
	rmf_out_B = f'../output/{dir_name}/analysis/B_models_clust{cl}_{st}_aligned.rmf3'

	cluster_frames_A = f'../output/{dir_name}/analysis/rmsd/cluster.{cl}.sample_A.txt'
	cluster_frames_B = f'../output/{dir_name}/analysis/rmsd/cluster.{cl}.sample_B.txt'

	m = IMP.Model()
	f = RMF.open_rmf_file_read_only(rmf_in_A)
	nframes_A = f.get_number_of_frames()

	frames_A = [int(l.strip()) for l in open(cluster_frames_A, 'r')]
	frames_B = [int(l.strip()) - nframes_A for l in open(cluster_frames_B, 'r')]

	del m, f

	#################################
	# Get reference and align
	#################################
	reference_coordinates = get_reference_coordinates(ref_rmf)
	rmsd_A = align_rmf(rmf_in_A, rmf_out_A, reference_coordinates, frames=frames_A)
	rmsd_B = align_rmf(rmf_in_B, rmf_out_B, reference_coordinates, frames=frames_B)

	#################################
	# Plot RMSD distribution
	#################################
	sns.set(font_scale=3)
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 40))
	ax1.set_title(f'RMSD_A', size=50, y=1.15, fontweight='bold')
	sns.histplot(x=rmsd_A, stat='density', fill=True, ax=ax1)
	ax2.set_title(f'RMSD_B', size=50, y=1.15, fontweight='bold')
	sns.histplot(x=rmsd_B, stat='density', fill=True, ax=ax2)
	plt.tight_layout(pad=3.0)
	# plt.show()
	plt.savefig(f'../output/{dir_name}/analysis/pict_tags_rmsd.png')

	print('\nDONE!\n')
	sys.exit(0)
