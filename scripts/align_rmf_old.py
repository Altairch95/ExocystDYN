import IMP
import IMP.pmi
import IMP.pmi.macros
import RMF
import sys
import numpy as np


def get_particles_at_lowest_resolution(hierarchy):
	"""
	Read rmf3 file and return only coordinates of beads at
	lowest resolution
    """
	particles_dict = {}
	for mol in IMP.atom.get_by_type(hierarchy, IMP.atom.MOLECULE_TYPE):
		copy = IMP.atom.Copy(mol).get_copy_index()
		sel = IMP.atom.Selection(mol, resolution=1)
		d = IMP.core.XYZ(sel.get_selected_particles()[0])
		crd = np.array([d.get_x(), d.get_y(), d.get_z()])
		particles_dict[mol.get_name()] = crd
		# particles_dict[mol.get_name()] = sel.get_selected_particles()

	return particles_dict


def get_coords_array(particles_list):
	"""Get all beads coordinates and radii"""
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


def get_coordinates_alignment(hier):
	coord_dict = {}

	for mol in IMP.atom.get_by_type(hier, IMP.atom.MOLECULE_TYPE):
		sel = IMP.atom.Selection(hier).get_selected_particles()

		coords = [np.array(IMP.core.XYZ(p).get_coordinates())
				  for p in sel]

		coord_dict[mol] = coords

	return coord_dict


def transform_coordinates(hier, transformation):
	# Transform all coordinates
	rbs, beads = IMP.pmi.tools.get_rbs_and_beads(hier)
	for rb in rbs:
		IMP.core.transform(rb, transformation)
	for p in beads:
		temp_coord = IMP.core.XYZ(p)
		IMP.core.transform(temp_coord, transformation)


def get_reference_coordinates(rmf_in):
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
	# ref_coord = get_coordinates_alignment(hier, selection)
	ref_coord = get_particles_at_lowest_resolution(hier)
	del m, f
	return ref_coord


def align_rmf(rmf_in, rmf_out, selection, ref_coord):
	"""
	Align selected frames in rmf_in to ref_coordinates and
	calculate RMSD.
	:param rmf_in: input rmf
	:param rmf_out: output rmf
	:param selection: selected frames after running Sampcon.py
	:param ref_coord: reference coordinates after running Sampcon.py
	:return:
	"""
	fh_out = RMF.create_rmf_file(rmf_out)

	m = IMP.Model()
	f = RMF.open_rmf_file_read_only(rmf_in)
	print('Number of frames', f.get_number_of_frames())

	hier = IMP.rmf.create_hierarchies(f, m)[0]
	states = IMP.atom.get_by_type(hier, IMP.atom.STATE_TYPE)
	for i, s in enumerate(states):
		if i == sel_state:
			p = IMP.Particle(m, 'System')
			hier_temp = IMP.atom.Hierarchy.setup_particle(p)
			hier_temp.add_child(s)
			IMP.rmf.add_hierarchy(fh_out, hier_temp)

	RMSD = []
	# Load selected frames to compare
	for i in selection:
		print('Frame:', i)
		IMP.rmf.load_frame(f, RMF.FrameID(i))

		# temp_coord = get_coordinates_alignment(hier, selection)
		temp_coord = get_particles_at_lowest_resolution(hier)

		ali = IMP.pmi.analysis.Alignment(ref_coord, temp_coord)
		(rmsd, transformation) = ali.align()
		RMSD.append(rmsd)

		transform_coordinates(hier, transformation)
		IMP.rmf.save_frame(fh_out, str(i))

		del temp_coord

	del f

	print('Mean RMSD:', np.mean(np.array(RMSD)))


#####################
# MAIN
#####################
# IO files
dir_name = sys.argv[1]   # mc_tags_k1
cl = sys.argv[2]         # 2
st = sys.argv[3]         # 0
sel_state = 0

rmf_in_A = f'../output/{dir_name}/analysis/A_models_clust{cl}_{st}.rmf3'
rmf_out_A = f'../output/{dir_name}/analysis/A_models_clust{cl}_{st}_aligned.rmf3'
sel_frames_A = np.loadtxt(f'../output/{dir_name}/analysis/rmsd/cluster.0.sample_A.txt')
rmf_in_B = f'../output/{dir_name}/analysis/B_models_clust{cl}_{st}.rmf3'
rmf_out_B = f'../output/{dir_name}/analysis/B_models_clust{cl}_{st}_aligned.rmf3'
sel_frames_B = np.loadtxt(f'../output/{dir_name}/analysis/rmsd/cluster.0.sample_B.txt')

#################################
# Get reference and align
#################################
ref_rmf = f'../output/{dir_name}/analysis/rmsd/cluster.{cl}/cluster_center_model.rmf3'

reference_coordinates = get_reference_coordinates(ref_rmf)
num_frames_A = RMF.open_rmf_file_read_only(rmf_in_A).get_number_of_frames()
for i in range(len(sel_frames_B)):
	sel_frames_B[i] -= num_frames_A
align_rmf(rmf_in_A, rmf_out_A, sel_frames_A, reference_coordinates)
align_rmf(rmf_in_B, rmf_out_B, sel_frames_B, reference_coordinates)
