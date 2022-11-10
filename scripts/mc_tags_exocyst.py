#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Using IMP Rex MC optimization to model the
exocyst subunits using the cryoEM, AlphaFold
predictions and the in vivo distances from
PICT.

Following the logic of mc_fluorophores.py
here we use the spatial information between
fluorophores and the known maximum distance
between each fluorophore and its corresponding
exocyst subunit termini (C-ter or N-ter) to model
the assembly of the exocyst from in vivo data.

---------------
Representation:
--------------
- Fluorophores as beads of 1 aa (LYS) --> fluorophore_representation.csv
- Exocyst subunits coarse-grained with resolution [1, 10]
using the cryoEM (Mei et al., 2018) to retrieve the structural
detail of each subunit --> exocyst_representation.csv

--------------
-- Restraints:
--------------
- Simple Harmonic distances between fluorophores --> pict_restraints.csv
    e.g,
       ANCHOR F              d1 +- err1      PREY1
    (bait-RFP-FKBP) --------------------  (prey-1-GFP)
        |          `
        |           `
        |  d2+-err2  `  d3 +- err3
        |             `
        PREY2           PREY3
    (prey-2-GFP)      (prey-3-GFP)

- Harmonic UpperBound distances between the exocyst subunits and its
corresponding fluorophore
    e.g,
      (EXOCYST SUBUNIT)  linker1 (max distance 1)   (FLUOROPHORE)
        Exo70 (C-ter)  ------------------------------- GFP
        OOO           `
      OOO  (AC-4 region)`   linker2 (max distance 2)
    OOO                   `
    O                        `
                              RFP (FLUOROPHORE)

- Connectivity between consecutive beads of the same subunit.

- Excluded volume between beads to avoid clashes.
"""
import os.path
import sys
import pandas as pd
import numpy
import IMP.atom
import IMP.algebra
import IMP.core
import IMP.pmi.dof
import IMP.pmi.macros
import IMP.pmi.restraints.basic
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.tools
import IMP.pmi.topology
import IMP.rmf

# ---------------------------
# 1. Define Input Data and
#       Output Directories
# ---------------------------
data_directory = "../input/data/"
f_rep_file = data_directory + "fluorophore_representation.csv"
exo_rep_file = data_directory + "exocyst_representation.csv"
pict_distances_tags = data_directory + "pict_restraints.csv"
pict_distances_exocyst = data_directory + "distance_restraints.csv"
output_directory = "../output/run_arch"
output_index = 'test' # sys.argv[1]  # N (number iterations )output prefix

# --------------------------
# 2. Scoring Parameters
# --------------------------

# --------------
# ----- Stereochemistry and Physical Restraints
ev_weight = 1.0  # Weight of excluded volume restraint (Don't change it!)
connectivity_scale = 1.0  # weight of Connectivity restraint    (Don't change it!)

# --------------------
# 3. Sampling Parameters
# --------------------
num_frames = 1000  # int(sys.argv[2])   # Number of frames in MC run
num_best_scoring_models = 1
num_mc_steps = 50  # Number of MC steps per frame
mc_temperature = 1.0  # Temperature for MC

# Replica Exchange (rex)
rex_temps = (1.0, 5.0)  # Temperature bounds for replica exchange

# ----------------
# ------------------------------
# MODELING
# ------------------------------
# ----------------

# Initialize model, setup System and State
m = IMP.Model()
system = IMP.pmi.topology.System(m)
state_tags = system.create_state()
# state_vesicle = system.create_state() # should we used it here??
state_exocyst = system.create_state()

# Read fluorophores and exocyst representation data
f_data = pd.read_csv(f_rep_file, sep=',')  # fluorophores representation
p_data = pd.read_csv(exo_rep_file, sep=',')  # protein representation
p_fasta = "5yfp.fasta"  # protein fasta file

# Read fluorophores and exocyst distance restraints
pict_tags_df = pd.read_csv(pict_distances_tags)
# convert nm to angstroms (comment if data already in angstroms!)
pict_tags_df.loc[:, ['distance', 'serr_mu', 'sigma', 'serr_sigma']] = pict_tags_df.copy().loc[:,
																	  ['distance', 'serr_mu',
																	   'sigma', 'serr_sigma']].apply(lambda x: x * 10)
max_distance = max(pict_tags_df.distance.to_list())
# Defining bounding box 3D based on the maximum distance allowed in the system
bb = IMP.algebra.BoundingBox3D(IMP.algebra.Vector3D(-max_distance / 2, -max_distance / 2, -max_distance / 2),
                               IMP.algebra.Vector3D(max_distance / 2, max_distance / 2, max_distance / 2))
# -------------
# TAG MOLECULES
# -------------
tag_molecules = list()
print("\nCreating Tag as IMP.pmi.Molecule:")
for f_mol in f_data.to_numpy():
	print('\tPMI: setting up tag ', f_mol[0])
	molecule = state_tags.create_molecule(name=f_mol[0],
										  sequence="K",
										  chain_id=f_mol[1])
	molecule.add_representation(color=f_mol[2],
								bead_default_coord=IMP.algebra.get_random_vector_in(bb),
								resolutions=[1])
	tag_molecules.append(molecule)

# ------------------
# EXOCYST MOLECULES
# ------------------
exocyst_molecules = list()
print("\nCreating Exocyst as IMP.pmi.Molecule:")
for p_mol, regions in list(p_data.groupby(['subunit_name', 'chain', 'color'], axis=0, as_index=False)):
	mol = p_mol[0]
	chain = p_mol[1]
	color = p_mol[2]
	print(f'PMI: setting up subunit {mol} -- chain {chain}')
	molecule = state_exocyst.create_molecule(name=mol,
											 sequence=IMP.pmi.topology.Sequences(data_directory
																				 + p_fasta)[f"5YFP:{chain}"],
											 chain_id=chain)
	for region in regions.to_numpy()[:, 1:]:
		pdb = region[0]
		mol_range = [int(r) for r in region[2].split('-')]
		resolution = [region[3]]
		if resolution == 1:
			resolution = [1, 10]
		if pdb.endswith('af.pdb'):
			chain = 'A'
		atomic = molecule.add_structure(pdb_fn=data_directory + pdb,
										chain_id=chain,
										res_range=[mol_range[0], mol_range[1]],
										soft_check=True,
										offset=0,
										ca_only=True)
		if atomic:
			molecule.add_representation(residues=atomic,
										color=color,
										bead_ca_centers=True,
										resolutions=resolution,
										bead_default_coord=IMP.algebra.get_random_vector_in(bb))
		if molecule[mol_range[0]:mol_range[1]] - atomic:
			molecule.add_representation(residues=molecule[mol_range[0]:mol_range[1]] - atomic,
										color=color,
										bead_ca_centers=True,
										resolutions=20,
										bead_default_coord=IMP.algebra.get_random_vector_in(bb))
	exocyst_molecules.append(molecule)

root_hierarchy = system.build()
# Uncomment this for verbose output of the representation
# IMP.atom.show_with_representations(root_hierarchy)

# Uncomment the next three lines to write PDB on this current state (system set for cryoEM)
# write a single-frame RMF
if not os.path.exists(output_directory):
	os.mkdir(output_directory)
out = IMP.pmi.output.Output()
# out.init_rmf(output_directory + "/previous_set.rmf3", hierarchies=[root_hierarchy])
# out.write_rmf(output_directory + "/previous_set.rmf3")
# out.init_pdb(output_directory + "/previous_set.pdb", prot=root_hierarchy)
# out.write_pdb(output_directory + "/previous_set.pdb")

# ------------------
# DEGREES OF FREEDOM
# ------------------
# Setup degrees of freedom and rigid bodies
dof = IMP.pmi.dof.DegreesOfFreedom(m)
for mol in tag_molecules:
	dof.create_rigid_body(mol,
						  nonrigid_parts=mol.get_non_atomic_residues(),
						  max_trans=4.0,
						  max_rot=0.5,
						  nonrigid_max_trans=0.1)
	IMP.pmi.tools.display_bonds(mol)

for mol in exocyst_molecules:
	atomic = mol.get_atomic_residues()
	non_atomic = mol.get_non_atomic_residues()
	for frags in p_data.loc[p_data.subunit_name == mol.get_name(), 'res_range'].to_list():
		frag = [int(r) for r in frags.split('-')]  # -1 because of the 0 index in IMP vs PDB index
		mol_frag = mol[frag[0] - 1:frag[1]]
		dof.create_rigid_body(mol_frag,
							  nonrigid_parts=mol_frag - atomic,
							  max_trans=4.0,
							  max_rot=0.5,
							  nonrigid_max_trans=0.1)
		IMP.pmi.tools.display_bonds(mol_frag)

# -------------------------------------------------
# ##################### RESTRAINTS ###############
# ------------------------------------------------
# Define Scoring Function Components
# -----------------------------------
output_objects = list()

# -------------------------------------------------
# ----------PICT Simple Harmonic --------------
# -------------------------------------------------
dr_list_tags = list()
for row, col in pict_tags_df.iterrows():
	dr = IMP.pmi.restraints.basic.DistanceRestraint(root_hier=root_hierarchy,
													tuple_selection1=(1, 1, col.bait),
													tuple_selection2=(1, 1, col.prey),
													distancemin=col.distance - col.serr_mu,
													distancemax=col.distance + col.serr_mu,
													kappa=IMP.core.Harmonic_get_k_from_standard_deviation(sd=col.sigma),
													label="restraint_{}-{}".format(col.bait, col.prey))
	dr.add_to_model()
	dr.evaluate()
	dr_list_tags.append(dr)
	output_objects.append(dr)
	print("\tSetting distance {} --> mu={}+-{} / sigma={}+-{}\n".format(dr.label, col.distance,
																		col.serr_mu, col.sigma, col.serr_sigma))

# -------------------------------------------------
# ----------Tag-to-termini Harmonic Upper Bound --------------
# -------------------------------------------------

# EXOCYST RESTRAINTS
pict_exocyst_df = pd.read_csv(pict_distances_exocyst)
# 'iterrows' is not convenient and should not be used with large data,
# though here we only have ~ 100 rows, and it is convenient
for row, col in pict_exocyst_df.iterrows():
	dr = IMP.pmi.restraints.basic.DistanceRestraint(root_hier=root_hierarchy,
													tuple_selection1=(col.tag_residue, col.tag_residue, col.tag),
													tuple_selection2=(col.protein_residue,
																	  col.protein_residue,
																	  col.protein),
													distancemin=0,
													distancemax=col.final_distance,
													kappa=1,
													label="restraint_{}-{}".format(col.tag_residue, col.protein))
	dr.add_to_model()
	dr.evaluate()
	dr_list_tags.append(dr)
	output_objects.append(dr)
	print("\tSetting max distance {} --> {}\n".format(dr.label, col.final_distance))

# -------------------------------------------------
# ----------Protein Connectivity --------------
# -------------------------------------------------
# NO CONNECTIVITY RESTRAINT FOR TAGS/FLUOROPHORES


# Connectivity keeps things connected along the backbone (ignores if inside
# same rigid body)
cr_list = list()
for mol in exocyst_molecules:
	mol_name = mol.get_name()
	IMP.pmi.tools.display_bonds(mol)
	cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol, scale=connectivity_scale)
	cr.add_to_model()
	cr.set_label(mol_name)
	output_objects.append(cr)
	cr_list.append(cr)

# -------------------------------------------------
# ----------EXCLUDED VOLUME --------------
# -------------------------------------------------
evr = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(included_objects=exocyst_molecules)
evr.set_weight(ev_weight)
evr.add_to_model()
output_objects.append(evr)

# ##################### SAMPLING #####################
# First shuffle the system
IMP.pmi.tools.shuffle_configuration(root_hierarchy,
									max_translation=30,
									verbose=True,
									niterations=300)

# Quickly move all flexible beads into place
dof.optimize_flexible_beads(100)

# Run replica exchange Monte Carlo sampling
rex = IMP.pmi.macros.ReplicaExchange0(m,
									  root_hier=root_hierarchy,
									  monte_carlo_sample_objects=dof.get_movers(),
									  output_objects=output_objects,
									  monte_carlo_temperature=mc_temperature,
									  replica_exchange_minimum_temperature=min(rex_temps),
									  replica_exchange_maximum_temperature=max(rex_temps),
									  monte_carlo_steps=num_mc_steps,
									  number_of_frames=num_frames,
									  number_of_best_scoring_models=num_best_scoring_models,
									  global_output_directory=output_directory + "_" + str(output_index))
# Start Sampling
rex.execute_macro()
