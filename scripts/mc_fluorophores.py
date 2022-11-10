#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Using IMP Rex MC optimization to trilaterate
the fluorophore positions
"""
import os
import sys
import pandas as pd

import IMP.algebra
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
# Data (input) directory where all the data files are located
data_directory = "../input/data/"
# Representation files
f_rep_file = data_directory + "fluorophore_representation.csv"
# Restraints files
pict_distances_tags = data_directory + "pict_restraints.csv"
# Output directory
out_dir_name = sys.argv[1]
output_directory = f"../output/{out_dir_name}/run"
# output prefix for each run
output_index = sys.argv[2]

if not os.path.exists(f"../output/{out_dir_name}"):
    os.mkdir(f"../output/{out_dir_name}")
# --------------------------
# 2. Scoring Parameters
# --------------------------

# --------------
# ----- Stereochemistry and Physical Restraints
ev_weight = 1.0           # Weight of excluded volume restraint (Don't change it!)
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

# Read fluorophores and exocyst representation data
f_data = pd.read_csv(f_rep_file, sep=',')       # fluorophores representation
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

root_hierarchy = system.build()
# Uncomment this for verbose output of the representation
# IMP.atom.show_with_representations(root_hierarchy)

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

# ##################### RESTRAINTS #####################
# -----------------------------------
# Define Scoring Function Components
# -----------------------------------
output_objects = list()
# -------------------------------------------------
# ----------PICT Simple Harmonic --------------
# -------------------------------------------------
dr_list = list()
# 'iterrows' is not convenient and should not be used with large data,
# though here we only have ~ 100 rows, and it is convenient
for row, col in pict_tags_df.iterrows():
    dr = IMP.pmi.restraints.basic.DistanceRestraint(root_hier=root_hierarchy,
                                                    tuple_selection1=(1, 1, col.bait),
                                                    tuple_selection2=(1, 1, col.prey),
                                                    distancemin=col.distance - col.serr_mu,
                                                    distancemax=col.distance + col.serr_mu,
                                                    kappa=3,
                                                    # IMP.core.Harmonic_get_k_from_standard_deviation(sd=col.sigma),
                                                    label="restraint_{}-{}".format(col.bait, col.prey))
    dr.add_to_model()
    dr.evaluate()
    dr_list.append(dr)
    output_objects.append(dr)
    print("\tSetting distance {} --> mu={}+-{} / sigma={}+-{}\n".format(dr.label, col.distance,
                                                                        col.serr_mu, col.sigma, col.serr_sigma))

# NO CONNECTIVITY RESTRAINT FOR TAGS/FLUOROPHORES

# -------------------------------------------------
# ----------EXCLUDED VOLUME --------------
# -------------------------------------------------
evr = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(included_objects=tag_molecules)
evr.set_weight(ev_weight)
evr.add_to_model()
output_objects.append(evr)

# ##################### SAMPLING #####################
# First shuffle the system
IMP.pmi.tools.shuffle_configuration(root_hierarchy,
                                    max_translation=30,
                                    verbose=True,
                                    niterations=100)

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
