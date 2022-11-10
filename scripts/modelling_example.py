"""
Python file to integrate the cryoEM model of the exocyst together with the
live-cell distances gathered in situ.

The script uses the IMP modeling protocol (https://integrativemodeling.org/2.15.0/doc/ref/namespaceIMP_1_1pmi.html)
to integrate the cryoEM model of the exocyst (input as PDB) and the distances from prepositioned fluorophores
to the termini of the exocyst subunits, deduced from live-cell imaging in situ.

Data tree:

input/
    data/
        5yfp.pdb  --> cryoEM PDB model
        5YFP.fasta --> sequence of each exocyst subunit in FASTA format
        topology_wrb.txt  --> formatted text file from ehre IMP is setting up the system and degrees of freedom
        distance_restraints.csv  --> maximum distances from fluorophores to the termini of the exocyst subunits
        fluorophores_modified/  --> FOLDER wuth all the fluorophore positions as PDB models
                                    It is "modified" just because I removed the first line from the ones
                                    in the fluorophores/ folder, the direct output when calculating the
                                    fluorohpore positions.

scripts/
        modelling_wrb.py --> main script to run the modeling.
        functions.py  --> file with all the functions necessary to integrate the in situ distances as restraints.
        analysis.py --> file to run the analysis of the output from the modelling.

output/

"""
import sys
import os
import shutil
import random
from functions import create_tag_molecules, distance_restraints_pict_cryo

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
topology_file = data_directory + "topology_wrb.txt"
cryo_pdb_model = data_directory + "5yfp.pdb"
pict_pdb_models = data_directory + "fluorophores_modified/" 
dr_tags_file = data_directory + "distance_restraints.csv"
output_directory = "../output/cryoEM/output/iterations/wr"
output_index = sys.argv[1]  # N (number iterations )output prefix

# --------------------------
# 2. Scoring Parameters
# --------------------------

# --------------
# ----- Sterochemistry and Physical Restraints
ev_weight = 1.0  # Weight of excluded volume restraint
connectivity_scale = 1.0  # weight of Connectivity restraint

# --------------------
# 3. Sampling Parameters
# --------------------
num_frames = 30  # int(sys.argv[2])   # Number of frames in MC run
num_best_scoring_models = 3
num_mc_steps = 50  # Number of MC steps per frame
mc_temperature = 1.0  # Temperature for MC

# --- Simulated Annealing (sa)
#  - Alternates between two MC temperatures
sim_annealing = True  # If true, run simulated annealing
sa_min_temp_steps = 100  # Steps at min temp
sa_max_temp_steps = 20  # Steps at max temp
sa_temps = (1.0, 5.0)  # Sim annealing temperatures

# Replica Exchange (rex)
rex_temps = (1.0, 5.0)  # Temperature bounds for replica exchange

# ----------------
# ------------------------------
# Now, let's start modelling...
# ------------------------------
# ----------------

# Initialize model
m = IMP.Model()

# Read in the topology file --> to coarse model of exocyst
# Specify the directory where the PDB files and fasta files
topology = IMP.pmi.topology.TopologyReader(topology_file,
                                           pdb_dir=data_directory,
                                           fasta_dir=data_directory)

# Use the BuildSystem macro to build states from the topology file
bs = IMP.pmi.macros.BuildSystem(m)

# Define tag molecules (fluorophores) as chains and component names and load state
pict_pdb = random.choice(os.listdir(pict_pdb_models))  # get a tag model file randomly
pict_pdb_seqs = IMP.pmi.topology.PDBSequences(m, pict_pdb_models + pict_pdb)  # using IMP.model
pict_chains = [chain for chain in pict_pdb_seqs.sequences]
pict_components = ["tag_sec3", "tag_sec5", "tag_sec6", "tag_sec8", "tag_sec10", "tag_sec15", "tag_exo70", "tag_exo84"]
st_tags = bs.system.create_state()

# Each state can be specified by a topology file.
# Create tag molecules and load to system
tag_molecules = create_tag_molecules(pict_pdb_models + pict_pdb, pict_components, pict_chains, st_tags)
bs.add_state(topology)
system_molecules = bs.get_molecules()
cryo_components = [mol[1][0] for mol in system_molecules[1].items()]

# Build the system representation and degrees of freedom
root_hierarchy, dof = bs.execute_macro(max_rb_trans=4.0,
                                       max_rb_rot=0.3,
                                       max_bead_trans=4.0,
                                       max_srb_trans=4.0,
                                       max_srb_rot=0.3)

# Uncomment the next three lines to write PDB on this current state (system set for cryoEM)
# write a single-frame RMF to view the helix
# out = IMP.pmi.output.Output()
# out.init_pdb(output_directory + "previous_set.pdb", prot=root_hierarchy)
# out.write_pdb(output_directory + "previous_set.pdb")

cryo_rbs = dof.get_rigid_bodies()
# Define tag molecules as another rigid body --> later we will fix
# it not to move when shuffling
tags_rbs = dof.create_rigid_body(rigid_parts=tag_molecules,
                                 max_trans=4.0,  # max rigid body translation
                                 max_rot=0.3,  # max rigid body rotation
                                 nonrigid_max_trans=4.0,  # max steps for the nonrigid bead particles
                                 name="tags_rbs")

# Uncomment the next line to see the hierarchy structure
# IMP.atom.show_with_representations(root_hierarchy)

# Uncomment the next two lines to write PDB output in this current state of the system
# out.init_pdb(output_directory + "after_set.pdb", prot=root_hierarchy)
# out.write_pdb(output_directory + "after_set.pdb")

# Fix tags rigid bodies but not the exocyst (the protein complex)
fixed_beads, fixed_rbs = dof.disable_movers(tag_molecules,
                                            [IMP.core.RigidBodyMover,
                                             IMP.pmi.TransformMover])

# Randomize the initial configuration before sampling, of only the molecules
# we are interested in (exocyst subunits)
IMP.pmi.tools.shuffle_configuration(root_hierarchy,
                                    excluded_rigid_bodies=dof.get_rigid_bodies(),
                                    max_translation=50,
                                    verbose=False,
                                    cutoff=5.0,
                                    niterations=100)

# Uncomment these lines two write out PDB after shuffling the system
# out.init_pdb(output_directory + "after_shuff.pdb", prot=root_hierarchy)
# out.write_pdb(output_directory + "after_shuff.pdb")

out_objects = list()  # reporter objects (for stat files)

# -----------------------------------
# Define Scoring Function Components
# -----------------------------------

# Here we are defining a number of restraints on our system.
#  For all of them we call add_to_model() so they are incorporated into scoring
#  We also add them to the output_objects list, so they are reported in stat files

# Distances restraints from fluorophores to cryo subunits

# in functions.py
dr_tags_dict, dr_list, output_objects = distance_restraints_pict_cryo(dr_tags_file, root_hierarchy, out_objects)

# Distance restraints from cryoEM contact residues

# contact_residues = get_contact_residues_from_cryo(cryo_pdb_model, 5, "all")
# print("\n{} contact residues to set as restraints\n".format(len(contact_residues)))

# output_objects = set_contacts_restraints_from_cryo(contact_residues, output_objects, root_hierarchy, 8)

# Connectivity keeps things connected along the backbone (ignores if inside
# same rigid body)
all_molecules = IMP.pmi.tools.get_molecules(root_hierarchy)
cryo_molecules = [mol for mol in all_molecules if "tag" not in mol.get_name()]
cr_list = list()
for mol in cryo_molecules:
    mol_name = mol.get_name()
    IMP.pmi.tools.display_bonds(mol)
    cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol, scale=connectivity_scale)
    cr.add_to_model()
    cr.set_label(mol_name)
    output_objects.append(cr)
    cr_list.append(cr)

# Excluded Volume Restraint
#  To speed up this expensive restraint, we evaluate it at resolution 10
ev = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(included_objects=cryo_molecules,
                                                             resolution=10)
ev.set_weight(ev_weight)
ev.add_to_model()
output_objects.append(ev)

# Quickly move all flexible beads into place
dof.optimize_flexible_beads(nsteps=100)

# --------------------------
# Monte-Carlo Sampling
# --------------------------

# This object defines all components to be sampled as well as the sampling protocol
mc1 = IMP.pmi.macros.ReplicaExchange0(m,
                                      root_hier=root_hierarchy,
                                      monte_carlo_sample_objects=dof.get_movers(),
                                      output_objects=output_objects,
                                      rmf_output_objects=output_objects,
                                      monte_carlo_temperature=mc_temperature,
                                      simulated_annealing=sim_annealing,
                                      simulated_annealing_minimum_temperature=min(sa_temps),
                                      simulated_annealing_maximum_temperature=max(sa_temps),
                                      simulated_annealing_minimum_temperature_nframes=sa_min_temp_steps,
                                      simulated_annealing_maximum_temperature_nframes=sa_max_temp_steps,
                                      replica_exchange_minimum_temperature=min(rex_temps),
                                      replica_exchange_maximum_temperature=max(rex_temps),
                                      number_of_best_scoring_models=num_best_scoring_models,
                                      monte_carlo_steps=num_mc_steps,
                                      number_of_frames=num_frames,
                                      global_output_directory=output_directory + "_" + str(output_index))

# Start Sampling
mc1.execute_macro()

# Move tag model to output dir
src = pict_pdb_models + pict_pdb
dst = output_directory + "_" + str(output_index) + "/" + pict_pdb
shutil.copyfile(src, dst)

# Move outdir + index to outdir name
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)
if not os.path.isdir(output_directory + "/temp/"):  # create temp dir to save results before renaming
    os.mkdir(output_directory + "/temp/")
shutil.move(output_directory + "_" + str(output_index), output_directory + "/temp/")