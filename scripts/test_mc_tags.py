"""
Using IMP Rex MC optimization to trilaterate
the fluorophore positions
"""

import sys
import pandas as pd

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
pict_distances_tags = data_directory + "pict_test.csv"
pict_distances_exocyst = data_directory + "distance_tags_exocyst_test.csv"
output_directory = "../output/run_test"
output_index = "res10"  # N (number iterations )output prefix

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
num_frames = 100  # int(sys.argv[2])   # Number of frames in MC run
num_best_scoring_models = 10
num_mc_steps = 100  # Number of MC steps per frame
mc_temperature = 1.0  # Temperature for MC

# --- Simulated Annealing (sa)
#  - Alternates between two MC temperatures
sim_annealing = False  # If true, run simulated annealing
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

# Initialize model, setup System and State
m = IMP.Model()
system = IMP.pmi.topology.System(m)
state_tags = system.create_state()
state_exocyst = system.create_state()

# Create fluorophores as beads of r=1, random position, Seq = ALA,
# name in dictionary, color in list

residues = ["A", "K", "L", "Y", "I", "V", "R", "N", "D", "C"]
exocyst_chain_dict_tags = {
    "A": [["Sec3-FRB", "Sec3_GFP_C", "Sec3_GFP_N"], "blue"],
    "B": [["Sec5_GFP_C", "Sec5_GFP_N"], "orange"],  # there is no Sec5_FRB
    "C": [["Sec6-FRB", "Sec6_GFP_C", "Sec6_GFP_N"], "yellow"],
    "D": [["Sec8-FRB", "Sec8_GFP_C"], "pink"],  # there is no Sec8-GFP_N in data
    "E": [["Sec10-FRB", "Sec10_GFP_C", "Sec10_GFP_N"], "brown"],
    "F": [["Sec15-FRB", "Sec15_GFP_C", "Sec15_GFP_N"], "purple"],
    "G": [["Exo70-FRB", "Exo70_GFP_C", "Exo70_GFP_N"], "red"],
    "H": [["Exo84-FRB", "Exo84_GFP_C", "Exo84_GFP_N"], "green"],
    "I": [["Sec2_GFP_C"], "gray"]}

exocyst_chain_dict_architecture = {
    "A": [["Sec3", "blue"],
          [106, 233],  # PH DOMAIN PDB sec3_af.pdb
          [640, 710],  # COREX I   PDB 5yfp.pdb
          [740, 881],
          [920, 1035],
          [1040, 1208],
          [1213, 1332]],
    "B": [["Sec5", "orange"], [155, 231],  # COREX I PDB 5yfp.pdb
          [238, 331],
          [350, 619],
          [623, 812],
          [813, 936]],
    "C": [["Sec6", "yellow"], [18, 85],  # COREX I PDB 5yfp.pdb
          [92, 253],
          [297, 396],
          [414, 609],
          [619, 725]],
    "D": [["Sec8", "pink"], [91, 177],  # COREX I PDB 5yfp.pdb
          [180, 335],
          [338, 570],
          [588, 858],
          [874, 968]],
    "E": [["Sec10", "brown"], [58, 172],  # COREX II PDB 5yfp.pdb
          [191, 279],
          [317, 428],
          [432, 696],
          [697, 825]],
    "F": [["Sec15", "purple"], [78, 186],  # COREX II PDB 5yfp.pdb
          [190, 445],
          [453, 667],
          [668, 830],
          [838, 890]],
    "G": [["Exo70", "red"], [5, 67],  # COREX II PDB 5yfp.pdb
          [74, 190],
          [194, 340],
          [341, 513],
          [514, 623]],
    "H": [["Exo84", "green"],
          [197, 276],  # COREX II PDB 5yfp.pdb
          [346, 452],  # PH DOMAIN PDB exo84_af.pdb
          [527, 622],
          [625, 753]],
}

# Defining bounding box 3D
bb = IMP.algebra.BoundingBox3D(IMP.algebra.Vector3D(-250, -250, -250),
                               IMP.algebra.Vector3D(250, 250, 250))
tag_molecules = list()
colors = ["blue", "orange", "yellow", "pink", "brown", "purple", "red", "green"]
for chain, tags in exocyst_chain_dict_tags.items():
    for tag in tags[0]:
        print('PMI: setting up tag ', tag)

        molecule = state_tags.create_molecule(name=tag,
                                              sequence=residues[tags[0].index(tag)],
                                              chain_id=chain)
        molecule.add_representation(color=tags[1],
                                    bead_default_coord=IMP.algebra.get_random_vector_in(bb),
                                    resolutions=[1])
        tag_molecules.append(molecule)

# Exocyst molecules
exocyst_molecules = list()
for chain, subunit in exocyst_chain_dict_architecture.items():
    print('PMI: setting up subunit ', subunit)
    if subunit[0][0] == "Sec3":
        molecule = state_exocyst.create_molecule(name=f"{subunit[0][0]}",
                                                 sequence=IMP.pmi.topology.Sequences(data_directory
                                                                                     + "5yfp.fasta")[f"5YFP:{chain}"],
                                                 chain_id=f"{chain}")
        atomic = molecule.add_structure(pdb_fn=data_directory + "sec3_af.pdb",
                                        chain_id=f"{chain}",
                                        res_range=subunit[1],  # add only specific set of residues
                                        soft_check=True,  # warns sequence mismatches between fasta and PDB sequence
                                        offset=0,
                                        ca_only=True)  # to sync PDB residue numbering with FASTA numbering.
        molecule.add_representation(residues=atomic,  # adding all residues of PDB to representation
                                    bead_extra_breaks=[],  # Additional breakpoints for splitting beads.
                                    color=subunit[0][1],
                                    bead_ca_centers=True,  # resolution=1 beads to be at CA centers
                                    resolutions=[1, 10],
                                    bead_default_coord=IMP.algebra.get_random_vector_in(bb))
        for sub_ran in subunit[2:]:
            atomic = molecule.add_structure(pdb_fn=data_directory + "5yfp.pdb",
                                            chain_id=f"{chain}",
                                            res_range=sub_ran,
                                            soft_check=True,
                                            offset=0,
                                            ca_only=True)
            molecule.add_representation(residues=atomic,
                                        bead_extra_breaks=[],
                                        color=subunit[0][1],
                                        bead_ca_centers=True,
                                        resolutions=[1, 10],
                                        bead_default_coord=IMP.algebra.get_random_vector_in(bb))
        exocyst_molecules.append(molecule)
    else:
        molecule = state_exocyst.create_molecule(name=f"{subunit[0][0]}",
                                                 sequence=IMP.pmi.topology.Sequences(data_directory
                                                                                     + "5yfp.fasta")[f"5YFP:{chain}"],
                                                 chain_id=f"{chain}")

        for sub_ran in subunit[1:]:
            atomic = molecule.add_structure(pdb_fn=data_directory + "5yfp.pdb",
                                            chain_id=f"{chain}",
                                            res_range=sub_ran,
                                            soft_check=True,
                                            offset=0,
                                            ca_only=True)
            molecule.add_representation(residues=atomic,
                                        bead_extra_breaks=[],
                                        color=subunit[0][1],
                                        bead_ca_centers=True,
                                        resolutions=[1, 10],
                                        bead_default_coord=IMP.algebra.get_random_vector_in(bb))
            exocyst_molecules.append(molecule)


root_hierarchy = system.build()
# Uncomment this for verbose output of the representation
IMP.atom.show_with_representations(root_hierarchy)

# Setup degrees of freedom
dof = IMP.pmi.dof.DegreesOfFreedom(m)
for mol in tag_molecules:
    dof.create_rigid_body(mol,
                          nonrigid_parts=mol.get_non_atomic_residues(),
                          max_trans=4.0,
                          max_rot=0.5,
                          nonrigid_max_trans=0.1)
    IMP.pmi.tools.display_bonds(mol)
for mol in exocyst_molecules:
    dof.create_rigid_body(mol,
                          nonrigid_parts=mol.get_non_atomic_residues(),
                          max_trans=4.0,
                          max_rot=0.5,
                          nonrigid_max_trans=0.1)
    IMP.pmi.tools.display_bonds(mol)
# ##################### RESTRAINTS #####################
output_objects = []
# -----------------------------------
# Define Scoring Function Components
# -----------------------------------
# TAG RESTRAINTS
dr_list_tags = list()
pict_tags_df = pd.read_csv(pict_distances_tags)
# convert nm to angstrom
pict_tags_df.loc[:, ['distance', 'serr_mu', 'sigma', 'serr_sigma']] = pict_tags_df.copy().loc[:,
                                                                      ['distance', 'serr_mu',
                                                                       'sigma', 'serr_sigma']].apply(lambda x: x * 10)
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

# EXOCYST RESTRAINTS
pict_exocyst_df = pd.read_csv(pict_distances_exocyst)
# for row, col in pict_exocyst_df.iterrows():
#     dr = IMP.pmi.restraints.basic.DistanceRestraint(root_hier=root_hierarchy,
#                                                     tuple_selection1=(col.tag_residue, col.tag_residue, col.tag),
#                                                     tuple_selection2=(col.protein_residue,
#                                                                       col.protein_residue,
#                                                                       col.protein),
#                                                     distancemin=0,
#                                                     distancemax=col.final_distance,
#                                                     kappa=1,
#                                                     resolution=120,
#                                                     label="restraint_{}-{}".format(col.tag_residue, col.protein))
#     dr.add_to_model()
#     dr.evaluate()
#     dr_list_tags.append(dr)
#     output_objects.append(dr)
#     print("\tSetting max distance {} --> \n".format(dr.label, col.final_distance))

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

# Excluded volume
evr = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(included_objects=tag_molecules + exocyst_molecules)
evr.set_weight(ev_weight)
evr.add_to_model()
output_objects.append(evr)

# ##################### SAMPLING #####################
# First shuffle the system
IMP.pmi.tools.shuffle_configuration(root_hierarchy, max_translation=30, verbose=True, niterations=100)

# Quickly move all flexible beads into place
dof.optimize_flexible_beads(100)

# Run replica exchange Monte Carlo sampling
rex = IMP.pmi.macros.ReplicaExchange0(m,
                                      root_hier=root_hierarchy,
                                      monte_carlo_sample_objects=dof.get_movers(),
                                      global_output_directory=output_directory + "_" + str(output_index),
                                      output_objects=output_objects,
                                      monte_carlo_steps=num_mc_steps,
                                      number_of_best_scoring_models=num_best_scoring_models,
                                      number_of_frames=num_frames)
rex.execute_macro()
