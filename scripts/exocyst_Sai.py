import IMP
import IMP.core
import IMP.algebra
import IMP.atom
import IMP.container
import IMP.pmi.restraints.crosslinking
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.restraints.em
import IMP.pmi.restraints.basic
#import IMP.pmi.representation
import IMP.pmi.tools
import IMP.pmi.samplers
import IMP.pmi.output
import IMP.pmi.macros
import IMP.pmi.topology
import IMP.pmi.mmcif
import IMP.bayesianem
import IMP.bayesianem.restraint
import ihm.cross_linkers
import RMF
import IMP.rmf

import os
import sys


#---------------------------
# Define Input Files
#---------------------------
datadirectory = "../data/"
topology_file = datadirectory+"topology.txt"
target_gmm_file = datadirectory+'emd_21226.map.mrc.gmm.200.txt'

# Initialize model
m = IMP.Model()

# Read in the topology file.
# Specify the directory wheere the PDB files, fasta files and GMM files are
topology = IMP.pmi.topology.TopologyReader(topology_file,
                                  pdb_dir=datadirectory,
                                  fasta_dir=datadirectory,
                                  gmm_dir=datadirectory)

# Use the BuildSystem macro to build states from the topology file
bs = IMP.pmi.macros.BuildSystem(m)


bs.dry_run = '--dry-run' in sys.argv

# Each state can be specified by a topology file.
bs.add_state(topology)

# Build the system representation and degrees of freedom
root_hier, dof = bs.execute_macro(max_rb_trans=4.0,
                                  max_rb_rot=1.0,
                                  max_bead_trans=4.0,
                                  max_srb_trans=4.0,
                                  max_srb_rot=0.3)

# Fix all rigid bodies but not Rpb4 and Rpb7 (the stalk)
# First select and gather all particles to fix.
fixed_particles=[]
for prot in ["Sec03","Sec05","Sec06","Sec08","Sec10","Sec15","Exo70","Exo84"]:
    fixed_particles+=IMP.atom.Selection(root_hier,molecule=prot).get_selected_particles()


# Randomize the initial configuration before sampling, of only the molecules
# we are interested in (Rpb4 and Rpb7)
IMP.pmi.tools.shuffle_configuration(root_hier,
                                    #excluded_rigid_bodies=fixed_rbs,
                                    max_translation=50,
                                    verbose=False,
                                    cutoff=5.0,
                                    niterations=100)

outputobjects = [] # reporter objects (for stat files)

#-----------------------------------
# Define Scoring Function Components
#-----------------------------------

# Here we are defining a number of restraints on our system.
#  For all of them we call add_to_model() so they are incorporated into scoring
#  We also add them to the outputobjects list, so they are reported in stat files

# Connectivity keeps things connected along the backbone (ignores if inside
# same rigid body)
mols = IMP.pmi.tools.get_molecules(root_hier)
for mol in mols:
    molname=mol.get_name()
    IMP.pmi.tools.display_bonds(mol)
    cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol)
    cr.add_to_model()
    cr.set_label(molname)
    outputobjects.append(cr)

# Excluded Volume Restraint
#  To speed up this expensive restraint, we operate it at resolution 20
ev = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(
                                         included_objects=root_hier,
                                         resolution=10)
ev.add_to_model()
outputobjects.append(ev)


# Crosslinks - dataset 1
#  To use this restraint we have to first define the data format
#  Here assuming that it's a CSV file with column names that may need to change
#  Other options include the linker length and the slope (for nudging components together)
kw = IMP.pmi.io.crosslink.CrossLinkDataBaseKeywordsConverter()
kw.set_unique_id_key("id")
kw.set_protein1_key("protein1")
kw.set_protein2_key("protein2")
kw.set_residue1_key("residue1")
kw.set_residue2_key("residue2")
kw.set_id_score_key(None)

xldb_dss = IMP.pmi.io.crosslink.CrossLinkDataBase(kw)
xldb_dss.create_set_from_file(datadirectory+'Exo.DSS.csv')

xls_dss = IMP.pmi.restraints.crosslinking.CrossLinkingMassSpectrometryRestraint(root_hier=root_hier,
                                                                            CrossLinkDataBase=xldb_dss,
                                                                            length=21,
                                                                            label="XLS_dss",
                                                                            resolution=1.0,
                                                                            slope=0.02)


xls_dss.add_to_model()
outputobjects.append(xls_dss)
dof.get_nuisances_from_restraint(xls_dss)
xls_dss.set_psi_is_sampled(True)
psi_dss = xls_dss.psi_dictionary["PSI"][0]
psi_dss.set_scale(0.1)
dof.get_nuisances_from_restraint(xls_dss) # needed to sample the nuisance particles (noise params)
###############################################


xldb_edc= IMP.pmi.io.crosslink.CrossLinkDataBase(kw)
xldb_edc.create_set_from_file(datadirectory+'Exo.EDC.csv')

xls_edc = IMP.pmi.restraints.crosslinking.CrossLinkingMassSpectrometryRestraint(root_hier=root_hier,
                                                                            CrossLinkDataBase=xldb_edc,
                                                                            length=16,
                                                                            label="XLS_edc",
                                                                            resolution=1.0,
                                                                            slope=0.02)
xls_edc.add_to_model()
outputobjects.append(xls_edc)
dof.get_nuisances_from_restraint(xls_edc)
xls_edc.set_psi_is_sampled(True)
psi_edc = xls_edc.psi_dictionary["PSI"][0]
psi_edc.set_scale(0.1)
dof.get_nuisances_from_restraint(xls_edc) # needed to sample the nuisance particles (noise params)



# Electron Microscopy Restraint
#  The GaussianEMRestraint uses a density overlap function to compare model to data
#   First the EM map is approximated with a Gaussian Mixture Model (done separately)
#   Second, the components of the model are represented with Gaussians (forming the model GMM)
#   Other options: scale_to_target_mass ensures the total mass of model and map are identical
#                  slope: nudge model closer to map when far away
#                  weight: experimental, needed becaues the EM restraint is quasi-Bayesian

densities = IMP.atom.Selection(root_hier,representation_type=IMP.atom.DENSITIES).get_selected_particles()
gem = IMP.bayesianem.restraint.GaussianEMRestraintWrapper(densities,datadirectory+target_gmm_file,slope=0.0000001,target_radii_scale=3.0,scale_target_to_mass=True)

gem.set_label("EM")
gem.add_to_model()
outputobjects.append(gem)

#--------------------------
# Monte-Carlo Sampling
#--------------------------

#--------------------------
# Set MC Sampling Parameters
#--------------------------
num_frames = 200000
if '--test' in sys.argv:
    num_frames=5
num_mc_steps = 20

# This object defines all components to be sampled as well as the sampling protocol
mc1=IMP.pmi.macros.ReplicaExchange0(m,
                                    root_hier=root_hier,
                                    monte_carlo_sample_objects=dof.get_movers(),
                                    output_objects=outputobjects,
                                    crosslink_restraints=[xls_dss,xls_edc],    # allows XLs to be drawn in the RMF files
                                    monte_carlo_temperature=1.0,
                                    replica_exchange_minimum_temperature=1.0,
                                    replica_exchange_maximum_temperature=5.0,
                                    number_of_best_scoring_models=0,
                                    monte_carlo_steps=num_mc_steps,
                                    number_of_frames=num_frames,
                                    global_output_directory="output",
                                    test_mode=bs.dry_run)


# Start Sampling
mc1.execute_macro()


