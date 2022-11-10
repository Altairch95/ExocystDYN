
import IMP.algebra
import IMP.pmi.restraints.crosslinking
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.tools
import IMP.pmi.topology
import IMP.rmf
import os
import sys

#---------------------------
# Define Input Files
#---------------------------
data_directory = "../input/data/"
protein_representation = data_directory + "exocyst_representation.csv"
p_fasta = "5yfp.fasta"  # protein fasta file

# Initialize model, setup System and State
m = IMP.Model()
system = IMP.pmi.topology.System(m)
state = system.create_state()

# Defining bounding box 3D
bb_size = 500
bb = IMP.algebra.BoundingBox3D(IMP.algebra.Vector3D(-bb_size/2, -bb_size/2, -bb_size/2),
							   IMP.algebra.Vector3D(bb_size/2, bb_size/2, bb_size/2))

# The components of your system
components = ["protein1", "protein2"]
colors = ['medium purple', 'goldenrod']
chains = "AB"
beadsize = 10

molecules = list()
print("\nCreating proteins as IMP.pmi.Molecule:")
for n in range(len(components)):
	print('\tPMI: setting up tag ', components[n])
	molecule = state.create_molecule(name=components[n],
										  sequence=IMP.pmi.topology.Sequences(data_directory
																				 + p_fasta)[f"5YFP:{chains[n]}"],
										  chain_id=chains[n])
	atomic = molecule.add_structure(
		pdb_fn='your_pdb_file',
		chain_id=chains[n],
		res_range=['resx', 'resy'],
		ca_only=True,
		offset=0)
	molecule.add_representation(atomic,  # res 1,10 for structured regions
						   resolutions=[1, 10],
						   bead_ca_centers=True,
						   color=colors[n])
	molecule.add_representation(molecule[:] - atomic,  # res 10 for unstructured regions
						   resolutions=[beadsize],
						   bead_ca_centers=True,
						   color=colors[n])
	molecules.append(molecule)


root_hierarchy = system.build()
# Uncomment this for verbose output of the representation
# IMP.atom.show_with_representations(root_hierarchy)

# Setup degrees of freedom
#  The DOF functions automatically select all resolutions
#  Objects passed to nonrigid_parts move with the frame but also have
#  their own independent movers.
dof = IMP.pmi.dof.DegreesOfFreedom(m)
for mol in molecules:
	dof.create_rigid_body(mol,
					   nonrigid_parts=mol.get_non_atomic_residues(),
					   max_trans=0.1,
					   max_rot=0.78,
					   nonrigid_max_trans=0.1)
	# display the bonds between consecutive fragments,
	# so that they are shown in the psf
	IMP.pmi.tools.display_bonds(mol)

# Randomize the initial configuration before sampling, of only the molecules
# we are interested in (Rpb4 and Rpb7)
IMP.pmi.tools.shuffle_configuration(root_hierarchy,
                                    max_translation=50,
                                    verbose=False,
                                    cutoff=5.0,
                                    niterations=100)
#-----------------------------------
# Define Scoring Function Components
#-----------------------------------
output_objects = list()
# Here we are defining a number of restraints on our system.
#  For all of them we call add_to_model() so they are incorporated into scoring
#  We also add them to the outputobjects list, so they are reported in stat files

# Connectivity keeps things connected along the backbone (ignores if inside
# same rigid body)
mols = IMP.pmi.tools.get_molecules(root_hierarchy)
for mol in mols:
	molname=mol.get_name()
	IMP.pmi.tools.display_bonds(mol)
	cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol)
	cr.add_to_model()
	cr.set_label(molname)
	output_objects.append(cr)

# Excluded Volume Restraint
#  To speed up this expensive restraint, we operate it at resolution 20
ev = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(
                                         included_objects=root_hierarchy,
                                         resolution=10)
ev.add_to_model()
output_objects.append(ev)

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
xldb_dss.create_set_from_file(data_directory+'Exo.DSS.csv')

xls_dss = IMP.pmi.restraints.crosslinking.CrossLinkingMassSpectrometryRestraint(root_hier=root_hierarchy,
                                                                            CrossLinkDataBase=xldb_dss,
                                                                            length=21,
                                                                            label="XLS_dss",
                                                                            resolution=1.0,
                                                                            slope=0.02)


xls_dss.add_to_model()
output_objects.append(xls_dss)
dof.get_nuisances_from_restraint(xls_dss)
xls_dss.set_psi_is_sampled(True)
psi_dss = xls_dss.psi_dictionary["PSI"][0]
psi_dss.set_scale(0.1)
dof.get_nuisances_from_restraint(xls_dss) # needed to sample the nuisance particles (noise params)
###############################################

# Continue modeling ...

