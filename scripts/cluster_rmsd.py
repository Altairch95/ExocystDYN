#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function
import argparse
import time

import IMP
import IMP.pmi
import IMP.pmi.analysis
import IMP.pmi.output
import IMP.atom
import RMF

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import sys
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.spatial.distance import pdist, cdist
import os
from Bio.PDB import *


mpl.rcParams.update({'font.size': 8})
sys.path.append('/home/altair/PycharmProjects/UCSF_colab/PMI_analysis/pyext/src/')

#########
# PARSER
#########
p = argparse.ArgumentParser(
            description="Get PDB files from selected rmf3 files and run a hierarchical "
                        "clustering based on RMSD. "
                        "The resulting HC is saved as csv in the hc/ directory, as well "
                        "as the dendrogram. \n"
                        "Example of usage: cluster_rmsd.py -d  mc_tags -cl 2 -st 0"
                        )
p.add_argument('-d', action="store", dest="dir_name",
               help="directory name to process")
p.add_argument('-cl', action="store", dest="cluster",
               help="Specify cluster")
p.add_argument('-st', action="store", dest="state",
               help="Specify RMF state")
parsero = p.parse_args()


#############
# FUNCTIONS #
#############


def write_pdb(particles, representation_df, out_file):
    """
    Write molecule coordinates in PDB format
    ----------
    representation_df: dataframe wit the following columns:
    ['name','chain','color'] where:
        - name: name of IMP particle.
        - chain: PDB chain to be assigned.
        - color: (optional) for chimeraX visualization
    """
    with open(out_file, "w") as pdbf:
        for molecule in representation_df.itertuples():
            x, y, z = particles[molecule.name]
            pdbf.write("{:>4}{:>7}{:>9}{:>2}{:>4}{:12.3f}{:8.3f}{:8.3f}{:>12}{:>13}\n".format(
                "ATOM", molecule.Index + 1, "CA LYS", molecule.chain, molecule.Index + 1, x, y, z, "1.00 36.92", "C"
            ))


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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def rmf_to_pdb(rmf_file, num_frames, out_tmp_pdb):
    """
    Write PDB file from an RMF frame coordinate
    :param rmf_file: rmf file
    :param num_frames: number of frames in the rmf file
    :param out_tmp_pdb: name of output PDB file

    """
    # Reading RMF frames
    m = IMP.Model()
    for fr in num_frames[ran]:
        start = time.time()  # Tracking out time
        try:
            hier = IMP.pmi.analysis.get_hiers_from_rmf(m, fr, rmf_file)[0]  # Load hierarchy from RMF
        except:
            print('Missing frame')
            continue
        frame_particles = get_particles_at_lowest_resolution(hier)
        write_pdb(frame_particles, tag_representation, out_tmp_pdb + f'model{fr}.pdb')
        end = time.time()
        print(f'Frame {fr} checked in {end - start} s\n')
    print(f'Chunk {ran} done!\n')


def rmsd(models, save_file):
    """
    Calculate RMSD from a list of models
    """
    sup = Superimposer()
    rmsd_list = list()
    for s1 in models:
        temp_list = list()
        for s2 in models:
            if s1 == s2:
                temp_list.append(0)  # the rmsd for a model against itself is 0
            else:
                fixed_atoms = list(s1.get_atoms())
                moving_atoms = list(s2.get_atoms())
                sup.set_atoms(fixed_atoms, moving_atoms)  # translation
                sup.apply(moving_atoms)  # applying rotation matrix to superposition
                temp_list.append(sup.rms)
        rmsd_list.append(temp_list)
    m = np.array(rmsd_list)
    np.save(save_file, m)
    print(f'Saving RMSD matrix as {save_file}\n')
    return m


def rmsd_clustering(pdbs_dir, hc_dir, rmsd_m_file, out_clustering_csv, verbose=True):
    """
    RMSD-based clustering from selected custom PDBs
    """
    p = PDBParser(QUIET=True)
    if not os.path.exists(pdbs_dir):
        raise FileNotFoundError
    models = list()
    for model in os.listdir(pdbs_dir):
        pdb_file = f'{pdbs_dir}/{model}'
        models.append(p.get_structure(model, pdb_file))
    if verbose:
        print(f"\n\nClustering with {len(models)} models...\n")

    # Calculating RMSD
    if not os.path.exists(rmsd_m_file):
        print('Calculating RMSD matrix...\n')
        m = rmsd(models, rmsd_m_file)
    else:
        print(f'Loading RMSD matrix {rmsd_m_file}...\n')
        m = np.load(rmsd_m_file)
    if verbose:
        print(f'Mean RMSD = {np.mean(m[:, 0])}')

    # HIERARCHICAL CLUSTERING
    z = linkage(m, method='ward', metric='euclidean')
    if verbose:
        print("\nCophenetic correlation coefficient is {}".format(cophenet(z, pdist(m))[0]))

    # Write clustering assignments csv file
    assignments = fcluster(z,
                           t=120000,  # int(clusters_distance),
                           criterion='distance')  # threshold line to choose clusters
    labels = np.array([s.get_id() for s in models])
    pd.DataFrame({'models': labels, 'cluster': assignments}).to_csv(f"{hc_dir}/{out_clustering_csv}", index=False)

    return z


def plot_clustering(hc_dir, file, z):
    """
    Plot RMSD-based clustering (dendrogram)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('RMSD Clustering', fontdict={"fontsize": 20}, loc="center", pad=15.0)
    ax.set_xlabel('Models', fontdict={"fontsize": 20}, labelpad=10.0)
    ax.set_ylabel(ylabel="Distance", fontdict={"fontsize": 20}, labelpad=10.0)
    # Dendrogram object
    dendrogram(z,
               truncate_mode="lastp",
               p=6,
               leaf_font_size=15,
               show_contracted=True,
               get_leaves=True)
    plt.savefig(f'{hc_dir}/{file}', dpi=72)


#############
# MAIN ######
#############
dir_name = parsero.dir_name   # mc_tags_k1
cl = parsero.cl         # 2
st = parsero.st         # 0

data_directory = "../input/data/"
output_directory = f'../output/{dir_name}/hc/'
rmf_A = f'../output/{dir_name}/analysis/A_models_clust{cl}_{st}.rmf3'
rmf_B = f'../output/{dir_name}/analysis/B_models_clust{cl}_{st}.rmf3'

# Read representation for tags
tag_representation = pd.read_csv(data_directory + 'tags_representation.csv')

# Read number of frames in RMF
rhA = RMF.open_rmf_file_read_only(rmf_A)
rhB = RMF.open_rmf_file_read_only(rmf_B)
frames_A = np.arange(0, rhA.get_number_of_frames(), 1)
frames_B = np.arange(0, rhB.get_number_of_frames(), 1)
ranges_A = list(chunks(range(1, len(frames_A)), 250))
ranges_B = list(chunks(range(1, len(frames_B)), 250))
print('Frames A', len(frames_A))
print('Frames B', len(frames_B))

# Out PDBs tmp
tmp_pdbs = output_directory + 'tmp_pbds/'
tmp_pdbs_A = tmp_pdbs + 'models_A/'
tmp_pdbs_B = tmp_pdbs + 'models_B/'
if not os.path.exists(tmp_pdbs):
    os.mkdir(tmp_pdbs)

# PDBs for models in A and B
if not os.path.exists(tmp_pdbs_A):
    os.mkdir(tmp_pdbs_A)
if not os.path.exists(tmp_pdbs_B):
    os.mkdir(tmp_pdbs_B)

sample_A = True
sample_B = False
if sample_A:
    # Write PDBs from RMF frames in models A
    if len(os.listdir(tmp_pdbs_A)) != len(frames_A):
        for ran in ranges_A:
            rmf_to_pdb(rmf_A, frames_A, tmp_pdbs_A)

    # Do HC
    z = rmsd_clustering(tmp_pdbs_A, output_directory, tmp_pdbs + 'rmsd_A.npy', 'clustering_A.csv')
    plot_clustering(output_directory, 'rc_A.png', z)

if sample_B:
    # Write PDBs from RMF frames in models B
    if len(os.listdir(tmp_pdbs_B)) != len(frames_B):
        for ran in ranges_B:
            rmf_to_pdb(rmf_B, frames_B, tmp_pdbs_B)

    # Do HC
    z = rmsd_clustering(tmp_pdbs_B, output_directory, tmp_pdbs + 'rmsd_B.npy', 'clustering_B.csv')
    plot_clustering(output_directory, 'rc_B.png', z)

# DONE
print('Done!')
exit()








