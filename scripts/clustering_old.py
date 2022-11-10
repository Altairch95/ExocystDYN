import os
import glob
import shutil
from time import sleep
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from Bio.PDB import *
from scipy.cluster.hierarchy import dendrogram, cophenet, fcluster, linkage
from scipy.spatial.distance import pdist

from pymol import cmd


dict_tags = {
    "A": [["Sec3-FRB", "Sec3_GFP_C", "Sec3_GFP_N"], "blue"],
    "B": [["Sec5_GFP_C", "Sec5_GFP_N"], "orange"],  # there is no Sec5_FRB
    "C": [["Sec6-FRB", "Sec6_GFP_C", "Sec6_GFP_N"], "yellow"],
    "D": [["Sec8-FRB", "Sec8_GFP_C"], "pink"],  # there is no Sec8-GFP_N in data
    "E": [["Sec10-FRB", "Sec10_GFP_C", "Sec10_GFP_N"], "brown"],
    "F": [["Sec15-FRB", "Sec15_GFP_C", "Sec15_GFP_N"], "purple"],
    "G": [["Exo70-FRB", "Exo70_GFP_C", "Exo70_GFP_N"], "red"],
    "H": [["Exo84-FRB", "Exo84_GFP_C", "Exo84_GFP_N"], "green"],
    "I": [["Sec2_GFP_C"], "gray"]
}

def modify_pdb(path):
    mod_files = list()
    for file in glob.glob(path):
        if os.path.isfile(file):
            # res = sorted(ini_list, key=lambda x: x.split()[1])
            f = sorted(open(file, "r").readlines()[:-1], key=lambda x: x.split()[4]) + ['ENDMDL\n']
            c = 1
            if len(f) != 0:
                lines_mod = list()
                for line in f:
                    if line[17:20] == "LYS":
                        line = line[:24] + " 2" + line[26:]
                    if line[17:20] == "LEU":
                        line = line[:24] + " 3" + line[26:]
                    lines_mod.append(line)
                open(file, "w").write("".join(lines_mod))
                mod_files.append(file)
    return mod_files


def rmsd_matrix(structures_dict):
    # Calculating RMSD
    print("\n\tGettinh RMSD matrix...\n")
    rmsd_list = list()
    sup = Superimposer()
    used_structures = list()
    for s1 in structures_dict:
        used_structures.append(s1)
        temp_list = list()
        for s2 in structures_dict:
            if s1 == s2:
                temp_list.append(0)  # the rmsd for a model against itself is 0
            else:
                fixed_atoms = structures_dict[s1]
                moving_atoms = structures_dict[s2]
                sup.set_atoms(fixed_atoms, moving_atoms)  # translation
                sup.apply(moving_atoms)  # applying rotation matrix to superposition
                temp_list.append(sup.rms)
        rmsd_list.append(temp_list)
    return np.array(rmsd_list)


def hc(matrix, structures_dict, cluster_cutoff, out_dir):
    """

    :param matrix:
    :param structures_dict:
    :param cluster_cutoff:
    :param out_dir:
    :return:
    """
    labels = np.array([structure.get_id() for structure in structures_dict])  # labels for leaves in dendogram
    z = linkage(m, method=linkage_method, optimal_ordering=False)  # linkage matrix (distance matrix)
    c, coph_dists = cophenet(z, pdist(m))  # cophenetic correlation coefficient
    print("\nCophenetic correlation coefficient is {}".format(c))
    assignments = fcluster(z, int(cluster_cutoff), 'distance')  # threshold line to choose clusters
    cluster_output = pandas.DataFrame({'models': labels, 'cluster': assignments})  # convert cluster to df
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    cluster_output.to_csv(out_dir + "clustering.csv", sep='\t', encoding='utf-8')  # output to csv file
    groups = cluster_output.groupby('cluster')["models"].apply(list)  # group clusters and convert to lists
    cluster_dict = dict()  # store separated clusters in a dictionary
    for group in groups:
        for n in range(1, len(groups) + 1):
            if "cluster_{}".format(n) not in cluster_dict:
                cluster_dict.setdefault("cluster_{}".format(n), group)
                break

    plt.figure(figsize=(10, 7))
    plt.title('Hierarchical Clustering Dendogram', fontsize=20)
    plt.xlabel("Number of Models", fontsize=15)
    plt.ylabel("Euclidean Distance", fontsize=15)
    d = dendrogram(
        z,
        labels=None,
        # color_threshold=int(cluster_cutoff),
        # leaf_rotation=90,
        truncate_mode="lastp",
        leaf_font_size=15,
        p=6,
        show_contracted=True,
        get_leaves=True,
    )
    plt.savefig(out_dir + "tree.png")
    # Create heatmap
    # Seaborn heatmap beased on RMSD 2D-matrix with clusters on top
    hmp = sns.clustermap(
        data=matrix,
        method='ward',
        metric='euclidean',
        cmap="mako",
        figsize=(15, 15),
    )
    hmp.savefig(out_dir + "hm.png")
    return cluster_dict


def make_pymol(path, n_cluster, sphere_size=1):

    pdbs = os.listdir(path + "cluster_{}".format(n_cluster))  # models of cluster folder
    # Load Structures in pymol
    for pdb in pdbs:
        model_path = "{}/{}".format(path + "cluster_{}".format(n_cluster), pdb)
        cmd.load(model_path, model)
    sleep(1)
    # Prepare objects to do the alignment
    object_list = cmd.get_names("public_objects", enabled_only=1)  # store all objects in a object list
    target = random.choice(object_list)  # select one model randomly to align all to this */CA
    aln_object = 'aln_all_to_' + target  # create aln_all_to_target object
    # Align all models to target
    print("\nStarting tag_model alignment in pymol...\n")
    for model_object in object_list:
        if model_object != target:
            cmd.align("polymer and name CA and {}".format(model_object),
                      "polymer and name CA and {}".format(target),
                      quiet=0, object=aln_object)
    print("\nAlignment done!\n")
    # Display: spheres and colors for each chain according to Cell paper (Picco A et al, 2017).
    cmd.hide("all")
    cmd.show_as("spheres")
    cmd.set("sphere_scale", value=int(sphere_size))  # value referred to sphere size
    cmd.color("marine", "chain A")
    cmd.color("orange", "chain B")
    cmd.color("yellow", "chain C")
    cmd.color("pink", "chain D")
    cmd.color("chocolate", "chain E")
    cmd.color("purple", "chain F")
    cmd.color("red", "chain G")
    cmd.color("green", "chain H")
    cmd.color("grey70", "chain I")  # if Sec 2
    cmd.bg_color('black')
    cmd.center("all", origin=1)  # Center the alignment in the origin (0, 0, 0)
    # Create objects for each cloud of tags (Sec3_FRB, Sec3_C, Sec3_N, ...)
    idx = 0
    tag_pt = list()
    tags_dictionary = dict_tags
    for protein_tags in tags_dictionary.values():
        for tag in protein_tags:
            # tag = tag.lowe()
            tag_pt.append(tag)
            idx += 1
            index = "id " + str(idx)
            cmd.select(name="sele", selection=index, quiet=1)
            cmd.create(name=tag, selection="sele")
    # Create Scaffold object
    cmd.select(name="sele", selection="all", quiet=1)
    cmd.create(name="scaffold", selection="sele")
    # Export as pse session and save
    cmd.set("pse_export_version", value=0)
    cmd.save("{}".format(path + "pymol_{}.pse".format(cluster)))
    # Get out!
    cmd.quit()


if __name__ == "__main__":
    output_directory = "../output/run_"
    hc_directory = "../output/hc/"

    files = modify_pdb(output_directory + "*/pdbs/model.*.pdb")
    p = PDBParser()
    structures = dict()
    linkage_method = "ward"
    for f in files:
        name = f.split("/")[2]
        print(f)
        s = p.get_structure(f, f)  # Creating a BioPDB structure
        # Getting Atoms objects from each structure to pursue the RMSD.
        structures.setdefault(s, list(s.get_atoms()))
    if not os.path.exists(hc_directory + 'rmsd_m.npy'):
        m = rmsd_matrix(structures)
        np.save(hc_directory + 'rmsd_m.npy', m)
    else:
        m = np.load(hc_directory + 'rmsd_m.npy')
    clusters = hc(m, structures, 4000, hc_directory)

    for cluster, model_list in clusters.items():
        if not os.path.isdir(hc_directory + "/{}".format(cluster)):
            os.mkdir(hc_directory + "/{}".format(cluster))
        for model in model_list:
            if os.path.isfile(model):
                new_name = model.split("/")[2] + "_" + model.split("/")[-1]
                shutil.copyfile(model, hc_directory + "/{}/{}".format(cluster, new_name))

    make_pymol(hc_directory, 1)
    make_pymol(hc_directory, 2)
