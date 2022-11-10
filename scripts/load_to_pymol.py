#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python script to display structures in pymol: scaffold model and final 3D architecture.

Use PyMOL API command line to display and represent the beads of the scaffold and final exocyst in
PyMOL. When the module is called as script, user must specify if want to align different scaffold models
(pdb_align_scaffold method) or to represent the  architecture (display_architecture_pymol). In the
first case user must type "python load_to_pymol.py align_scaffold". In the second case, type
"python load_to_pymol.py display_architecture modelXX.pdb", if a specific model is gonna made display
in PyMOL.

For information about PyMOL commands in Python there are some links of interest:
Simple scripting --> https://pymolwiki.org/index.php/Simple_Scripting
Extend API-only function --> https://pymolwiki.org/index.php/Extend
"""
import os
import random
import datetime
from time import sleep
import sys
import __main__
import pymol
from pymol import cmd
from chempy import cpv  # to work with vectors in 3D with PyMOL (Chemical Python)
from statistics import mean, stdev
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import itertools
import csv

__main__.pymol_argv = ['pymol', '-qc']  # Quiet and no GUI (not loading PyMOL GUI)

__author__ = "Altair C. Hernandez"
__copyright__ = "Copyright 2020, The Exocyst Modeling Project"
__credits__ = ["Ibai Irastorza", "Damien P. Devos", "Oriol Gallego"]
__version__ = "1.0", "IMP 2.13.0"
__maintainer__ = "Altair C. Hernandez"
__email__ = "altairch95@gmail.com"
__status__ = "Development"

# DATA DICTIONARIES FOR TAGS AND SUBUNITS

pymol.finish_launching()  # necessary to import pymol modules

exocyst_chain_dict_tags = {
    "A": ["Sec3-FRB", "Sec3_GFP_C", "Sec3_GFP_N"],
    "B": ["Sec5_GFP_C", "Sec5_GFP_N"],  # there is no Sec5_FRB
    "C": ["Sec6-FRB", "Sec6_GFP_C", "Sec6_GFP_N"],
    "D": ["Sec8-FRB", "Sec8_GFP_C"],  # there is no Sec8-GFP_N in data
    "E": ["Sec10-FRB", "Sec10_GFP_C", "Sec10_GFP_N"],
    "F": ["Sec15-FRB", "Sec15_GFP_C", "Sec15_GFP_N"],
    "G": ["Exo70-FRB", "Exo70_GFP_C", "Exo70_GFP_N"],
    "H": ["Exo84-FRB", "Exo84_GFP_C", "Exo84_GFP_N"]
}

exocyst_chain_dict_architecture = {
    "A": ["Sec3_n", "Sec3_2", "Sec3_3", "Sec3_4", "Sec3_5", "Sec3_6", "Sec3_c"],
    "B": ["Sec5_n", "Sec5_2", "Sec5_3", "Sec5_4", "Sec5_5", "Sec5_c"],
    "C": ["Sec6_n", "Sec6_2", "Sec6_3", "Sec6_4", "Sec6_5", "Sec6_c"],
    "D": ["Sec8_n", "Sec8_2", "Sec8_3", "Sec8_4", "Sec8_5", "Sec8_c"],
    "E": ["Sec10_n", "Sec10_2", "Sec10_3", "Sec10_4", "Sec10_5", "Sec10_c"],
    "F": ["Sec15_n", "Sec15_2", "Sec15_3", "Sec15_4", "Sec15_5", "Sec15_c"],
    "G": ["Exo70_n", "Exo70_2", "Exo70_3", "Exo70_4", "Exo70_c"],
    "H": ["Exo84_n", "Exo84_2", "Exo84_3", "Exo84_c"]

}

cog_chain_dict_tags = {
    "A": ["Cog1-FRB"],  # only Cog1-FRB for Cog1 protein
    "B": ["Cog2-FRB", "Cog2_GFP_C", "Cog2_GFP_N"],
    "C": ["Cog3-FRB", "Cog3_GFP_C", "Cog3_GFP_N"],
    "D": ["Cog4-FRB", "Cog4_GFP_C", "Cog4_GFP_N"],
    "E": ["Cog5-FRB", "Cog5_GFP_C", "Cog5_GFP_N"],
    "F": ["Cog6-FRB", "Cog6_GFP_C", "Cog6_GFP_N"],
    "G": ["Cog7-FRB", "Cog7_GFP_C", "Cog7_GFP_N"],
    "H": ["Cog8-FRB", "Cog8_GFP_C"]  # there is no Cog8-GFP_N in data
}

cog_chain_dict_architecture = {
    "B": ["Cog2_n", "Cog2_c"],
    "C": ["Cog3_n", "Cog3_2", "Cog3_3", "Cog3_4", "Cog3_5", "Cog3_c"],
    "D": ["Cog4_n", "Cog4_2", "Cog4_3", "Cog4_4", "Cog4_5", "Cog4_6", "Cog4_c"],
    "E": ["Cog5_n", "Cog5_2", "Cog5_3", "Cog5_c"],
    "F": ["Cog6_n", "Cog6_2", "Cog6_3", "Cog6_4", "Cog6_5", "Cog6_6", "Cog6_c"],
    "G": ["Cog7_n", "Cog7_2", "Cog7_c"],
    "H": ["Cog8_n", "Cog8_2", "Cog8_3", "Cog8_4", "Cog8_4", "Cog8_c"]
}


def pdb_align_scaffold(protein_complex, path="", sphere_size=1, bg_color="white", com=False, custom_pdb=False):
    """
    This is a recurse that generate a pymol session in which all tag/exocyst models from a cluster (after
    the hierarchical cluster performed) are aligned(see hierarchical_clustering method in file exocyst_main.py).
    The cluster must be specified from the command line. The output session is stored in /output/tags/filter/pymol/
    :param protein_complex: exocyst or cog.
    :param path: Define path to output depending if is exocyst or cog complex.
    :param bg_color: back ground color of session (if not specified is white as default).
    :param sphere_size: integer/float to design sphere size (in Amstrongs)
    :param com: calculate center of mass of each tag distribution and write to file in PDB format.
    :param custom_pdb: if True it will save output in file with custom PDB format.
    """
    cluster = input("\nType which cluster you want to superimpose in pymol (1, 2, 3, ...): ")
    path_to_scaffold_files = path + "/tags/filter/pdb/cluster_{}".format(cluster)
    # Read User Input
    models = os.listdir(path_to_scaffold_files)  # models of cluster folder
    # Load Structures in pymol
    for model in models:
        model_path = "{}/{}".format(path_to_scaffold_files, model)
        cmd.load(model_path, model)
    sleep(0.5)  # it is advisable when working with pymol
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
    if protein_complex == "exocyst":
        cmd.color("marine", "chain A")
        cmd.color("orange", "chain B")
        cmd.color("yellow", "chain C")
        cmd.color("pink", "chain D")
        cmd.color("chocolate", "chain E")
        cmd.color("purple", "chain F")
        cmd.color("red", "chain G")
        cmd.color("green", "chain H")
        cmd.color("grey70", "chain I")  # if Sec 2
    elif protein_complex == "cog":
        cmd.color("marine", "chain A")
        cmd.color("red", "chain B")
        cmd.color("yellow", "chain C")
        cmd.color("blue", "chain D")
        cmd.color("grey40", "chain E")
        cmd.color("teal", "chain F")
        cmd.color("grey90", "chain G")
        cmd.color("orange", "chain H")
        cmd.color("grey70", "chain I")  # if Sec 2
    cmd.bg_color(bg_color)
    cmd.center("all", origin=1)  # Center the alignment in the origin (0, 0, 0)
    # Create objects for each cloud of tags (Sec3_FRB, Sec3_C, Sec3_N, ...)
    idx = 0
    tag_pt = list()
    tags_dictionary = dict()
    if protein_complex == "exocyst":
        tags_dictionary = exocyst_chain_dict_tags
    elif protein_complex == "cog":
        tags_dictionary = cog_chain_dict_tags
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
    # Calculate COM and write to file in PDB format:
    if com:
        print("Calculating COM per tag distribution")
        # Select tag-object names from pymol session
        tag_names = [name for name in cmd.get_names("all") if name.startswith("Sec") or name.startswith("Exo")]
        # Create com_particles that represents the center of mass for each tag cloud
        com_list = list()
        atom = 1
        for name in tag_names:
            com_name = "com_{}".format(name)
            cmd.pseudoatom(com_name, selection=name, vdw=1, color="white")
            cmd.show_as("spheres", selection=com_name)
            com_list.append(com_name)
        if custom_pdb:
            # Write custom PDB file
            path_to_save_session = path + "/tags/filter/pymol/"
            with open(path_to_save_session + "cluster{}_com_tags_custom.pdb".format(cluster), "w") as pdb:
                pdb.write("COM_POSITIONS\n")
                for chain in exocyst_chain_dict_tags:
                    for tag in exocyst_chain_dict_tags[chain]:
                        coord = [round(x, 3) for x in cmd.get_atom_coords(selection="com_{}".format(tag))]
                        pdb.write("{:>4}".format("ATOM") + "{:>7}".format(atom) + "{:>9}".format(
                            "CA LYS") + "{:>2}".format(chain) + "{:>4}".format(atom) + "{:12.3f}".format(
                            coord[0]) + "{:8.3f}".format(
                            coord[1]) + "{:8.3f}".format(coord[2]) + "{:>12}".format(
                            "1.00 36.92") + "{:>13}".format("C\n"))
                        atom += 1
                    pdb.write("TER\n")
                pdb.write("END")
        cmd.disable("all")
        cmd.enable("com_*")
        # Save session
        path_to_save_session = path + "/tags/filter/pymol/"
        print("Saving session to --> {}".format(path_to_save_session))
        if not os.path.isdir(path_to_save_session):
            os.mkdir(path_to_save_session)
            print("\nPymol directory has been successfully created! :)\n")
        cmd.save(path_to_save_session + "cluster{}_com_tags.pdb".format(cluster), "enabled")
    # Calculate s2 table distances for center of mass of each cloud as average:
    # write_s2_table = input("Write S2 Table distances?(y/n): ")
    write_s2_table = "n"
    if write_s2_table == "y":
        print("\nWriting scaffold_s2_table...\n")
        sleep(1)
        try:
            scaffold_s2_table(protein_complex=protein_complex, cluster=cluster, path=path)
            print("File created!\n")
        except pymol.CmdException as err:
            print("Pymol cmd execption: {}\nPossibly due to your PyMOL version".format(err))
    sleep(1)
    # Save session
    path_to_save_session = path + "/tags/filter/pymol/"
    print("Saving session to --> {}".format(path_to_save_session))
    if not os.path.isdir(path_to_save_session):
        os.mkdir(path_to_save_session)
        print("\nPymol directory has been successfully created! :)\n")
    # Export as pse session and save
    cmd.set("pse_export_version", value=0)
    cmd.save("{}".format(path_to_save_session + "{}_scaffold_cluster_{}.pse".format(datetime.date.today(), cluster)))
    # Get out!
    cmd.quit()


def scaffold_s2_table(protein_complex, cluster, path="", single=False):
    """
    Method to construct s2 table from Cell paper. It calculates center of mass of each
    tag and bait cloud of solutions and calculate distances.
    :param cluster: cluster we are working on
    :param protein_complex: exocyst or cog.
    :param path: Define path depending on protein complex: exocyst, cog.
    :param single: if True it can calculate ditances for a single PDB model.
    """
    # if "single == True" then capture model name for output file's name.
    model = ""
    if single:
        model_path = path + "tags/filter/"
        model = input("Write name of pdb scaffold model: ")  # file have to be in model_path
        cmd.load(model_path + model, object="scaffold")
        # Create objects for each cloud of tags (Sec3_FRB, Sec3_C, Sec3_N, ...)
        idx = 0
        tag_pt = list()
        tags_dict = dict()
        if p_complex == "exocyst":
            tags_dict = exocyst_chain_dict_tags
        elif p_complex == "cog":
            tags_dict = cog_chain_dict_tags
        for protein_tags in tags_dict.values():
            for tag in protein_tags:
                tag_pt.append(tag)
                idx += 1
                index = "id " + str(idx)
                cmd.select(name="sele", selection=index, quiet=1)
                cmd.create(name=tag, selection="sele")
    else:
        pass
    # Define path to output file and path to raw data (S3 Table).
    pymol_path = path + "/tags/filter/pymol"
    if not os.path.isdir(pymol_path):
        os.mkdir(pymol_path)
    pth_s2 = pymol_path + "/s2t_cluster{}.txt".format(cluster)
    if single:
        pth_s2 = pth_s2.replace(".txt", "_" + model + ".txt")
    pth_raw_s2 = "../input/mmc2.csv"
    # Append all raw distances to a list for a later process.
    raw_distance_dict = dict()
    with open(pth_raw_s2, "r") as cf:
        csv_reader = csv.reader(cf, delimiter=',')  # to parse the csv file delimited with commas.
        cf.readline()  # skip first line
        if protein_complex == "exocyst":
            for line in csv_reader:
                if "og" not in line[0] and "Sec2" not in line[1]:
                    raw_distance_dict["{}-{}".format(line[0].lower().replace("-", "_"), line[1].lower())] = \
                        [[round(float(line[2]) * 10, 3), round(float(line[3]) * 10, 3)]]
                else:
                    break
        elif protein_complex == "cog":
            for line in csv_reader:
                if "og" in line[0]:
                    tag = line[1]
                    if tag.endswith("GFP"):
                        tag = tag.replace("-3xGFP", "_GFP_C")
                    elif tag.endswith("GFPN"):
                        tag = tag.replace("-GFPN", "_GFP_N")
                    raw_distance_dict["{}-{}".format(line[0].lower().replace("-", "_"), tag.lower())] = \
                        [[round(float(line[2]) * 10, 3), round(float(line[3]) * 10, 3)]]
                else:
                    pass
    # Generate output file of S2 distances
    distances = list()
    with open(pth_s2, "w") as f:
        f.write("{:>9}{:>12}{:>23}{:>15}{:>20}{:>33}\n\n".format("Bait", "Prey", "Estimated dist(A)", "raw dist(A)",
                                                                 "estimated se(A)", "Fulfilled / Not Fulfilled"))
        all_tags = list()
        if protein_complex == "exocyst":
            all_tags = [name for name in cmd.get_names("all") if name.startswith("sec") or name.startswith("exo")]
        elif protein_complex == "cog":
            all_tags = [name for name in cmd.get_names("all") if name.startswith("cog")]
        centroid_gfp_list = list()
        centroid_frb_list = list()
        for name in all_tags:
            # Create centroid for each tag cloud
            centroid_name = "com_{}".format(name)
            cmd.pseudoatom(centroid_name, selection=name, vdw=1, color="white")
            cmd.show_as("spheres", selection=centroid_name)
            cmd.set("sphere_transparency", 0.5, selection=name)
            if "frb" not in centroid_name:
                centroid_gfp_list.append(centroid_name)
            else:
                centroid_frb_list.append(centroid_name)
        for tag in centroid_gfp_list:
            for frb in centroid_frb_list:
                pair = ""
                if protein_complex == "exocyst":
                    pair = "{}-{}".format(frb.replace("com_", "").replace("-", "_"), tag.replace("com_", ""))
                elif protein_complex == "cog":
                    pair = "{}-{}".format(frb.replace("com_", "").replace("-", "_"), tag.replace("com_", ""))
                if pair in raw_distance_dict:
                    # print(pair + "\n")
                    sleep(5)
                    frb_coord = [round(x, 3) for x in cmd.get_atom_coords(selection=frb)]  # frb coordinate
                    tag_coord = [round(x, 3) for x in cmd.get_atom_coords(selection=tag)]  # gfp tag coordinates
                    atom_dist = round(cpv.distance(tag_coord, frb_coord), 3)
                    raw_distance_dict[pair][0].append(atom_dist)
                    raw_distance_dict[pair].append([frb, tag])
                else:
                    pass
        for dist_list in raw_distance_dict.values():
            frb = dist_list[1][0]
            tag = dist_list[1][1]
            atom_dist = dist_list[0][2]
            raw_dist = dist_list[0][0]
            se = dist_list[0][1]
            if raw_dist - se < atom_dist < raw_dist + se:
                f.write("{:<13}{:>13}{:>12}{:18}{:>18}{:>22}\n".format(frb, tag, atom_dist, raw_dist, se, "F"))
            else:
                f.write("{:<13}{:>13}{:>12}{:18}{:>18}{:>22}\n".format(frb, tag, atom_dist, raw_dist, se, "NF"))
        f.write("\n\nNumber of distances = {}\n".format(len(raw_distance_dict)))
        f.write("Date: {}".format(datetime.date.today()))


def distances_architecture(model, path_to_file, protein_complex):
    """
    This method generates the S3 table as output file from Cell paper (Picco A, et al. 2017)
    from a given architecture PDB model. It also display the architecture and save the
    resulting pymol session.
    :param protein_complex: exocyst or cog.
    :param model: model we are working on.
    :param path_to_file: path to file.
    """
    # Define path to output file and path to raw data (S3 Table).
    pth_s3 = path_to_file + "S3table_{}.md".format(model.replace(".pdb", ""))
    pth_raw_s3 = "../input/mmc2_II.csv"
    # Append all raw distances to a list for a later processing.
    raw_distance_list = list()
    with open(pth_raw_s3, "r") as cf:
        csv_reader = csv.reader(cf, delimiter=',')  # to parse the csv file delimited with commas.
        cf.readline()  # skip first line
        if protein_complex == "exocyst":
            for line in csv_reader:
                if "og" not in line[0]:
                    raw_distance_list.append(line[2])
                else:
                    pass
        elif protein_complex == "cog":
            for line in csv_reader:
                if "og" in line[0]:
                    raw_distance_list.append(line[2])
                else:
                    pass
    # Generate output file of S3 distances
    architecture_distances = list()
    with open(pth_s3, "w") as f:
        f.write("{}{:>12}{:>13}{:>13}\n\n".format("Bead_1", "Bead_2", "Distance(A)", "Raw_Data_S2"))
        protein_names = list()
        if protein_complex == "exocyst":
            protein_names = [name for name in cmd.get_names("all") if name.startswith("Sec") or name.startswith("Exo")]
        elif protein_complex == "cog":
            protein_names = [name for name in cmd.get_names("all") if name.startswith("Cog")]
        for protein in protein_names:
            all_atoms = cmd.get_model(protein, 1).get_coord_list()
            atoms = list(all_atoms for all_atoms, _ in itertools.groupby(all_atoms))  # avoiding repeated atoms
            n_ter = atoms[0]
            c_ter = atoms[-4]
            if protein_complex == "exocyst":
                if protein == "Sec5_":
                    protein = protein.replace("_", "")
                    c_ter = atoms[-3]
                    gfp_c = atoms[-2]
                    gfp_n = atoms[-1]
                    distance_n = round(cpv.distance(n_ter, gfp_n), 3)
                    distance_c = round(cpv.distance(c_ter, gfp_c), 3)
                    f.write("{:<11}{:>8}_N{:>9}\n".format(protein + "_GFP_N", protein, distance_n))
                    f.write("{:<11}{:>8}_C{:>9}\n".format(protein + "_GFP_C", protein, distance_c))
                    architecture_distances.append(distance_n), architecture_distances.append(distance_c)
                elif protein == "Sec8_":
                    protein = protein.replace("_", "")
                    c_ter = atoms[-3]
                    frb = atoms[-2]
                    gfp_c = atoms[-1]
                    distance_c = round(cpv.distance(c_ter, gfp_c), 3)
                    distance_frb = round(cpv.distance(c_ter, frb), 3)
                    f.write("{:<11}{:>8}_C{:>9}\n".format(protein + "_GFP_C", protein, distance_c))
                    f.write("{:<11}{:>8}_C{:>9}\n".format(protein + "_RFP", protein, distance_frb))
                    architecture_distances.append(distance_c), architecture_distances.append(distance_frb)
                else:
                    protein = protein.replace("_", "")
                    frb = atoms[-3]
                    gfp_c = atoms[-2]
                    gfp_n = atoms[-1]
                    distance_n = round(cpv.distance(n_ter, gfp_n), 3)
                    distance_c = round(cpv.distance(c_ter, gfp_c), 3)
                    distance_frb = round(cpv.distance(c_ter, frb), 3)
                    f.write("{:<11}{:>8}_N{:>9}\n".format(protein + "_GFP_N", protein, distance_n))
                    f.write("{:<11}{:>8}_C{:>9}\n".format(protein + "_GFP_C", protein, distance_c))
                    f.write("{:<11}{:>8}_C{:>9}\n".format(protein + "_RFP", protein, distance_frb))
                    architecture_distances.append(distance_n), architecture_distances.append(distance_c)
                    architecture_distances.append(distance_frb)
            elif protein_complex == "cog":
                if protein == "Cog8_":
                    protein = protein.replace("_", "")
                    c_ter = atoms[-3]
                    gfp_c = atoms[-2]
                    gfp_n = atoms[-1]
                    distance_n = round(cpv.distance(n_ter, gfp_n), 3)
                    distance_c = round(cpv.distance(c_ter, gfp_c), 3)
                    f.write("{:<11}{:>8}_N{:>9}\n".format(protein + "_GFP_N", protein, distance_n))
                    f.write("{:<11}{:>8}_C{:>9}\n".format(protein + "_GFP_C", protein, distance_c))
                    architecture_distances.append(distance_n), architecture_distances.append(distance_c)
                else:
                    protein = protein.replace("_", "")
                    frb = atoms[-3]
                    gfp_c = atoms[-2]
                    gfp_n = atoms[-1]
                    distance_n = round(cpv.distance(n_ter, gfp_n), 3)
                    distance_c = round(cpv.distance(c_ter, gfp_c), 3)
                    distance_frb = round(cpv.distance(c_ter, frb), 3)
                    f.write("{:<11}{:>8}_N{:>9}\n".format(protein + "_GFP_N", protein, distance_n))
                    f.write("{:<11}{:>8}_C{:>9}\n".format(protein + "_GFP_C", protein, distance_c))
                    f.write("{:<11}{:>8}_C{:>9}\n".format(protein + "_RFP", protein, distance_frb))
                    architecture_distances.append(distance_n), architecture_distances.append(distance_c)
                    architecture_distances.append(distance_frb)
        for protein in protein_names:
            all_atoms = cmd.get_model(protein, 1).get_coord_list()
            atoms = list(all_atoms for all_atoms, _ in itertools.groupby(all_atoms))  # avoiding repeated atoms
            protein_atoms = list()
            if protein_complex == "exocyst":
                if protein == "Sec5_" or protein == "Sec8_":
                    protein_atoms += atoms[:len(atoms) - 2]
                else:
                    protein_atoms += atoms[:len(atoms) - 3]
            elif protein_complex == "cog":
                if protein == "Cog8_":
                    protein_atoms += atoms[:len(atoms) - 2]
                else:
                    protein_atoms += atoms[:len(atoms) - 3]
            protein = protein.replace("_", "")
            c = 0
            for atom in protein_atoms:
                at1 = atom
                at2 = protein_atoms[c + 1]
                distance_at1_at2 = round(cpv.distance(at1, at2), 3)
                architecture_distances.append(distance_at1_at2)
                if c == len(protein_atoms) - 2:
                    f.write("{:<11}{:>8}_C{:>9}\n".format(protein + "_" + str(c + 1), protein, distance_at1_at2))
                    break
                elif c == 0:
                    f.write("{:<11}{:>8}_{}{:>9}\n".format(protein + "_N", protein, c + 2, distance_at1_at2))
                else:
                    f.write(
                        "{:<11}{:>8}_{}{:>9}\n".format(protein + "_" + str(c + 1), protein, c + 2, distance_at1_at2))
                c += 1
        f.write("END")
    # Rewrite file and modify: writing raw data column from S3 table.
    compare_distances = list(zip(raw_distance_list, architecture_distances))
    temp_file = "S2_temp.txt"
    with open(pth_s3, "r+") as f, open(temp_file, "w") as tf:
        tf.write("{}{:>12}{:>13}{:>13}{:>6}\n\n".format("Bead_1", "Bead_2", "Distance(A)", "Raw_Data_S3", "F/NF"))
        # f.seek(0, 0)
        lines = f.readlines()[2:]  # all lines as a list of strings
        for line in lines:
            break_loop = False
            for raw in raw_distance_list:
                if break_loop:
                    break
                else:
                    for pair in compare_distances:
                        if float(pair[0]) > pair[1]:
                            tf.write(line.rstrip("\n") + "{:>13}{:>5}\n".format(raw, "F"))
                            raw_distance_list.pop(0)
                            compare_distances.pop(0)
                            break_loop = True
                            break
                        else:
                            tf.write(line.rstrip("\n") + "{:>13}{:>5}\n".format(raw, "NF"))
                            raw_distance_list.pop(0)
                            compare_distances.pop(0)
                            break_loop = True
                            break

        tf.write("\nLEGEND:\n\n")
        tf.write("**Bead_1 and Bead_2**: pairwise bead from which the distance is calculated using PyMOL.\n")
        tf.write("**Distance(A)**: pairwise distance in Angstroms, calculated using PyMOL.\n")
        tf.write("**Raw_Data_S3**: extracted from S3 Table of the Cell Paper (Picco A, et al. 2017).\n")
        tf.write(
            "**F/NF**: if the distance Fulfill or Not Fulfill the maximum distance restranint from the S3 table.\n")
        tf.write("\n\nGenerated on date {}".format(datetime.date.today()))
    os.remove(pth_s3)
    os.rename(temp_file, pth_s3)


def display_architecture_pymol(sphere_size, path="", bg_color="white", vdw=False, protein_complex=""):
    """
    This is a recurse to display a single architecture model of the exocyst (in pdb format) in pymol,
    and save the session in the folder pymol
    :param protein_complex: refeered to exocyst or cog.
    :param path: Define path to output depending if is exocyst or cog complex.
    :param vdw:
    :param bg_color: back ground color of session (if not specified is white as default).
    :param sphere_size: size of the spheres in Angstroms
    """
    # Read User Input
    model = "model" + input("\nType the model name to display in pymol ('1', '2', '3'..): ") + ".pdb"
    cluster = input("\nIn which cluster is this model? ('1','2','3'...) --> ")
    # Load Structures in pymol
    path_to_model = ""
    if protein_complex == "exocyst":
        path_to_model = path + "/3D_exocyst/filter/pdb/cluster_{}".format(cluster)
    elif protein_complex == "cog":
        path_to_model = path + "/3D_cog/filter/pdb/cluster_{}".format(cluster)
    # path_to_model = "../output/3D_Exocyst/test2/pdb/"
    model_path = "{}/{}".format(path_to_model, model)
    cmd.load(model_path, model)
    sleep(0.5)  # it is advisable when working with pymol
    # Display: spheres and colors for each chain
    cmd.hide("all")
    cmd.show_as("spheres")
    cmd.show("cartoon")  # show lines between consecutive beads
    cmd.set("sphere_scale", value=int(sphere_size))  # value regarding to sphere radius
    if protein_complex == "exocyst":
        cmd.color("marine", "chain A")
        cmd.color("orange", "chain B")
        cmd.color("yellow", "chain C")
        cmd.color("pink", "chain D")
        cmd.color("chocolate", "chain E")
        cmd.color("purple", "chain F")
        cmd.color("red", "chain G")
        cmd.color("green", "chain H")
        # Select N-ter and C-ter to display different
        cmd.select("N_ter", "resi 1+13+23+34+44+55+66+76")
        # cmd.color("white", "N_ter")
        cmd.set("sphere_transparency", 0.1, "N_ter")
        cmd.deselect()
        cmd.select("C_ter", "resi 7+18+28+39+49+60+70+79")
        # cmd.color("black", "C_ter")
        cmd.deselect()
    elif protein_complex == "cog":
        cmd.color("red", "chain B")
        cmd.color("yellow", "chain C")
        cmd.color("blue", "chain D")
        cmd.color("grey40", "chain E")
        cmd.color("teal", "chain F")
        cmd.color("grey70", "chain G")
        cmd.color("orange", "chain H")
        # Select N-ter and C-ter to display different
        cmd.select("N_ter", "resi 8+19+31+40+52+60")
        cmd.color("white", "N_ter")
        cmd.set("sphere_transparency", 0.1, "N_ter")
        cmd.deselect()
        cmd.select("C_ter", "resi 13+25+34+46+54+65")
        cmd.color("black", "C_ter")
        cmd.deselect()
    cmd.bg_color(bg_color)
    cmd.center("all", origin=1)  # Center the alignment in the origin (0, 0, 0)
    # Create objects for each chain (Sec3, Sec5, Sec6, ...)
    # Create heteroatoms object (as are in the PDB) corresponding to tags in scaffold
    cmd.select("sele", "hetatm")
    cmd.create("hetatms", "sele")
    cmd.set("sphere_scale", value=4, selection="hetatms")
    count = 0
    protein_dictionary_architecture = dict()
    if protein_complex == "exocyst":
        protein_dictionary_architecture = exocyst_chain_dict_architecture
    elif protein_complex == "cog":
        protein_dictionary_architecture = cog_chain_dict_architecture
    for chain, res_list in protein_dictionary_architecture.items():
        chain = "chain {}".format(chain)
        chain_name = res_list[0][:5]
        cmd.select(name="sele", selection=chain, quiet=1)
        cmd.create(name=chain_name, selection="sele")
        cmd.deselect()
    if protein_complex == "exocyst":
        cmd.select("sele", "resi 101-122")
        cmd.set("sphere_scale", 1, "sele")
        # Create exocyst object
        cmd.select(name="sele", selection="all", quiet=1)
        cmd.create(name="exocyst_complex", selection="sele")
    elif protein_complex == "cog":
        cmd.select("sele", "resi 101-120")
        cmd.set("sphere_scale", 4, "sele")
        # Create exocyst object
        cmd.select(name="sele", selection="all", quiet=1)
        cmd.create(name="cog_complex", selection="sele")
    if vdw:
        # Set Van der Waals sphere to 1
        cmd.alter("all", "vdw=1")
        cmd.rebuild("all")
    ####
    # Save session and image (optional)
    path_to_save_session = ""
    if protein_complex == "exocyst":
        path_to_save_session += "../output/exocyst/3D_exocyst/filter/pymol/"
    elif protein_complex == "cog":
        path_to_save_session += "../output/cog/3D_cog/filter/pymol/"
    print(
        "\n\nSaving session to {}\n\n".format(path_to_save_session))
    print("     ##############################   ")
    if not os.path.isdir(path_to_save_session):
        os.mkdir(path_to_save_session)
        print("pymol directory has been successfully created")
    cmd.set("pse_export_version", value=0)
    cmd.save("{}".format(path_to_save_session + "{}_{}.pse".format(model, datetime.date.today())))
    write_s3_table = input("Write S3 Architecture Table distances?(y/n): ")
    if write_s3_table == "y":
        try:
            # Calculate distances between beads and to tags in scaffold
            distances_architecture(model, path_to_file=path_to_save_session, protein_complex=protein_complex)
            print("File created!\n")
        except pymol.CmdException as err:
            print("Pymol cmd execption: {}\nPossibly due to your PyMOL version".format(err))
    sleep(1)
    # Get out!
    cmd.quit()


def align_3D(sphere_size, protein_complex, path, clusters, bg_color="white"):
    """
    This is a recurse that generate a pymol session in which all exocyst models from a cluster (after
    the hierarchical cluster performed) of the dendogram are aligned(see hierarchical_clustering
    method in file exocyst_main.py).
    :param path: Define path to output depending if is exocyst or cog complex.
    :param protein_complex: exocyst or cog
    :param clusters: clusters specified in input.
    :param sphere_size: sphere size to display
    :param bg_color: back ground color of session (if not specified is white as default).
    """
    cluster_ = ""
    path_to_models = ""
    models = list()
    # Accrue models first depending on the number of models to evaluate
    if type(clusters) is list:
        if len(clusters) == 1:
            cluster_ = str(clusters[0])
            path_to_models = path + "/3D_{}/filter/pdb/cluster_{}".format(protein_complex, cluster_)
            models = os.listdir(path_to_models)  # models of cluster folder
            # Load Structures in pymol
            for model in models:
                model_path = "{}/{}".format(path_to_models, model)
                cmd.load(model_path, model)
        elif len(clusters) > 1:
            for cluster_ in clusters:
                path_to_models = path + "/3D_{}/filter/pdb/cluster_{}".format(protein_complex, cluster_)
                models.append([path_to_models, os.listdir(path_to_models)])  # models of each cluster
            # Load Structures in pymol
            for cluster in models:
                for model in cluster[1]:
                    model_path = "{}/{}".format(cluster[0], model)
                    cmd.load(model_path, model)
    elif clusters == "all":
        path_to_models += path + "/3D_{}/filter/pdb".format(protein_complex)
        # load all pdb files
        for (dirpath, dirnames, filenames) in os.walk(path_to_models):
            if ".pdb" in filenames[0]:
                model_path = dirpath + os.sep
                models.append([model_path, filenames])  # models of cluster folder
        # Load Structures in pymol
        for cluster in models:
            for model in cluster[1]:
                model_path = cluster[0] + model
                cmd.load(model_path, model)
    # Working in PyMOL session
    sleep(0.5)  # it is advisable when working with pymol
    object_list = cmd.get_names("public_objects", enabled_only=1)  # store all objects in a object list
    target = random.choice(object_list)  # select one model randomly to align all to this */CA
    aln_object = 'aln_all_to_' + target  # create aln_all_to_target object
    # Align all models to target
    print("\nStarting tag_model alignment in pymol...\n")
    cmd.extra_fit("name ca", target, "align", object=aln_object)
    print("\nAlignment done!\n")
    # Display: spheres and colors for each chain
    cmd.hide("all")
    cmd.show_as("spheres", "name ca")
    cmd.show("cartoon")
    cmd.set("sphere_scale", value=int(sphere_size))  # value regarding to sphere size
    if protein_complex == "exocyst":
        cmd.color("marine", "chain A")
        cmd.color("orange", "chain B")
        cmd.color("yellow", "chain C")
        cmd.color("pink", "chain D")
        cmd.color("chocolate", "chain E")
        cmd.color("purple", "chain F")
        cmd.color("red", "chain G")
        cmd.color("green", "chain H")
        cmd.color("grey70", "chain I")
        # Select N-ter and C-ter to display different
        cmd.select("N_ter", "resi 1+13+23+34+44+55+66+76")
        # cmd.color("white", "N_ter")
        cmd.set("sphere_transparency", 0.1, "N_ter")
        cmd.deselect()
        cmd.select("C_ter", "resi 7+18+28+39+49+60+70+79")
        # cmd.color("black", "C_ter")
        cmd.bg_color("white")
        cmd.deselect()
    elif protein_complex == "cog":
        cmd.color("red", "chain B")
        cmd.color("yellow", "chain C")
        cmd.color("blue", "chain D")
        cmd.color("grey40", "chain E")
        cmd.color("teal", "chain F")
        cmd.color("grey70", "chain G")
        cmd.color("orange", "chain H")
        # Select N-ter and C-ter to display different
        cmd.select("N_ter", "resi 8+19+31+40+52+60")
        cmd.deselect()
        cmd.select("C_ter", "resi 13+25+34+46+54+65")
        cmd.deselect()
        cmd.bg_color("white")
    cmd.bg_color(bg_color)
    cmd.center("all", origin=1)  # Center the alignment in the origin (0, 0, 0)
    cmd.show("cartoon", "all")
    cmd.set("cartoon_tube_radius", 1.5)
    # Create objects for each chain (Sec3, Sec5, Sec6, ...)
    chains_names = cmd.get_chains()
    names = list()
    if protein_complex == "exocyst":
        names = ["sec3", "sec5", "sec6", "sec8", "sec10", "sec15", "exo70", "exo84"]
    elif protein_complex == "cog":
        names = ["cog2", "cog3", "cog4", "cog5", "cog6", "cog7", "cog8"]
    chains = list(zip(chains_names, names))
    for ch, name in chains:
        chain = "chain " + ch
        cmd.select(name="sele", selection=chain, quiet=1)
        cmd.create(name=name, selection="sele")
        cmd.show_as("cartoon", name)
        cmd.show_as("spheres", name)
    cmd.hide("cartoon spheres", "not name ca")  # hide tags from scaffold
    # Save session
    path_to_save_session = ""
    if protein_complex == "exocyst":
        path_to_save_session += path + "/3D_{}/filter/pymol/".format(protein_complex)
    elif protein_complex == "cog":
        path_to_save_session += path + "/3D_{}/filter/pymol/".format(protein_complex)
    print(
        "Saving session to folder {}".format(path_to_save_session))
    if not os.path.isdir(path_to_save_session):
        os.mkdir(path_to_save_session)
        print("pymol directory has been successfully created")
    cmd.set("pse_export_version", value=0)
    if type(clusters) is list:
        if len(clusters) == 1:
            cmd.save("{}".format(path_to_save_session + "{}superpose_{}.pse".format(datetime.date.today(), cluster_)))
        elif len(clusters) > 1:
            cl_names = ""
            for cluster in clusters:
                cl_names += str(cluster)
            cmd.save(
                "{}".format(path_to_save_session + "{}superpose_cls{}.pse".format(datetime.date.today(), cl_names)))
    else:
        cmd.save("{}".format(path_to_save_session + "{}superpose_all.pse".format(datetime.date.today())))
    # Get out!
    cmd.quit()


def calculate_tags_distribution(density_plot=False, tag_name=None):  # not finished
    """"""
    # Define path to pymol session, select and load
    path_to_session = "../output/tags/filter/pymol/"
    sessions = [session for session in os.listdir(path_to_session) if session.endswith(".pse")]
    pymol_session = sessions[0]
    cmd.load(path_to_session + pymol_session, object="scaffold")
    # Select tag-object names from pymol session
    tag_names = [name for name in cmd.get_names("all") if name.startswith("sec") or name.startswith("exo")]
    # For each tag calculate distances to a pseudoatom in centroid position and write a final file
    pseudoatom_list = list()
    for name in tag_names:
        # Create Pseudoatoms that represents the centroid for each tag cloud
        pseudoatom_name = "com_{}".format(name)
        cmd.pseudoatom(pseudoatom_name, selection=name, vdw=1, color="white")
        cmd.show_as("spheres", selection=pseudoatom_name)
        cmd.set("sphere_transparency", 0.5, selection=name)
        pseudoatom_list.append(pseudoatom_name)
        # For each tag cloud, calculate distances from all points to corresponding pseudoatom (centroid position)
        xyz_coord = cmd.get_model(name, 1).get_coord_list()  # Atom coordinate list
        ori = [round(x, 3) for x in cmd.get_atom_coords(selection=pseudoatom_name)]  # pseudoatom coordinates
        # Write a file with all distances to psudoatom, mean, sd and other settings
        path_to_distance_file = "../output/tags/filter/pymol/"
        with open(path_to_distance_file + name + "_distances.txt", "w") as f:
            f.write("Distances and calculated mean of {} population\n".format(name))
            f.write("ORIGIN: {}, {} \n".format(pseudoatom_name, ori))
            distance_list = list()
            for atom_xyz in xyz_coord:
                atom_xyz = [round(x, 3) for x in atom_xyz]
                atom_dist = round(cpv.distance(ori, atom_xyz), 3)
                f.write("{} - {} = {}\n".format(pseudoatom_name, atom_xyz, atom_dist))
                distance_list.append(atom_dist)
            mean_distance = round(mean(distance_list), 3)
            standard_deviation = round(stdev(distance_list), 3)
            maximum_dist = round(max(distance_list), 3)
            minimum_dist = round(min(distance_list), 3)
            f.write("\nMean = {}\nSd = {}\nMax distance = {}\nMin distance = {}"
                    .format(mean_distance, standard_deviation, maximum_dist, minimum_dist))
            if density_plot:
                # matplotlib histogram
                x = np.array(distance_list)
                range_values = maximum_dist - minimum_dist
                bandwidth = range_values / math.sqrt(len(x))
                # plt.hist(x, color='blue', edgecolor='black', bins=int(range_values / 5))
                sns.distplot(x, hist=False, kde=True, hist_kws={'edgecolor': 'black'},
                             kde_kws={'shade': True, 'linewidth': 2})
                # Add labels
                plt.title('Distribution of distances - {}'.format(name))
                plt.xlabel('Distance')
                plt.ylabel('Density')
        print("--> " + name + "_distances.txt created!!\n")
    # Save session
    path_to_save_session = "../output/tags/filter/pymol/distances/"
    print("Saving session at: \n--> {}\n".format(path_to_save_session))
    if not os.path.isdir(path_to_save_session):
        os.mkdir(path_to_save_session)
        print("\ndistances directory has been successfully created")
    cmd.set("pse_export_version", value=0)
    cmd.save("{}".format(path_to_save_session + "{}_{}.pse".format(datetime.date.today(),
                                                                   "modified")))
    for distance_file in os.listdir(path_to_session):
        if distance_file.endswith(".txt"):
            src = "../output/tags/filter/pymol/" + distance_file
            dst = "../output/tags/filter/pymol/distances/" + distance_file
            os.rename(src, dst)
    # Get out!
    cmd.quit()  # not fishi


if __name__ == "__main__":
    # CHOOSE PROTEIN COMPLEX
    # p_complex = input("\nWhich complex to deal with? (exocyst or cog): ")
    p_complex = "exocyst"
    # Define path to output folders for all functions (in output folder)
    path_to_output = "../output/"
    if p_complex == "exocyst":
        path_to_output += "exocyst"
    elif p_complex == "cog":
        path_to_output += "cog"
    # CHOOSE MODE #
    mode = input("\nType one of the following options: \n"
                 "'1' to display 'align_scaffold'\n"
                 "'2' to display 'single_model_architecture'\n"
                 "'3' to display 'superpose_3D_architecture'\n"
                 "'4' to display 'distribution of tags'\n"
                 "'5' to display 's2table_single_pdb'\n"
                 "'6' to display 'calculate distances to atom'\n"
                 "Type here: ")
    if mode == "1":
        pdb_align_scaffold(protein_complex=p_complex, path=path_to_output, sphere_size=1, bg_color="black")
    if mode == "2":
        display_architecture_pymol(sphere_size=17.5, vdw=True, path=path_to_output, protein_complex=p_complex)
    if mode == "3":
        input_clusters = input("\nChoose clusters (number of cluster / all) to superimpose: ")
        if input_clusters != "all":
            num_clusters = [int(s) for s in input_clusters.split() if s.isdigit()]
            align_3D(clusters=num_clusters, protein_complex=p_complex, path=path_to_output, sphere_size=1)
        elif input_clusters == "all":
            align_3D(clusters="all", protein_complex=p_complex, path=path_to_output, sphere_size=1)
    if mode == "4":
        calculate_tags_distribution()
    if mode == "5":
        scaffold_s2_table(protein_complex=p_complex, cluster="", single=True)
