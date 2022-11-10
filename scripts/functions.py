#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python functions to model the protein complexes (exocyst and cog e.g.).

Modeling is performed from distance measurements between subunits of the complex.
Distances are read from a csv file, create IMP particles, IMP
decorators (coordinates), assign initial coordinates and apply distance restraints to build
up the tag model or 3D architecture of the protein complex.
"""

import csv
import datetime
import glob
import os
import pwd
import random
import subprocess
from itertools import chain
import time

import IMP
import IMP.algebra
import IMP.atom
import IMP.container
import IMP.core
import IMP.display
import IMP.pmi.dof
import IMP.pmi.macros
import IMP.pmi.output
import IMP.pmi.restraints.basic
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.tools
import IMP.pmi.topology
import IMP.rmf
# import fastcluster
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from Bio.PDB import *
from scipy.cluster.hierarchy import dendrogram, cophenet, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering

sns.set(color_codes=True)

__author__ = "Altair C. Hernandez"
__copyright__ = "Copyright 2020, The Exocyst Modeling Project"
__credits__ = ["Ibai Irastorza", "Damien P. Devos", "Oriol Gallego"]
__version__ = "1.0", "IMP 2.13.0"
__maintainer__ = "Altair C. Hernandez"
__email__ = "altairch95@gmail.com"
__status__ = "Development"

# DATA DICTIONARIES FOR TAGS AND SUBUNITS (needed to write PDB outputs).
exocyst_chain_dict_tags = {
    "A": ["Sec3-FRB", "Sec3_GFP_C", "Sec3_GFP_N"],
    "B": ["Sec5_GFP_C", "Sec5_GFP_N"],  # there is no Sec5_FRB
    "C": ["Sec6-FRB", "Sec6_GFP_C", "Sec6_GFP_N"],
    "D": ["Sec8-FRB", "Sec8_GFP_C"],  # there is no Sec8-GFP_N in data
    "E": ["Sec10-FRB", "Sec10_GFP_C", "Sec10_GFP_N"],
    "F": ["Sec15-FRB", "Sec15_GFP_C", "Sec15_GFP_N"],
    "G": ["Exo70-FRB", "Exo70_GFP_C", "Exo70_GFP_N"],
    "H": ["Exo84-FRB", "Exo84_GFP_C", "Exo84_GFP_N"],
    "I": ["Sec2_GFP_C"]
}

exocyst_chain_dict_architecture = {
    "A": ["Sec3_n", "Sec3_2", "Sec3_3", "Sec3_4", "Sec3_5", "Sec3_6", "Sec3_c"],
    "B": ["Sec5_n", "Sec5_2", "Sec5_3", "Sec5_4", "Sec5_5", "Sec5_c"],
    "C": ["Sec6_n", "Sec6_2", "Sec6_3", "Sec6_4", "Sec6_5", "Sec6_c"],
    "D": ["Sec8_n", "Sec8_2", "Sec8_3", "Sec8_4", "Sec8_5", "Sec8_c"],
    "E": ["Sec10_n", "Sec10_2", "Sec10_3", "Sec10_4", "Sec10_5", "Sec10_c"],
    "F": ["Sec15_n", "Sec15_2", "Sec15_3", "Sec15_4", "Sec15_5", "Sec15_c"],
    "G": ["Exo70_n", "Exo70_2", "Exo70_3", "Exo70_4", "Exo70_c"],
    "H": ["Exo84_n", "Exo84_2", "Exo84_3", "Exo84_c"],
    "I": ["Sec2_GFP_C"]

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


def get_protein_input(prompt):
    """
    :param prompt:
    :return:
    """
    protein_complex = input("\n" + prompt)
    if protein_complex == "exocyst" or protein_complex == "cog":
        return protein_complex
    elif protein_complex == "q":
        quit()
    else:
        print("\n'{}' protein not recognized, try again or 'q' to quit!\n".format(protein_complex))
        return get_protein_input(prompt)


def get_option_input(prompt):
    """"""
    try:
        option = int(input(prompt))
    except ValueError:
        print("\nSorry, it has to be a number, try again!")
        get_option_input(prompt)
    else:
        if int(option) not in (1, 2, 3):
            print("\nNot an appropiate choice. Try again!")
            get_option_input(prompt)
        else:
            return option


def data_from_csv(csv_file, tags=True, cog=False):
    """
    The function reads data from a csv file and returns a dictionary with the data structured as:
    "Anchor_name" : [["Tag_name", Distance, Standard Deviation], etc.]
    :param tags: (bool, default True) if True, it works for tags modeling --> modeling part I.
                        if set to False, it works for the Exocyst modeling --> modeling part II.
    :param csv_file: Experimental data separated with ","
    :param cog: if True it will model the COG complex (False as default)
    :return: main_dictionary
    """
    with open(csv_file, "r") as csvf:
        csv_reader = csv.reader(csvf, delimiter=',')  # to parse the csv file delimited with commas.
        csvf.readline()  # skip first line
        my_dict = dict()
        if tags:
            # We can model exocyst or cog complex
            if cog:
                for line in csv_reader:
                    if "-FRB" in line[0] and line[0].startswith("Cog"):  # 22 elements and 84 measurements
                        anchor = line[0]
                        tag = line[1]
                        if tag.endswith("GFP"):
                            tag = tag.replace("-3xGFP", "_GFP_C")
                        elif tag.endswith("GFPN"):
                            tag = tag.replace("-GFPN", "_GFP_N")
                        distance = "{:.3f}".format(float(line[2]) * 10)  # in Angstroms
                        sd = "{:.3f}".format(float(line[3]) * 10)  # in Angstroms
                        if anchor not in my_dict:
                            my_dict.setdefault(anchor, [[tag, distance, sd]])
                        else:
                            my_dict[anchor].append([tag, distance, sd])
                    if not line[0].strip():  # if empty line, break out of the loop
                        break
            else:  # exocyst
                for line in csv_reader:
                    if "-FRB" in line[0] and not line[0].startswith("Cog"):  # 22 elements and 80 measurements (no
                        # sec2) // 23 elements and 86 measurements
                        anchor = line[0]
                        tag = line[1]
                        distance = "{:.3f}".format(float(line[2]) * 10)  # in Angstroms
                        sd = "{:.3f}".format(float(line[3]) * 10)  # in Angstroms
                        if anchor not in my_dict:
                            my_dict.setdefault(anchor, [[tag, distance, sd]])
                        else:
                            my_dict[anchor].append([tag, distance, sd])
                    if not line[0].strip():  # if empty line, break out of the loop
                        break

        else:  # to make 3D architecture
            if not cog:
                for line in csv_reader:
                    if line[0] in ['\n', '\r\n']:
                        break
                    if "cog" and "cog".capitalize() not in line[1]:
                        ter = str(line[1]).capitalize()
                        if ter not in my_dict:
                            my_dict.setdefault(ter, [[line[0].capitalize(), line[2]]])
                        else:
                            my_dict[ter].append([line[0].capitalize(), line[2]])
                my_dict.setdefault('Sec8_n', [['Sec8_2', '75']])
            elif cog:
                for line in csv_reader:
                    if line[0] in ['\n', '\r\n']:
                        break
                    if "cog" and "cog".capitalize() in line[1]:
                        ter = str(line[1]).capitalize()
                        if line[0].endswith("-RFP"):
                            line[0] = line[0].replace("RFP", "FRB")
                        else:
                            pass
                        if ter not in my_dict:
                            my_dict.setdefault(ter, [[line[0].capitalize(), line[2]]])
                        else:
                            my_dict[ter].append([line[0].capitalize(), line[2]])
                my_dict.setdefault('Cog8_n', [['Cog8_2', '75']])
    return my_dict


def create_particles(exp_dictionary, model, tags=True):
    """
    Create IMP Particles
    :param model: IMP model.
    :param tags: see function data_from_csv.
    :param exp_dictionary: dictionary with all the experimental data (anchors, tags, distances, standard deviation)
    :return: dictionary with IMP particles
    """
    # Create dictionary with defined PARTICLES as IMP models.
    particles_dict = dict()
    if tags:
        for anchor, lists in exp_dictionary.items():
            particles_dict["{0}".format(anchor)] = [IMP.Particle(model, anchor)]
            for tag in lists:
                tag_particle = tag[0]
                particles_dict["{0}".format(tag_particle)] = [IMP.Particle(model, tag_particle)]  # IMP particles have
                # the same name as anchors/tags
    else:
        # Create dictionary with defined PARTICLES as IMP models.
        for val in exp_dictionary:
            particles_dict["{0}".format(val)] = [IMP.Particle(model, val)]  # IMP particles have the same name as keys
    return particles_dict


def particles_with_coordinates(particles_dictionary, radius, tags=True):
    """
    Create IMP Decorators, Sphere like-particles, with coordinates XYZR
    :param tags: see function data_from_csv.
    :param particles_dictionary: dictionary with IMP particles
    :param radius: given radius of the spheres
    :return: dictionary with IMP Decorators
    """
    # Defining bounding box 3D
    bb = IMP.algebra.BoundingBox3D(IMP.algebra.Vector3D(-1000, -1000, -1000),
                                   IMP.algebra.Vector3D(1000, 1000, 1000))
    particles = set()
    if tags:
        for p in particles_dictionary.keys():
            particle = particles_dictionary[p][0]
            d = IMP.core.XYZR.setup_particle(particle,
                                             IMP.algebra.Sphere3D(IMP.algebra.get_random_vector_in(bb),
                                                                  float(radius)))
            d.set_coordinates_are_optimized(True)
            particles_dictionary[p].append(d)  # adding IMP decorators (XYZR) to particles_dictionary
            particles.add(particle)
        return list(particles)
    else:
        for p in particles_dictionary.keys():
            particle = particles_dictionary[p][0]
            d = IMP.core.XYZR.setup_particle(particle,
                                             IMP.algebra.Sphere3D(IMP.algebra.get_random_vector_in(bb),
                                                                  float(radius)))
            d.set_coordinates_are_optimized(True)
            particles_dictionary[p].append(d)  # adding IMP decorators (XYZR) to particles_dictionary
            particles.add(particle)
        return list(particles)


def create_restrains(model, data_dictionary, particles_dictionary, tags=True):
    """
    Define restrains to IMP particles of the model in the 3D Space
    :param tags: see function data_from_csv.
    :param model: IMP model defined previously.
    :param data_dictionary: main dictionary with experimental data
    :param particles_dictionary: with IMP particles
    :return: dictionary with restrains in the following output:

        restrain_dictionary =
         {
            restrain(x) : [[IMP_restrain, bait_particle(y), tag_particle(z), bait_decorator(y), bait_decorator(z)],
            [bait_name(y), tag_name(z)], [SD, experimental_distance]]
            }
    """
    restrain_dict = dict()
    for anchor, tag_list in data_dictionary.items():
        for particle in particles_dictionary:
            bait = particles_dictionary[anchor][0]
            d_bait = particles_dictionary[anchor][1]
            for data in tag_list:
                tag = particles_dictionary[data[0]][0]
                d_tag = particles_dictionary[data[0]][1]
                distance = float(data[1])
                sd = float(data[2])
                k = IMP.core.Harmonic_get_k_from_standard_deviation(sd=sd)
                f = IMP.core.Harmonic(distance, k)
                s = IMP.core.DistancePairScore(f)
                r = IMP.core.PairRestraint(model, s, (bait, tag))
                r.set_name("{} - {}".format(bait, tag))
                restrain_dict["{0}".format(r)] = [r, [bait, tag],
                                                  [d_bait, d_tag],
                                                  [sd, distance]]
            break
    return restrain_dict


def calculate_sf(restrain_dictionary, *args, tags=True):
    """
    Perform IMP scoring function of restrains.
    :param tags:
    :param restrain_dictionary.
    :return: Scoring Function.
    """
    if tags:
        # Create a set with all the restrains
        restrain_set = set()
        for restrain in restrain_dictionary.values():
            restrain_set.add(restrain[0])
        scoring_function = IMP.core.RestraintsScoringFunction(list(restrain_set), "scoring function")
        return scoring_function
    else:
        if args:  # we are passing the exclude volume restraint as an argument
            restraint_set = args[0]
            scoring_function = IMP.core.RestraintsScoringFunction(list(restraint_set), "scoring function")
            return scoring_function


def conjugate_gradient_optimization(model_to_optimize, scoring_function, num_iterations):
    """
    Procedure to Optimize the IMP model, given a Scoring Function and number of iterations/steps.
    :param tag: see function data_from_csv.
    :Optimization method: Conjugate Gradient Optimization
    :param model_to_optimize
    :param scoring_function
    :param num_iterations
    """
    o = IMP.core.ConjugateGradients(model_to_optimize)
    o.set_scoring_function(scoring_function)  # this method return the scoring function that is being used
    o.optimize(num_iterations)  # The maximum number of iteration of the optimizer to perform.
    # Returns the final score
    # print(scoring_function.evaluate(False))


def check_model_restrains(restrain_dictionary, tags=True):
    """
    Check the percentage of restrains fulfilled the condition distance +/- SD in the model
    :param tags:
    :param restrain_dictionary:
    :return: list with fulfilled restraints, and list with non-fulfilled restraints
    """
    f_restrains, non_f_restrains = list(), list()
    if tags:
        # Append the obtained distance after optimization into restrain dictionary
        for restrain in restrain_dictionary.values():
            b1 = restrain[1][0]
            b2 = restrain[1][1]
            x_dx = IMP.core.XYZ(b1)
            x_dy = IMP.core.XYZ(b2)
            new_distance = IMP.core.get_distance(x_dx, x_dy)
            restrain[3].append(new_distance)
            exp_distance = restrain[3][1]
            sd = restrain[3][0]
            if exp_distance - sd < new_distance < exp_distance + sd:
                f_restrains.append(restrain[0])
            else:
                non_f_restrains.append(restrain[0])
    else:
        for restrain in restrain_dictionary.values():
            if "UpperBound" in restrain[0].get_name():
                b1 = restrain[1][0]
                b2 = restrain[1][1]
                x_dx = IMP.core.XYZ(b1)
                x_dy = IMP.core.XYZ(b2)
                new_distance = IMP.core.get_distance(x_dx, x_dy)
                restrain[3].append(new_distance)
                exp_distance = restrain[3][1]
                if exp_distance > new_distance:
                    f_restrains.append(restrain[0])
                else:
                    non_f_restrains.append(restrain[0])
            else:
                pass
    return f_restrains, non_f_restrains


def write_hetatm_pdb(protein_tag_dictionary, atom_number, chain_id, chain, p_dict, pdb):
    """
    Write heteroatoms in pdb file, corresponding to tag positions from scaffold
    :param protein_tag_dictionary: used in the previous write_pdb function (exocyst or cog)
    :param atom_number: atom serial number
    :param chain_id: chain id of heteroatom (1, 2, 3..)
    :param chain: chain of residue (A, B, C...)
    :param p_dict: particles dictionary
    :param pdb: pdb file with writing mode
    :return: atom serial number and chain_id
    """
    # Write tag positions from scaffold in PDB as HETEROATOM (see chain_dict_tags)
    atom = atom_number
    atom += 1
    het_id = 0
    for tag in protein_tag_dictionary[chain]:
        chain_id += 1
        het_id += 1
        coord = p_dict[2][tag.capitalize()][1].get_coordinates()
        pdb.write("{:>6}".format("HETATM") + "{:>5}".format(atom) + "{:>9}".format(
            "C{} LYS".format(het_id)) + "{:>2}".format(chain) + "{:>4}".format(chain_id) + "{:12.3f}".format(
            coord[0]) + "{:8.3f}".format(
            coord[1]) + "{:8.3f}".format(coord[2]) + "{:>12}".format(
            "1.00 36.92") + "{:>13}".format("C\n"))
        atom += 1
    return atom, chain_id


def write_pdb(fm_dictionary, path_to_out_file, tags=True, cog=False):
    """
    Write PDB output file.
    :param cog:
    :param tags:
    :param fm_dictionary:
    :param path_to_out_file:
    :return: no return statement
    """
    used_fm = set()
    used_atoms = set()
    if tags:
        for model, p_dict in fm_dictionary.items():  # filtered model
            if model not in used_fm:
                used_fm.add(model)
            if not os.path.isdir(path_to_out_file + "/pdb/"):
                os.mkdir(path_to_out_file + "/pdb/")
            with open(path_to_out_file + "/pdb/%s.pdb" % str(model).replace('"', '').replace("_", "").lower(),
                      "w") as pdb:
                pdb.write("KEYWDS    IMP-score: {}, Acceptable restraints: {}%\n#\n"
                          .format(p_dict[1], p_dict[6]))
                atom = 1
                if not cog:
                    for chain in exocyst_chain_dict_tags:
                        for tag in exocyst_chain_dict_tags[chain]:
                            if atom not in used_atoms:
                                used_atoms.add(atom)
                                coord = p_dict[2][tag][1].get_coordinates()
                                pdb.write("{:>4}".format("ATOM") + "{:>7}".format(atom) + "{:>9}".format(
                                    "CA LYS") + "{:>2}".format(chain) + "{:>4}".format(atom) + "{:12.3f}".format(
                                    coord[0]) + "{:8.3f}".format(
                                    coord[1]) + "{:8.3f}".format(coord[2]) + "{:>12}".format(
                                    "1.00 36.92") + "{:>13}".format("C\n"))
                                atom += 1
                    used_atoms = set()
                elif cog:
                    for chain in cog_chain_dict_tags:
                        for tag in cog_chain_dict_tags[chain]:
                            if atom not in used_atoms:
                                used_atoms.add(atom)
                                coord = p_dict[2][tag][1].get_coordinates()
                                pdb.write("{:>4}".format("ATOM") + "{:>7}".format(atom) + "{:>9}".format(
                                    "CA LYS") + "{:>2}".format(chain) + "{:>4}".format(atom) + "{:12.3f}".format(
                                    coord[0]) + "{:8.3f}".format(
                                    coord[1]) + "{:8.3f}".format(coord[2]) + "{:>12}".format(
                                    "1.00 36.92") + "{:>13}".format("C\n"))
                                atom += 1
                    used_atoms = set()

    else:  # write pdb for architecture with tags as heteroatoms
        for model, p_dict in fm_dictionary.items():
            if model not in used_fm:
                used_fm.add(model)
                if not os.path.isdir(path_to_out_file + "/pdb/"):
                    os.mkdir(path_to_out_file + "/pdb/")
                with open(path_to_out_file + "/pdb/%s.pdb" % str(model).replace('"', '').replace("_", "").lower(),
                          "w") as pdb:
                    pdb.write("KEYWDS    IMP-score: {}, Acceptable restraints: {}%, Scaffold model: {}\n#\n"
                              .format(p_dict[1], p_dict[6], p_dict[7]))
                    # Write PDB to define atoms and chains (see chain_dict_exocyst dictionary)
                    atom = 1
                    chain_id = 100
                    protein_dict = dict()
                    protein_tag_dict = dict()
                    if not cog:
                        protein_dict = exocyst_chain_dict_architecture
                        protein_tag_dict = exocyst_chain_dict_tags
                    elif cog:
                        protein_dict = cog_chain_dict_architecture
                        protein_tag_dict = cog_chain_dict_tags
                    for chain in protein_dict:
                        for bead in protein_dict[chain]:
                            if atom not in used_atoms:
                                if bead.endswith("_c"):  # if last atom of chain write "TER" residue
                                    used_atoms.add(atom)
                                    coord = p_dict[2][bead][1].get_coordinates()
                                    pdb.write("{:>4}".format("ATOM") + "{:>7}".format(atom) + "{:>9}".format(
                                        "CA LYS") + "{:>2}".format(chain) + "{:>4}".format(
                                        atom) + "{:12.3f}".format(
                                        coord[0]) + "{:8.3f}".format(
                                        coord[1]) + "{:8.3f}".format(coord[2]) + "{:>12}".format(
                                        "1.00 36.92") + "{:>13}".format("C\n"))
                                    atom += 1
                                    pdb.write("{:>3}".format("TER") + "{:>8}".format(atom) + "{:>9}".format(
                                        "LYS") + "{:>2}".format(chain) + "{:>4}".format(atom - 1) + "\n")
                                    used_atoms.add(atom)
                                    atom, chain_id = write_hetatm_pdb(protein_tag_dict, atom, chain_id, chain, p_dict,
                                                                      pdb)
                                    atom += 1
                                elif bead.endswith("_C"):  # to write Sec2_GFP_C tags in model as heteroatoms
                                    atom, chain_id = write_hetatm_pdb(protein_tag_dict, atom, chain_id, chain, p_dict,
                                                                      pdb)
                                else:
                                    used_atoms.add(atom)
                                    coord = p_dict[2][bead][1].get_coordinates()
                                    pdb.write("{:>4}".format("ATOM") + "{:>7}".format(atom) + "{:>9}".format(
                                        "CA LYS") + "{:>2}".format(chain) + "{:>4}".format(
                                        atom) + "{:12.3f}".format(
                                        coord[0]) + "{:8.3f}".format(
                                        coord[1]) + "{:8.3f}".format(coord[2]) + "{:>12}".format(
                                        "1.00 36.92") + "{:>13}".format("C\n"))
                                    atom += 1
                    # Write tag positions from scaffold in PDB as HETEROATOM (see chain_dict_tags)

                    pdb.write("END")
                    used_atoms = set()


def write_model_restraint_log(path_log_file, restrain_dictionary, i, bef_opt=False, aft_opt=False):
    """
    Write log file of restraints
    :param i: iteration over models
    :param path_log_file:
    :param restrain_dictionary:
    :param bef_opt: True if log file before optimization step(e.g, conjugate gradient). Default False.
    :param aft_opt: True if log file before optimization step(e.g, conjugate gradient). Default False.
    :return: no return statement
    """
    if not os.path.isdir(path_log_file + "/log"):
        os.mkdir(path_log_file + "/log")
    if bef_opt:
        with open(path_log_file + "/log/log_before_%s.txt" % i, "w") as lf:  # DEFINING RESTRAINS #
            lf.write("###Model %s###\n\n###BEFORE OPTIMIZATION###\n\n" % i)
            for restrain in restrain_dictionary.values():
                b1 = restrain[1][0]
                b2 = restrain[1][1]
                dx = restrain[2][0]
                dy = restrain[2][1]
                x_dx = IMP.core.XYZ(b1)
                x_dy = IMP.core.XYZ(b2)
                exp_distance = restrain[3][1]
                sd = restrain[3][0]
                random_distance = IMP.core.get_distance(x_dx, x_dy)
                lf.write("Pair %s %s\n coor_%s = %s\n coor_%s = %s\n "
                         "random distance = %s\n exp_distance = %s\n sd = %s\n\n"
                         % (b1, b2, b1, dx, b2, dy, random_distance, exp_distance, sd))

    if aft_opt:
        with open(path_log_file + "/log/log_after_%s.txt" % i, "w") as lf:
            lf.write("###Model %s###\n\n###AFTER OPTIMIZATION###\n\n" % i)
            for restrain in restrain_dictionary.values():
                b1 = restrain[1][0]
                b2 = restrain[1][1]
                dx = restrain[2][0]
                dy = restrain[2][1]
                x_dx = IMP.core.XYZ(b1)
                x_dy = IMP.core.XYZ(b2)
                exp_distance = restrain[3][1]
                sd = restrain[3][0]
                optimized_distance = IMP.core.get_distance(x_dx, x_dy)  # adding radius distances
                lf.write("Pair %s %s\n coor_%s = %s\n coor_%s = %s\n "
                         "optimized distance = %s\n exp_distance = %s\n sd = %s\n\n"
                         % (b1, b2, b1, dx, b2, dy, optimized_distance, exp_distance, sd))


def write_filtered_models_log_file(path_log_file, filtered_dictionary, option=1):
    """
    Write file with 200 best models, its IMP score and % of fulfilled restraints
    :param option: option from input (1 as default - normal workflow) and 3 if batches used.
    :param filtered_dictionary: dictionary with 200 best filtered models
    :param path_log_file: path to save the output file
    """
    if not os.path.isdir(path_log_file + "/filter/log_final"):
        os.mkdir(path_log_file + "/filter/log_final")

    with open(path_log_file + "/filter/log_final/log_final.txt", "w") as lf:
        lf.write("###Filtered Models --> A total of {} filtered models.###\n\n".format(len(filtered_dictionary)))
        lf.write("Date: {}\n\n".format(datetime.date.today()))
        lf.write("{:^10}{:^20}{:^20}\n\n".format("Model", "IMPscore", "Restrains fulfilled(%)"))
        for model, items in filtered_dictionary.items():
            m = model
            imp_score = items[1]
            r_fulfilled = items[6]
            lf.write("{:^10}{:^20.3f}{:^20.2f}\n".format(m, imp_score, r_fulfilled))
        lf.write("\n#########################################\n\n")
        for model, items in filtered_dictionary.items():
            if option == 1:
                num_particles = len(items[2])
                num_restraints = len(items[3])
                lf.write("Number of particles in models: {}\n".format(num_particles))
                lf.write("Number of restraints in models: {}\n".format(num_restraints))
                break  # I just check it for one model, only to retrieve this variables
            elif option == 3:
                num_particles = len(items[2])
                lf.write("Number of particles in models: {}\n".format(num_particles))
                break  # I just check it for one model, only to retrieve this variables
    if option == 1:
        with open(path_log_file + "/filter/log_final/not_fulfilled.txt", "w") as lf_nf:
            lf_nf.write("### File with non-fulfilled restrains per each model ###\n\n")
            lf_nf.write(
                "### Filtered Models --> A total of {} filtered models. ###\n\n".format(len(filtered_dictionary)))
            for model, items in filtered_dictionary.items():
                m = model
                not_fulfilled = str(items[5]).strip("[]")
                lf_nf.write("\n{:^10} Not fulfilled({}): {:300}\n".format(m, len(items[5]), not_fulfilled))
    else:
        pass


def set_positions_from_scaffold(pdb_path, model, radius, cog=False):
    """
    Process coordinates from anchors and tags in scaffold model and returns a dictionary with the
    corresponding model particles and its coordinates as a IMP points (not spheres).
    :param cog: Bool working with exocyst or cog complex.
    :param radius: radius to represent the tag particles.
    :param model: IMP model.
    :param pdb_path: path of filtered models to randomly pick one.
    :return: dictionary with scaffold particles.
    """
    # Take tags and atoms from the structure
    p = PDBParser(PERMISSIVE=1)
    file = random.choice(os.listdir(pdb_path))
    # print("\n%s chosen as scaffold\n" % file)
    scaffold_model = p.get_structure(str(file), pdb_path + str(file))  # getting the structure from the pdb
    scaffold_model_atoms = [atom for atom in scaffold_model.get_atoms()]  # list of atoms
    scaffold_tag_list = list()
    if not cog:
        scaffold_tag_list = [el.capitalize() for lst in exocyst_chain_dict_tags.values() for el in
                             lst]  # list with tags and anchors
    elif cog:
        scaffold_tag_list = [el.capitalize() for lst in cog_chain_dict_tags.values() for el in
                             lst]  # list with tags and anchors
    # Create dict with tag/anchor as key and atom coordinates as value --> getting scaffold coordinates.
    scaffold_dict = dict()
    used_atoms = list()
    for tag in scaffold_tag_list:
        for atom in scaffold_model_atoms:
            if atom not in used_atoms:
                used_atoms.append(atom)
                if tag not in scaffold_dict:
                    scaffold_dict.setdefault(tag, atom.get_coord())
                    break
    # Create IMP particles and set atom coordinates of scaffold
    scaffold_particles_dict = dict()
    for tag, coord in scaffold_dict.items():
        p = IMP.Particle(model, tag)
        d = IMP.core.XYZR.setup_particle(p, IMP.algebra.Sphere3D(IMP.algebra.Vector3D(coord[0], coord[1], coord[2]),
                                                                 float(radius)))
        # d.set_coordinates_are_optimized(True) would move and optimized all these particles again
        if p not in scaffold_particles_dict:
            scaffold_particles_dict.setdefault(tag, [p, d])
    return scaffold_particles_dict, file


def set_y2h_restraint(model, particles_dictionary, p1, p2, radius):
    """
    Create yeast-two-hybrid restraint between selection p1 and selection p2 particles of the protein complex.
    :param p2: Protein 2 (e.g. Sec8)
    :param p1: Protein 1 (e.g Sec6)
    :param model: IMP model.
    :param particles_dictionary
    :param radius
    :return: y2h
    """
    p1_core = [particles_dictionary[p][0] for p in particles_dictionary if p.startswith(p1)]
    p2_core = [particles_dictionary[p][0] for p in particles_dictionary if p.startswith(p2)]
    # ps = [x for x in p1_core if x.get_name().endswith("_n")][0],
    # [x for x in p2_core if x.get_name().endswith("_n")][0]]  # force sec6 N interact with Sec8 N
    # ps = p1_core + p2_core  # --> for a normal Y2H
    ps = [random.choice(p1_core), [x for x in p2_core if x.get_name().endswith("_n")][0]]
    container = IMP.container.ListSingletonContainer(model, ps)
    hpb = IMP.core.Harmonic(radius * 2, IMP.core.Harmonic_get_k_from_standard_deviation(sd=2))
    dps = IMP.core.DistancePairScore(hpb)
    y2h = IMP.core.ConnectivityRestraint(dps, container)  # ensures particles will be connected somehow.
    y2h.set_name("Y2H_{}-{}".format(p1, p2))
    return y2h


def create_restrains_from_scaffold(model, data_dictionary, particles_dictionary, scaffold_dictionary):
    """
    Create spacial restraints taken a Tag model as scaffold, and apply two more restrains:
        - Concatenated beads (each protein must be a string of concatenated beads (N-2-3-4-5-C, e.g)
          where N and C are terminals connected by 4 beads.
        - Exclude Volume: two or more beads can not occupy the same volume in space.
    Returns the corresponding restraint dictionary
    :param scaffold_dictionary: contains IMP particles of the scaffold model.
    :param model: IMP model.
    :param data_dictionary: main dictionary with experimental data.
    :param particles_dictionary: Particles dictionary from data.
    :return: Scaffold_restraint_dictionary and dictionary with restraints for exclude volume

        r_dict[0] = { PairRestraint(x) : [PairRestraint(x), p1, p2, d1, d2], [bead, anchor/tag], [sd, distance]}
        of len(r_dict[0]) => 61 restraints.
        r_dict[1] = { PairRestraint(x) : [PairRestraint(x), bead1, bead2]} of len(r_dict[1]) => 39
    """
    restrain_dict = dict()
    restrains_set = set()
    for bead, l in data_dictionary.items():
        p_temp_list = list()  # particle temporary list where p1 and p2 will be captured
        pairs = list()  # bead and bead/anchor/tag
        coord_list = list()  # coord of the two items
        pairs.append(bead)
        coord_list.append(particles_dictionary[bead][1])
        p1 = particles_dictionary[bead][0]  # IMP particle
        p_temp_list.append(p1)
        d = float()
        k = float()
        d2 = float()
        k2 = float()
        if len(l) > 1:  # for protein C ter distances to (a) concatenated bead or (b) scaffold GFP_C and anchor tags
            for ele in l:
                pairs.append(ele[0])
                if ele[0] in particles_dictionary:  # distance to concatenated bead in protein
                    p2 = particles_dictionary[ele[0]][0]
                    d = float(ele[1]) - 40
                    p_temp_list.append(p2)
                    coord_list.append(particles_dictionary[ele[0]][1])
                    k = IMP.core.Harmonic_get_k_from_standard_deviation(sd=5)
                    d2 = float(ele[1]) - 35
                    k2 = IMP.core.Harmonic_get_k_from_standard_deviation(sd=0.1)
                elif ele[0] in scaffold_dictionary:  # to GFP_C and anchor tag distances
                    p2 = scaffold_dictionary[ele[0]][0]
                    d = float(ele[1]) / 2
                    p_temp_list.append(p2)
                    coord_list.append(scaffold_dictionary[ele[0]][1])
                    k = IMP.core.Harmonic_get_k_from_standard_deviation(sd=d)
                    d2 = float(ele[1])
                    k2 = IMP.core.Harmonic_get_k_from_standard_deviation(sd=0.1)
                # Harmonic Restraint
                f1 = IMP.core.Harmonic(d, k)
                s = IMP.core.DistancePairScore(f1)
                harmonic_restraint = IMP.core.PairRestraint(model, s, (p_temp_list[0], p_temp_list[1]))
                harmonic_restraint.set_name("Harmonic {} - {}".format(p_temp_list[0], p_temp_list[1]))
                restrain_dict["{0}".format(harmonic_restraint.get_name())] = [harmonic_restraint,
                                                                              [p_temp_list[0], p_temp_list[1]],
                                                                              [coord_list[0], coord_list[1]],
                                                                              [float(k), float(d)]]
                # UpperBound Restraint
                f2 = IMP.core.HarmonicUpperBound(mean=d2, k=k2)
                upper_bound = IMP.core.DistanceRestraint(model, f2, p_temp_list[0], p_temp_list[1])
                upper_bound.set_name("UpperBound {} - {}".format(p_temp_list[0], p_temp_list[1]))
                restrain_dict["{0}".format(upper_bound.get_name())] = [upper_bound,
                                                                       [p_temp_list[0], p_temp_list[1]],
                                                                       [coord_list[0], coord_list[1]],
                                                                       [float(k2), float(d2)]]
                restrains_set.add(harmonic_restraint)
                restrains_set.add(upper_bound)
                del p_temp_list[-1]
                del pairs[-1]
                del coord_list[-1]
        else:
            if l[0][0] in particles_dictionary:  # Distance between concatenated beads inside the protein
                p2 = particles_dictionary[l[0][0]][0]
                d = float(l[0][1]) - 40
                p_temp_list.append(p2)
                coord_list.append(particles_dictionary[l[0][0]][1])
                pairs.append(l[0][0])
                k = IMP.core.Harmonic_get_k_from_standard_deviation(sd=5)
                d2 = float(l[0][1]) - 35
                k2 = IMP.core.Harmonic_get_k_from_standard_deviation(sd=0.1)
            elif l[0][0] in scaffold_dictionary:  # to GFP_N ter distances
                p2 = scaffold_dictionary[l[0][0]][0]
                d = float(l[0][1]) / 2
                p_temp_list.append(p2)
                coord_list.append(scaffold_dictionary[l[0][0]][1])
                pairs.append(l[0][0])
                k = IMP.core.Harmonic_get_k_from_standard_deviation(sd=d)
                d2 = float(l[0][1])
                k2 = IMP.core.Harmonic_get_k_from_standard_deviation(sd=0.1)
            # Harmonic Restraint
            f1 = IMP.core.Harmonic(d, k)
            s = IMP.core.DistancePairScore(f1)
            harmonic_restraint = IMP.core.PairRestraint(model, s, (p_temp_list[0], p_temp_list[1]))
            harmonic_restraint.set_name("Harmonic {} - {}".format(p_temp_list[0], p_temp_list[1]))
            restrain_dict["{0}".format(harmonic_restraint.get_name())] = [harmonic_restraint,
                                                                          [p_temp_list[0], p_temp_list[1]],
                                                                          [coord_list[0], coord_list[1]],
                                                                          [float(k), float(d)]]
            # UpperBound Restraint
            f2 = IMP.core.HarmonicUpperBound(mean=d2, k=k2)
            upper_bound = IMP.core.DistanceRestraint(model, f2, p_temp_list[0], p_temp_list[1])
            upper_bound.set_name("UpperBound {} - {}".format(p_temp_list[0], p_temp_list[1]))
            restrain_dict["{0}".format(upper_bound.get_name())] = [upper_bound,
                                                                   [p_temp_list[0], p_temp_list[1]],
                                                                   [coord_list[0], coord_list[1]],
                                                                   [float(k2), float(d2)]]
            restrains_set.add(harmonic_restraint)
            restrains_set.add(upper_bound)
    return restrain_dict, restrains_set


def create_models(path_to_csv, path_to_output_dir, radius, opt_iterations, n, l=0,
                  log_file=False, tags=True, cog=False):
    """
    Creates IMP model from experimental data, applying distance constrains in 3D Space between
    FBR-Anchors and GFP-Tags(N/C -ter) and generates output file with PyMOL readable format.
    :param l:
    :param cog: bool (if True it model the COG complex, default:False)
    :param log_file: bool (if True it writes log file, default:False)
    :param tags: bool (if True it model the fluorophore positions, default:False)
    :param opt_iterations: Optimization Iterations
    :param radius: Sphere Radius
    :param path_to_csv: Path to csv data file
    :param path_to_output_dir: Path to output directory
    :param n: Number of models to create
    :return: model, IMP score, Fulfilled restrains, Non-fulfilled restrains, % acceptable restrains
    """
    models_dict = dict()
    # Create Tag Model
    if tags:
        print("\nCreating Models. Please wait...\n")
        for i in range(l, n):
            try:
                if (int(i) / n) * 100 == 25:
                    print("\n25% of the models created...\n")
                elif (int(i) / n) * 100 == 50:
                    print("\n50% of the models created...\n")
                elif (int(i) / n) * 100 == 75:
                    print("\n75% of the models created...\n")
                elif i == n:
                    print("\nCreating last model...\n")
                m = IMP.Model()  # Create IMP model instance
                m.set_name("Model_%s" % i)
                if cog:
                    m_dictionary = data_from_csv(path_to_csv, cog=True)
                else:
                    m_dictionary = data_from_csv(path_to_csv)
                p_dict = create_particles(m_dictionary, m)  # DEFINING PARTICLES #
                particles_with_coordinates(p_dict, radius)  # DEFINING DECORATORS #
                r_dict = create_restrains(m, m_dictionary, p_dict)  # DEFINING RESTRAINS #
                if log_file:
                    write_model_restraint_log(path_to_output_dir, r_dict, i,
                                              bef_opt=True)  # WRITE LOG FILE (OPTIONAL) #
                sf = calculate_sf(r_dict)  # CALCULATE SCORING FUNCTION #
                # print(sf.evaluate(False))
                conjugate_gradient_optimization(m, sf, opt_iterations)  # OPTIMIZATION STEP and calculate score
                # print(sf.evaluate(False))
                if log_file:
                    write_model_restraint_log(path_to_output_dir, r_dict, i,
                                              aft_opt=True)  # WRITE LOG FILE (OPTIONAL) #
                # Check percentage of restrains fulfilled #
                check_restrains = check_model_restrains(r_dict)
                f_restrains, nf_restrains = check_restrains[0], check_restrains[1]
                acceptable_r = float(len(f_restrains)) / float(len(r_dict)) * 100
                imp_score = sf.evaluate(True)
                # print("Model %s created\t %s \t %s" % (i, acceptable_r, imp_score))
                # Create Models dictionary and Write Output to PDB #
                models_dict.setdefault(m,
                                       (m, imp_score, p_dict, r_dict, f_restrains, nf_restrains, acceptable_r))
            except IMP.ModelException as e:
                print("Exception {} risen but continue".format(e))
                pass

        # WRITE PDB FILES
        print("\nWriting PDB files of tag models...\n")
        if not os.path.isdir(path_to_output_dir + "all/"):
            os.mkdir(path_to_output_dir + "all/")
        if not cog:
            write_pdb(models_dict, path_to_output_dir + "all/")
        elif cog:
            write_pdb(models_dict, path_to_output_dir + "all/", cog=True)

    # Create Exocyst Architecture From Scaffold

    else:
        # cluster = "cluster_1"
        cluster = "cluster_" + input("\nType scaffold cluster to build the 3D architecture ('1 or 2, e.g'): ")
        print("\nCreating Models. Please wait...\n")
        # 1. Gathering of data:
        m_dictionary = dict()
        if not cog:
            m_dictionary = data_from_csv(path_to_csv, tags=False)
        elif cog:
            m_dictionary = data_from_csv(path_to_csv, tags=False, cog=True)
        for i in range(l, n):
            try:
                if (int(i) / n) * 100 == 25:
                    print("\n25% of the models created...\n")
                elif (int(i) / n) * 100 == 50:
                    print("\n50% of the models created...\n")
                elif (int(i) / n) * 100 == 75:
                    print("\n75% of the models created...\n")
                elif i == n:
                    print("\n100% of the models created...\n")
                # print("creating model number %s" % i)
                m = IMP.Model()
                m.set_name("Model_%s" % i)
                # 2. Representation of subunits as IMP particles and set random positions:
                p_dict = create_particles(m_dictionary, m, tags=False)  # DEFINING PARTICLES #
                ps = particles_with_coordinates(p_dict, radius, tags=False)  # DEFINING DECORATORS #
                # Define container with all the particles (with coordinates)
                sc = IMP.container.ListSingletonContainer(m, ps)
                excluded_volume = IMP.core.ExcludedVolumeRestraint(sc, 0.01)
                excluded_volume.set_name("EV")
                # 3. Set Scaffold (random choose from filtered Tag models)
                random_pdb_path = ""
                scaffold = dict()
                scaffold_p_dict = dict()
                if not cog:
                    random_pdb_path += "../output/exocyst/tags/filter/pdb/{}/".format(cluster)
                    scaffold_p_dict, scaffold = set_positions_from_scaffold(random_pdb_path, m,
                                                                            radius=1)  # returns 2 dictionaries
                elif cog:
                    random_pdb_path += "../output/cog/tags/filter/pdb/{}/".format(cluster)
                    scaffold_p_dict, scaffold = set_positions_from_scaffold(random_pdb_path, m,
                                                                            radius=1, cog=True)

                # 4. Add spacial restrains:
                restraints = create_restrains_from_scaffold(m, m_dictionary, p_dict,
                                                            scaffold_p_dict)  # DEFINING RESTRAINS #
                r_dict, restraint_set = restraints[0], restraints[1]
                # Calculate Yeast-2-Hybrid restraint between Sec6 and Sec8
                y2h = list()
                if not cog:
                    y2h = set_y2h_restraint(m, p_dict, p1="Sec6", p2="Sec8", radius=17.5)
                elif cog:
                    y2h = set_y2h_restraint(m, p_dict, p1="Cog6", p2="Cog8", radius=17.5)
                restraint_set.add(y2h)
                restraint_set.add(excluded_volume)
                # Create Models dictionary and Write Output to PDB or Pymol #
                """models_dict.setdefault(m, (m, m, all_particles, r_dict,
                                           m, m, m, scaffold))
                write_pdb(models_dict, path_to_output_dir + "test1/", tags=False)"""
                if log_file:
                    write_model_restraint_log(path_to_output_dir, r_dict, i,
                                              bef_opt=True)  # WRITE LOG FILE (OPTIONAL) #
                # 5. Scoring function and Conjugate Gradient optimization
                sf = calculate_sf(r_dict, restraint_set, tags=False)  # CALCULATE SCORING FUNCTION #
                conjugate_gradient_optimization(m, sf, opt_iterations)  # OPTIMIZATION STEP #
                if log_file:
                    write_model_restraint_log(path_to_output_dir, r_dict, i, radius,
                                              aft_opt=True)  # WRITE LOG FILE (OPTIONAL)
                # Check percentage of restrains fulfilled #
                check_restrains = check_model_restrains(r_dict, tags=False)
                f_restrains, nf_restrains = check_restrains[0], check_restrains[1]
                # Calculate IMP score and acceptable restrains
                imp_scoring = sf.evaluate(False)
                ev_score = excluded_volume.evaluate(False)
                y2h_score = y2h.evaluate(False)
                f_restrains.append(ev_score), f_restrains.append(y2h_score)
                r_dict["{0}".format(excluded_volume)] = ev_score
                r_dict["{0}".format(y2h)] = y2h_score
                acceptable_r = float(len(f_restrains)) / (float(len(f_restrains)) + float(len(nf_restrains))) * 100
                all_particles = {**p_dict, **scaffold_p_dict}
                # Create Models dictionary and Write Output to PDB or Pymol #
                models_dict.setdefault(m, (m, imp_scoring, all_particles, r_dict,
                                           f_restrains, nf_restrains, acceptable_r, scaffold))
                # write_pdb(models_dict, path_to_output_dir + "test2/", tags=False)
            except IMP.ModelException as e:
                print("Exception {} risen but continue".format(e))
                pass

        # WRITE PDB FILES
        print("\nWriting PDB files of tag models...\n")
        if not os.path.isdir(path_to_output_dir + "all/"):
            os.mkdir(path_to_output_dir + "all/")
        if not cog:
            write_pdb(models_dict, path_to_output_dir + "all/", tags=False)
        elif cog:
            write_pdb(models_dict, path_to_output_dir + "all/", tags=False, cog=True)
    return models_dict


def filter_models_by_imp_score(models_dictionary, path_to_out_filt_file, f, option=1, pdb=True, tags=True, cog=False):
    """
    From initial set of models, the function filters by IMP score a desired number of models.
    Writes output files in .pym and .pdb format.
    :param option: option from input (default is 1 --> complete workflow), if 3 it filters models from folder using
    biopython.
    :param cog:
    :param tags:
    :param pdb: True/False statement to write PDB output file.
    :param path_to_out_filt_file: path to output filter model file
    :param models_dictionary: dictionary with models and its IMP scores.
    :param f: number of best models to filter from the initial models
    :return: dictionary with best models
    """
    print("\nSorting models by IMP score and filtering best {} models...\n".format(f))
    filtered_model_names = ["filt_model_" + str(x) for x in range(f)]
    used_filtered_model_names = set()
    filtered_models = set()
    # Sort dictionary by IMP score
    sorted_imp = sorted(models_dictionary.values(), key=lambda key_value: key_value[1],
                        reverse=False)  # return a list of tuples
    sorted_restraints = sorted(sorted_imp, key=lambda key_value: key_value[6],
                               reverse=True)  # return a list of tuples
    # Filter the best models by IMP score
    best_models = dict()
    for n in range(int(f)):
        for model_tpl in sorted_restraints:
            if option == 1:  # complete workflow
                if tags:
                    if str(model_tpl[0]).replace(" ", "_") not in best_models:
                        imp_score, p_dict, r_dict, f_restraints, nf_restraints, acceptable = \
                            model_tpl[1], model_tpl[2], model_tpl[3], model_tpl[4], model_tpl[5], model_tpl[6]
                        if acceptable == float(100):
                            best_models.setdefault(str(model_tpl[0]).replace(" ", "_"),
                                                   (
                                                       model_tpl[0], imp_score, p_dict, r_dict, f_restraints,
                                                       nf_restraints,
                                                       acceptable))
                            break
                        else:
                            best_models.setdefault(str(model_tpl[0]).replace(" ", "_"),
                                                   (
                                                       model_tpl[0], imp_score, p_dict, r_dict, f_restraints,
                                                       nf_restraints,
                                                       acceptable))
                            break
                else:
                    if str(model_tpl[0]).replace(" ", "_") not in best_models:
                        imp_score, p_dict, r_dict, f_restraints, nf_restraints, acceptable, scaffold = \
                            model_tpl[1], model_tpl[2], model_tpl[3], model_tpl[4], model_tpl[5], model_tpl[6], \
                            model_tpl[7]
                        best_models.setdefault(str(model_tpl[0]).replace(" ", "_"),
                                               (model_tpl[0], imp_score, p_dict, r_dict, f_restraints, nf_restraints,
                                                acceptable, scaffold))
                        break
            elif option == 2:  # workflow using batches
                if model_tpl[3] not in best_models:
                    imp_score, structure, name, restraints = model_tpl[1], model_tpl[2], model_tpl[3], model_tpl[6]
                    best_models.setdefault(model_tpl[3], [name, imp_score, list(structure.get_atoms()), structure, [],
                                                          [], restraints])
                    break
    if option == 1:
        if pdb:
            print("\nWriting filtered PDB...\n")
            if not os.path.isdir(path_to_out_filt_file + "/filter/"):
                os.mkdir(path_to_out_filt_file + "/filter/")
            if tags:
                if not cog:
                    write_pdb(best_models, path_to_out_filt_file + "filter")
                elif cog:
                    write_pdb(best_models, path_to_out_filt_file + "filter", cog=True)
            else:
                if not cog:
                    write_pdb(best_models, path_to_out_filt_file + "/filter", tags=False)
                elif cog:
                    write_pdb(best_models, path_to_out_filt_file + "/filter", tags=False, cog=True)
        write_filtered_models_log_file(path_to_out_filt_file, best_models)  # writing a log final text file
    elif option == 2:
        io = PDBIO()
        print("\nWriting filtered PDB...\n")
        if not os.path.isdir(path_to_out_filt_file + "/filter/"):
            os.mkdir(path_to_out_filt_file + "/filter/")
            os.mkdir(path_to_out_filt_file + "/filter/" + "pdb/")
        for name, s_values in best_models.items():
            io.set_structure(s_values[3])
            io.save(file=path_to_out_filt_file + "/filter/" + "pdb/" + name)
        write_filtered_models_log_file(path_to_out_filt_file, best_models, option=3)  # writing a log final text file
    return best_models


def hierarchical_clustering(path_to_pdb_files, linkage_method, dendogram_threshold, only_dendogram=False,
                            tags=True, cog=False, agglomerative_clustering=False):
    """
    Generates a Hierarchical Clustering and outputs a dendogram.
    Requires Biopython Bio.PDB, numpy, pyplot, and scipy.
    :param only_dendogram: if True it do hierarchical clustering with already existing models in the output folder
    :param cog:
    :param dendogram_threshold: threshold line to differentiate clusters in dendogram
    :param tags: (bool, default True)
    :param path_to_pdb_files: complete path where the PDB model files are.
    :param agglomerative_clustering: if True it will perform a Hierarchical Clustering with the given number of
    clusters and show the resulting plot.
    :param linkage_method: hierarchical clustering method with scipy.cluster.hierarchy to cluster and make dendogram:
        - linkage: performs hierarchical/agglomerative clustering minimizing the distance with the method specified.
        - method:
            - single: Perform single/min/nearest linkage on the condensed distance matrix M.
            - complete: Perform complete/max/farthest point linkage on a condensed distance matrix.
            - average: Perform average/UPGMA linkage on a condensed distance matrix.
            - weighted: Perform weighted/WPGMA linkage on the condensed distance matrix.
            - centroid: Perform centroid/UPGMC linkage.
            - median: Perform median/WPGMC linkage.
            - ward: Perform Ward’s linkage on a condensed distance matrix.
    :return: cluster dictionary
    info:
    - https://towardsdatascience.com/understanding-the-concept-of-hierarchical-clustering-technique-c6e8243758ec
    - https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    """
    # Using Biopython modules
    p = PDBParser(PERMISSIVE=0)
    structures = dict()
    if only_dendogram:
        for folder in os.listdir(path_to_pdb_files):
            if not folder.endswith(".csv") and not folder.endswith(".py") and not folder.endswith(".pyc"):
                path = path_to_pdb_files + "{}/".format(folder)
                for f in os.listdir(path):
                    if f.endswith(".pdb"):
                        s = p.get_structure(str(f), path + str(f))  # Creating a BioPDB structure
                        structures.setdefault(s,
                                              list(s.get_atoms()))
    else:
        for file in os.listdir(path_to_pdb_files):
            if file.endswith(".pdb"):
                s = p.get_structure(str(file), path_to_pdb_files + str(file))  # Creating a BioPDB structure
                # Getting Atoms objects from each structure to pursue the RMSD.
                structures.setdefault(s,
                                      list(s.get_atoms()))
    # Calculating RMSD
    print("\n\nStarting Hierarchical Clustering with %s models...\n Method: %s\n" % (len(structures), linkage_method))
    rmsd_list = list()
    sup = Superimposer()
    used_structures = list()
    for s1 in structures:
        used_structures.append(s1)
        temp_list = list()
        for s2 in structures:
            if s1 == s2:
                temp_list.append(0)  # the rmsd for a model against itself is 0
            else:
                fixed_atoms = structures[s1]
                moving_atoms = structures[s2]
                sup.set_atoms(fixed_atoms, moving_atoms)  # translation
                sup.apply(moving_atoms)  # applying rotation matrix to superposition
                temp_list.append(sup.rms)
        rmsd_list.append(temp_list)
    m = np.array(rmsd_list)
    # HIERARCHICAL CLUSTERING (AGGLOMERATIVE)
    print("\n####################\n")
    labels = np.array([s.get_id() for s in structures])  # labels for leaves in dendogram
    # z = fastcluster.linkage(m, method=linkage_method, metric='euclidean')
    z = linkage(m, method=linkage_method, optimal_ordering=False)  # linkage matrix (distance matrix)
    c, coph_dists = cophenet(z, pdist(m))  # cophenetic correlation coefficient
    print("\nCophenetic correlation coefficient is {}".format(c))
    assignments = fcluster(z, int(dendogram_threshold), 'distance')  # threshold line to choose clusters
    cluster_output = pandas.DataFrame({'models': labels, 'cluster': assignments})  # convert cluster to df
    cluster_groups = cluster_output.groupby('cluster')  # group models by cluster number
    cluster_output.to_csv(path_to_pdb_files + "clustering.csv", sep='\t',
                          encoding='utf-8')  # output to csv file
    groups = cluster_output.groupby('cluster')["models"].apply(list)  # group clusters and convert to lists
    cluster_dict = dict()  # store separated clusters in a dictionary
    for group in groups:
        for n in range(1, len(groups) + 1):
            if "cluster_{}".format(n) not in cluster_dict:
                cluster_dict.setdefault("cluster_{}".format(n), group)
                break
    # Make figure and dendogram
    plt.figure(figsize=(10, 7))
    plt.title('Hierarchical Clustering Dendogram', fontsize=20)
    plt.xlabel("Number of Models", fontsize=15)
    plt.ylabel("Euclidean Distance", fontsize=15)
    d = dendrogram(
        z,
        labels=None,
        color_threshold=int(dendogram_threshold),
        # leaf_rotation=90,
        truncate_mode="lastp",
        leaf_font_size=15,
        p=6,
        show_contracted=True,
        get_leaves=True,
    )
    leaves = dict(zip(d["ivl"], d["color_list"]))
    path = ""
    if tags:
        if not cog:
            path += "../output/exocyst/tags/hc_plots/"
        elif cog:
            path += "../output/cog/tags/hc_plots/"
    else:
        if not cog:
            path += "../output/exocyst/3D_exocyst/hc_plots/"
        elif cog:
            path += "../output/cog/3D_cog/hc_plots/"
    if not os.path.isdir(path):
        os.mkdir(path)
    num_files = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    dendogram_plot = "dendogram_" + str(num_files)  # name for dendrogram plot
    heatmap_plot = "hm_" + str(num_files)  # name for normal heatmap plot
    hmp_plot = "hmp_" + str(num_files)  # name for clustermap pot
    # Save dendogram plot in folder hc_plots
    if not os.path.isfile(path + dendogram_plot):
        plt.savefig(path + dendogram_plot)
        print("Successfully created the plot %s" % dendogram_plot)
    else:
        num_files += 1
        dendogram_plot = "/dendogram_" + str(num_files)
        plt.savefig(path + dendogram_plot)
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 7))
    # Seaborn heatmap beased on RMSD 2D-matrix
    hm = sns.heatmap(m, cbar_kws={'label': 'RMSD', 'orientation': 'horizontal'}, xticklabels=False,
                     yticklabels=False, cmap="YlGnBu", linewidths=0.30, ax=ax)
    # Seaborn heatmap beased on RMSD 2D-matrix with clusters on top
    hmp = sns.clustermap(
        data=m,
        method='ward',
        metric='euclidean',
        cmap="mako",
        # vmin=0,
        # vmax=150,
        figsize=(15, 15),
    )
    if not os.path.isfile(path + heatmap_plot):
        hm_figure = hm.get_figure()
        hm_figure.savefig(path + heatmap_plot, dpi=400)
        hmp.savefig(path + hmp_plot)
        print("\nSuccessfully created the plot %s" % heatmap_plot)
        print("\nSuccessfully created the plot %s" % hmp_plot)
    else:
        hm_figure = hm.get_figure()
        hm_figure.savefig(path + heatmap_plot, dpi=400)
        hmp.savefig(path + hmp_plot)
    if agglomerative_clustering:
        cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean',
                                          linkage='%s' % linkage_method)
        cluster.fit_predict(m)
        plt.figure(figsize=(10, 7))
        plt.scatter(m[:, 0], m[:, 1], c=cluster.labels_, cmap='rainbow')  # plot all points
        plt.show()
    return cluster_dict

#########
# IMP.pmi #
#########


def create_tag_molecules(pict_bdb, pict_components, pict_chains, state_tags):
    """
    Create IMP.pmi molecules using fluorophore coordinates
    :param pict_bdb:
    :param pict_components:
    :param pict_chains:
    :param state_tags:
    :return:
    """
    colors = ["blue", "orange", "yellow", "pink", "brown", "purple", "red", "green"]
    tag_molecules = list()
    for n in range(len(pict_components)):
        print('PMI: setting up tag', pict_components[n])
        molecule = state_tags.create_molecule(name=pict_components[n],
                                              sequence="",
                                              chain_id=pict_chains[n])

        atomic = molecule.add_structure(pdb_fn=pict_bdb,
                                        chain_id=pict_chains[n],
                                        res_range=[],  # add only specific set of residues
                                        soft_check=True,  # warns sequence mismatches between fasta and PDB sequence
                                        offset=0)  # to sync PDB residue numbering with FASTA numbering.

        molecule.add_representation(residues=atomic,  # adding all residues of PDB to representation
                                    bead_extra_breaks=[],  # Additional breakpoints for splitting beads.
                                    color=colors[n],
                                    bead_ca_centers=True,  # resolution=1 beads to be at CA centers
                                    resolutions=[1])

        tag_molecules.append(molecule)

    return tag_molecules


def create_tag_molecules_with_sec2(pict_bdb, pict_components, pict_chains, state_tags):
    """
    Create IMP.pmi molecules using fluorophore coordinates
    :param pict_bdb:
    :param pict_components:
    :param pict_chains:
    :param state_tags:
    :return:
    """
    colors = ["blue", "orange", "yellow", "pink", "brown", "purple", "red", "green", "cyan"]
    tag_molecules = list()
    for n in range(len(pict_components)):
        print('PMI: setting up tag', pict_components[n])
        molecule = state_tags.create_molecule(name=pict_components[n],
                                              sequence="",
                                              chain_id=pict_chains[n])

        atomic = molecule.add_structure(pdb_fn=pict_bdb,
                                        chain_id=pict_chains[n],
                                        res_range=[],  # add only specific set of residues
                                        soft_check=True,  # warns sequence mismatches between fasta and PDB sequence
                                        offset=0)  # to sync PDB residue numbering with FASTA numbering.

        molecule.add_representation(residues=atomic,  # adding all residues of PDB to representation
                                    bead_extra_breaks=[],  # Additional breakpoints for splitting beads.
                                    color=colors[n],
                                    bead_ca_centers=True,  # resolution=1 beads to be at CA centers
                                    resolutions=[1])

        tag_molecules.append(molecule)

    return tag_molecules


def get_pict_distance_restraints_dict(dr_tags_file):
    """
    Return dictionary of distances between tags and termini of cryo exocyst
    :param dr_tags_file:
    """
    # Defining PICT distances (from tags to termini of subunits)
    distances_to_tags_dict = dict()
    with open(dr_tags_file, "r") as dr:
        "dr_file --> tag,tag_residue,protein,protein_residue,raw_distance,Added distance,Final Distance"
        csv_reader = csv.reader(dr, delimiter=',')  # to parse the csv file delimited with commas.
        dr.readline()  # skip first line
        for line in csv_reader:
            tag = line[0]
            protein = line[2]
            tag_residue = int(line[1])
            protein_residue = int(line[3])
            chain_id = str(line[4])
            max_distance = float(line[7])
            raw_dist = float(line[5])
            sd = float(line[8])
            if raw_dist == 202:
                distances_to_tags_dict.setdefault("{}_frb-{}_C".format(tag, protein),
                                                  [tag, protein, tag_residue, protein_residue, chain_id, max_distance,
                                                   sd])

            elif raw_dist == 110:
                distances_to_tags_dict.setdefault("{}_gfp_c-{}_C".format(tag, protein),
                                                  [tag, protein, tag_residue, protein_residue, chain_id, max_distance,
                                                   sd])

            else:
                distances_to_tags_dict.setdefault("{}_gfp_n-{}_N".format(tag, protein),
                                                  [tag, protein, tag_residue, protein_residue, chain_id, max_distance,
                                                   sd])
    return distances_to_tags_dict


def set_pict_distance_restraints(distances_to_tags_dict, root_hierarchy, output_objects):
    """
    Setting and adding basic Harmonic Upper Bound distance restraints from distances_to_tags
    dictionary
    :return: distances_to_tags_dict, dr_list, output_objects
    """
    # Distance restraints to positioned tags:
    dr_list = list()
    for r, data in distances_to_tags_dict.items():
        tag = data[0]
        protein = data[1]
        tag_rn = int(data[2])
        protein_rn = int(data[3])
        max_distance = float(data[5])
        sd = float(data[6])
        kappa = IMP.core.Harmonic_get_k_from_standard_deviation(sd)
        dr = IMP.pmi.restraints.basic.DistanceRestraint(root_hier=root_hierarchy,
                                                        tuple_selection1=(tag_rn, tag_rn, tag),
                                                        tuple_selection2=(protein_rn, protein_rn, protein),
                                                        distancemin=0,
                                                        distancemax=max_distance,
                                                        # kappa=kappa,
                                                        label="restraint_{}".format(r))
        dr.add_to_model()
        dr.evaluate()
        dr_list.append(dr)
        output_objects.append(dr)
        print("\nSetting distance {}\n".format(dr.label))
        print("Max distance {}\n".format(max_distance))

    return distances_to_tags_dict, dr_list, output_objects


def distance_restraints_pict_cryo(dr_tags_file, root_hierarchy, output_objects):
    # Defining PICT distances (from tags to termini of subunits)
    distances_to_tags_dict = get_pict_distance_restraints_dict(dr_tags_file)

    # Distance restraints to positioned tags:
    return set_pict_distance_restraints(distances_to_tags_dict, root_hierarchy, output_objects)


def define_corex_regions(file, cryo_molecules):
    # Defining rigid regions from rigid_regions.csv
    rigid_regions_dict = dict()
    corex_1, corex_2 = list(), list()
    tetramer_1, tetramer_2 = list(), list()
    with open(file, "r") as rbf:
        "rbf_file --> subunit,res_ini,res_final,region,num_residues"
        csv_reader = csv.reader(rbf, delimiter=',')  # to parse the csv file delimited with commas.
        rbf.readline()  # skip first line
        for line in csv_reader:
            subunit = line[0]
            res_range = [line[1], line[2]]
            region_name = line[3]
            if subunit not in rigid_regions_dict:
                rigid_regions_dict.setdefault(subunit, [[res_range[0], res_range[1], region_name]])
            else:
                rigid_regions_dict[subunit] += [[res_range[0], res_range[1], region_name]]

    for cryo_mol in cryo_molecules:
        cc_region = list(chain.from_iterable((region[0], region[1]) for region in rigid_regions_dict[cryo_mol] if
                                             region[2] == 'CC'))
        corex_region = cryo_mol.residue_range(cc_region[0], cc_region[1])
        if cryo_mol in ["sec3", "sec5", "sec6", "sec8"]:
            tetramer_1.append(cryo_mol)
            corex_1 += corex_region
        else:
            tetramer_2.append(cryo_mol)
            corex_2 += corex_region

    return corex_1, corex_2


def add_tag_positions_to_cryo(tags_to_add_path, models_dir):
    # First, add to models PDB files (in /output/pdbs/0/) the tags positions
    tags_data = list(open(tags_to_add_path))
    # Rewrite files with tag information
    for file in glob.glob(models_dir + '*.pdb'):
        cryo_data = list(open(file))
        with open(file, "w") as f:
            f.writelines(cryo_data[:-1])
            f.writelines(tags_data)


def get_contact_residues_from_cryo(cryo_pdb_model, contact_distance=5, close="all"):
    """
    Getting contacts from cryoEM at a given distance
    :param cryo_pdb_model:
    :param contact_distance: desired contact distance (defalut=5)
    :param close: "all" for all contacts, "tetra1" or "tetra2" for contacts on tetramer1 or tetrarmer2
    :return: list of tuples of contact pairs from different chains
    """
    print("\nGetting contacts residues of {} atoms from cryoEM\n".format(close))
    p = PDBParser()
    structure = p.get_structure("cryoEM", cryo_pdb_model)
    tetra1_chains = ["A", "B", "C", "D"]
    tetra2_chains = ["E", "F", "G", "H"]
    tetra1_atoms, tetra2_atoms, all_atoms = list(), list(), list()
    for chain in structure.get_chains():
        if chain.id in tetra1_chains:
            tetra1_atoms += list(chain.get_atoms())
        elif chain.id in tetra2_chains:
            tetra2_atoms += list(chain.get_atoms())
        all_atoms += list(chain.get_atoms())
    nb_tetra1 = NeighborSearch(tetra1_atoms)
    nb_tetra2 = NeighborSearch(tetra2_atoms)
    nb_all = NeighborSearch(all_atoms)
    all_close_cryo = nb_all.search_all(contact_distance, "R")
    close_inter_proteins = [pair for pair in all_close_cryo if
                            pair[0].parent.id != pair[1].parent.id]
    all_close_tetra1 = nb_tetra1.search_all(contact_distance, "R")  # at distance of 3 Angstroms and al Residue level
    close_residues_tetra1 = [pair for pair in all_close_tetra1 if
                             pair[0].parent.id != pair[1].parent.id]  # between chains
    all_close_tetra2 = nb_tetra2.search_all(contact_distance, "R")
    close_residues_tetra2 = [pair for pair in all_close_tetra2 if
                             pair[0].parent.id != pair[1].parent.id]  # between chains

    if close == "all":
        return close_inter_proteins
    elif close == "tetra1":
        return close_residues_tetra1
    elif close == "tetra2":
        return close_residues_tetra2


def set_contacts_restraints_from_cryo(close_residues, output_objects, root_hierarchy, contact_distance):
    """
    Set basic IMP.pmi.basic distance restraints between 2 - contact distance from close pairs.
    :param close_residues: close residues (list of tuples)
    :param output_objects: list of output objects
    :param root_hierarchy: root hierarchy
    :param contact_distance: desired maximum contact distance (default = 8)
    :return:
    """
    chain_protein_dict = {
        "A": "sec3", "B": "sec5", "C": "sec6", "D": "sec8", "E": "sec10", "F": "sec15", "G": "exo70", "H": "exo84"
    }
    n = 0
    for contact_tple in close_residues:
        n += 1
        # residue 1
        res1 = contact_tple[0]
        res1_chain = res1.parent.id
        res1_idx = int(res1.id[1])
        mol1 = chain_protein_dict[res1_chain]
        # residue 2
        res2 = contact_tple[1]
        res2_chain = res2.parent.id
        res2_idx = int(res2.id[1])
        mol2 = chain_protein_dict[res2_chain]
        dr = IMP.pmi.restraints.basic.DistanceRestraint(root_hier=root_hierarchy,
                                                        tuple_selection1=(res1_idx, res1_idx, mol1),
                                                        tuple_selection2=(res2_idx, res2_idx, mol2),
                                                        distancemin=2,
                                                        distancemax=contact_distance,
                                                        label="restraint{}_{}_{}-{}_{}".format(n, mol1, res1_idx,
                                                                                               mol2, res2_idx))
        dr.add_to_model()
        dr.evaluate()
        output_objects.append(dr)
        print("Setting connectivity {}\n".format(dr.label))

    return output_objects


# -------------#
# Cluster      #
# -------------#

def get_username():
    """
    This functions returns the user name.

    """

    return pwd.getpwuid(os.getuid())[0]


def number_of_jobs_in_queue(qstat="qstat"):
    """
    This functions returns the number of jobs in queue for a given
    user.

    """

    # Initialize #
    user_name = get_username()

    process = subprocess.check_output([qstat, "-u", user_name])

    return len([line for line in process.split("\n") if user_name in line])


def submit_command_to_queue(command, queue=None, max_jobs_in_queue=None, queue_file=None, dummy_dir="/tmp",
                            submit="sbatch", qstat="qstat"):
    """
    This function submits any {command} to a cluster {queue}.

    @input:
    command {string}
    queue {string} by default it submits to any queue
    max_jobs_in_queue {int} limits the number of jobs in queue
    queue_file is a file with information specific of the cluster for running a queue

    """
    import hashlib

    if max_jobs_in_queue is not None:
        try:
            while number_of_jobs_in_queue(qstat) >= max_jobs_in_queue:
                time.sleep(5)
        except:
            print("Queue error to get the number of jobs. Continue submitting jobs")

    cwd = os.path.join(dummy_dir, "sh")
    if not os.path.exists(cwd):
        os.makedirs(cwd)
    script = os.path.join(cwd, "submit_" + hashlib.sha224(command.encode('utf-8')).hexdigest() + ".sh")
    if queue_file is not None:
        fd = open(script, "w")
        with open(queue_file, "r") as queue_standard:
            data = queue_standard.read()
            fd.write(data)
            fd.write("%s\n\n" % command)
        fd.close()
        queue_standard.close()
        if queue is not None:
            os.system("%s -q %s %s" % (submit, queue, script))
        else:
            os.system("%s %s" % (submit, script))
    else:
        if queue is not None:
            os.system("echo \"%s\" | %s -q %s" % (command, submit, queue))
        else:
            os.system("echo \"%s\" | %s" % (submit, command))


if __name__ == "__main__":
    print("\nThis is a python file with all the functions needed to reconstruct the scaffold"
          "and the hole architecture of protein complexes (exocyst and cog CATCHR complexes")
