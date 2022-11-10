#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import pandas as pd
import pymol

from model_classes import *
import time
from scipy.spatial.distance import cdist
import __main__
from pymol import cmd

__main__.pymol_argv = ['pymol', '-qc']  # Quiet and no GUI (not loading PyMOL GUI)

__author__ = "Altair C. Hernandez"
__copyright__ = "Copyright 2022, The Exocyst Modeling Project"
__credits__ = ["Ibai Irastorza", "Damien P. Devos", "Oriol Gallego"]
__version__ = "IMP 2.17.0"
__maintainer__ = "Altair C. Hernandez"
__email__ = "altair.chinchilla@upf.edu"
__status__ = "Development"


@cmd.extend
def color_tags(sphere_size=1):
    """
    Color tags:
        Sec3-chain A: marine
        Sec5-chain B: orange
        Sec6-chain C: yellow
        Sec8-chain D: pink
        Sec10-chain E: chocolate
        Sec15-chain F: purple
        Exo70-chain G: red
        Exo84-chain H: green
        Sec2-chain I: grey
    """
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
    cmd.color("grey70", "chain I")
    cmd.bg_color('black')


def label_tags(tag_repr_file):
    """
    Assign names to each tag position from
    a tag representation csv file wit columns:
        [name,chain,color]
    """
    tags_repr = pd.read_csv(tag_repr_file)
    idx = 1
    for name in tags_repr.name.to_list():
        cmd.create(name=name, selection=f'id {idx}')
        idx += 1


def align_tags(cluster_file, pdbs_dir, hc_dir,  tag_repr_file):
    """
        Align tag models from clusters
        using 'cmd.align' method
    """
    # Get models from cluster 1 and cluster 2
    data = pd.read_csv(cluster_file, sep=',')
    # Align models for each cluster and save session
    for cl in data.cluster.unique():
        cl_models = data.loc[data.cluster == cl, 'models'].to_list()
        # Align clusters
        print(f"\nStarting alignment of cluster {cl}...\n")
        for model in cl_models:
            cmd.load(f'{pdbs_dir}/{model}', model)
        time.sleep(1)
        cmd.alignto(cl_models[0], method='align', object=f'all_to_{cl_models[0]}')
        print("\nAlignment done!\n")
        # color tags
        color_tags()
        # label tags
        label_tags(tag_repr_file)
        # Save session
        cmd.save(f'{hc_dir}/cl{cl}.pse')
        cmd.reinitialize()


def tag_dispersion(session):
    """
    Read data from PyMOL session of a given
    tag cluster and output the mean of distance
    and standard deviation from all model positions
    to the centroid as a measure of dispersion.
    --------------
    session: name.pse
    """
    tags_repr = pd.read_csv(params_2.tag_repr_file)
    # Load session
    # cmd.load(f"{params_2.rmsd_dir}/{session}")
    cmd.load(f"{tags_path}/{session}")

    # Create pseudo-atom as the centroid of each tag position and calculate mean and stdev
    # of distances from each position to the centroid as a measure of dispersion
    with open(f'{tags_path}/dispersion_{session.split(".")[0]}.csv', 'w') as f:
        f.write(f'name,mean_distance,stdev\n')
        for name in tags_repr.name.to_list():
            try:
                # print(name)
                cmd.pseudoatom(f'com_{name}', selection=name, vdw=1, color="white")
                cmd.show_as("spheres", selection=f'com_{name}')
                cmd.set("sphere_transparency", 0.2, selection=name)

                # Get list of coordinates for a given tag position from all models
                coords = cmd.get_model(name).get_coord_list()

                # Ref/ori is the centroid
                ori = cmd.get_coords(f'com_{name}')
                # print(ori)
                mean_distance = np.round(np.mean(cdist(coords, ori)), 3)
                stdev_distance = np.round(np.std(cdist(coords, ori)), 3)
                f.write(f'{name},{mean_distance},{stdev_distance}\n')
            except pymol.CmdException:
                pass

    # cmd.save(f"{params_2.rmsd_dir}/{session}")


def plot_dispersion(file, name):
    """
        Point plot with error bars from dispersion file
        -----------
        file: name of the csv file
        name: name of the cluster
    """
    # data = pd.read_csv(f"{params_2.rmsd_dir}/{file}")
    data = pd.read_csv(f"{tags_path}/{file}")
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_title(f'Dispersion {name}', fontdict={"fontsize": 20}, loc="center", pad=15.0)
    ax.errorbar(x=[x * 0.1 for x in list(range(len(data)))], y=data.mean_distance, yerr=data.stdev,
                fmt="^", c="orange", capsize=5, markersize=7)
    ax.set_xticks([x * 0.1 for x in list(range(len(data)))])
    ax.set_xticklabels(labels=data.name, rotation=40, ha="right")
    ax.set_xlabel(xlabel="Tags", fontdict={"fontsize": 20}, labelpad=10.0)
    ax.set_ylabel(ylabel="distance (A)", fontdict={"fontsize": 20}, labelpad=10.0)
    plt.tight_layout()
    plt.savefig(f"{tags_path}/dispersion_{name}.png", dpi=100)


if __name__ == '__main__':
    align = True
    dispersion = True
    dispersion_2 = False
    do_rmsd_clustering = False
    if align:
        hc_dir = '../output/hc/'
        cl_file = hc_dir + 'clustering_A.csv'
        pdbs_dir = hc_dir + 'tmp_pbds/models_A/'
        tags_rep = '../input/data/tags_representation.csv'
        align_tags(cl_file, pdbs_dir, hc_dir, tags_rep)
    if dispersion:
        tag_dispersion("cl1.pse")
        plot_dispersion('dispersion_cl1.csv', 'Cluster_1')
        tag_dispersion("cl2.pse")
        plot_dispersion('dispersion_cl2.csv', 'Cluster_2')
    if do_rmsd_clustering:
        data = pd.read_csv(f'{params_2.rmsd_dir}/clustering.csv')
        sel_models = data.models.to_list()
        z = rmsd_clustering(sel_models, params_2.pdbs_dir, verbose=True)
        plot_clustering('rmsd_tags.png', z)
    if dispersion_2:
        tags_path = sys.argv[1]
        cl = sys.argv[2]
        tag_dispersion(f"{cl}.pse")
        plot_dispersion(f'dispersion_{cl}.csv', f'{cl}')
    print('\nDONE!\n')
    sys.exit()
