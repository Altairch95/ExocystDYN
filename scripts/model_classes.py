#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python functions to model the protein complexes (exocyst and cog e.g.).

Modeling is performed from distance measurements between subunits of the complex.
Distances are read from a csv file, create IMP particles, IMP
decorators (coordinates), assign initial coordinates and apply distance restraints to build
up the tag model or 3D architecture of the protein complex.
"""

import sys
import matplotlib.pyplot as plt

import params_2
import os
import glob
import numpy as np
import pandas as pd

import IMP
import IMP.algebra
import IMP.core
import IMP.atom
import IMP.container
import IMP.display

import seaborn as sns
from Bio.PDB import *
# import fastcluster
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.spatial.distance import pdist

sns.set(color_codes=True)

__author__ = "Altair C. Hernandez"
__copyright__ = "Copyright 2022, The Exocyst Modeling Project"
__credits__ = ["Ibai Irastorza", "Damien P. Devos", "Oriol Gallego"]
__version__ = "IMP 2.17.0"
__maintainer__ = "Altair C. Hernandez"
__email__ = "altair.chinchilla@upf.edu"
__status__ = "Development"


class tag_model:
    """
    Class to build a 'tag' model with IMP.
    Here 'tag' means 'fluorophore tagging a
    protein subunit on its N- or C- termini.

    This class uses the Integrative Modeling Platform
    (IMP) (Russel et al., 2012) developed in Andrej
    Ŝali's lab to trilaterate the fluorophore positions
    in 3D space given a set of pairwise distances from
    a PICT experiment (Picco et al., 2017,
    Gallego et al., 2013).
    """

    def __init__(self, name):
        self.model_name = name
        self.model = IMP.Model()
        self.particles = dict()
        self.decorators = list()
        self.pict_restraints = list()
        self.cg_scores = list()
        self.fulfilled = list()
        self.sf = None
        self.optimizer = None
        self.bounding_box = None

    @property
    def imp_score(self):
        """
        IMP score of the model.
        """
        if self.sf is not None:
            return self.sf.evaluate(True)
        else:
            return None

    @property
    def acceptable_restraints(self):
        """
        Tant percent of restraints fulfilled after
        optimization.
        """
        if len(self.fulfilled) != 0:
            return np.round(np.count_nonzero(list(zip(*self.fulfilled))[1]) * 100 /
                            len(list(zip(*self.fulfilled))[1]), 3)
        else:
            return None

    def set_3D_bounding_box(self, dim=int()):
        """
        Set 3D bounding box for modeling.
        -----------
        dim: dimentions of the box (i.e., 100 x 100 x 100)
        """
        dim = dim / 2
        self.bounding_box = IMP.algebra.BoundingBox3D(IMP.algebra.Vector3D(0 - dim, 0 - dim, 0 - dim),
                                                      IMP.algebra.Vector3D(0 + dim, 0 + dim, 0 + dim))

    def set_particles(self, names=None, radius=float()):
        """
        Set IMP particles given a list of desired particles
        names and a radius
        """
        # Create particles as spheres of a given radius
        # and assign random position inside a pre-defined
        # bounding box
        if self.bounding_box is not None:
            for mol in names:
                particle = IMP.Particle(self.model, mol)
                decorator = IMP.core.XYZR.setup_particle(particle,
                                                         IMP.algebra.Sphere3D(
                                                             IMP.algebra.get_random_vector_in(self.bounding_box),
                                                             float(radius)))
                decorator.set_coordinates_are_optimized(True)
                self.particles.setdefault(mol, particle)
        else:
            # User should set a bounding box to assign particle positions
            sys.stderr.write('You must set bounding box first\n')
            sys.exit(1)

    def set_harmonic_restraints(self, df):
        """
        Set Harmonic restraints from dataframe:
        df.columns = ['bait','prey','distance','distance_se','sigma','sigma_se','n']
        where:
            - bait: name of bait-FRB protein
            - prey: name of prey-GFP protein
            - distance: estimated mu parameter from distance
            distribution between bait and prey (angstroms)
            - distance_se: estimated standard error from the distance (angstroms)
            - sigma: estimated sigma parameter from distance
            distribution between bait and prey (angstroms)
            - sigma_se: estimated standard error from the distance (angstroms)
            - n: number of distances (observations) from where the 'distance',
            'distance_se','sigma', and 'sigma_se' have been calculated.

        The Harmonic restraint is a simple score modeling a harmonic oscillator.
        The score is 0.5 * k * x * x, where k is the 'force constant' and x is
        the distance from the mean.

        Here, the k ('force constant' or 'spring parameter') is calculated from
        the 'sigma' as a representation of the uncertainty around the 'distance'
        collected from PICT.
        -------------------
        df: dataframe with input PIC data
        """
        # Pair Harmonic upperbound restraints
        for row in df.itertuples():
            k = IMP.core.Harmonic_get_k_from_standard_deviation(sd=row.sigma)
            f = IMP.core.Harmonic(row.distance, k)
            s = IMP.core.DistancePairScore(f)
            r = IMP.core.PairRestraint(self.model, s, (self.particles[row.bait], self.particles[row.prey]))
            r.set_name(f"{row.bait}-{row.prey}")
            self.pict_restraints.append(r)

    def optimize_cg(self, max_steps=int()):
        """
        Optimize model using Conjugate Gradients optimization
        (Steepest descend algorithm).

        This algorithm optimizes a scoring function by minimization,
        maximizing the probability to find a model configuration
        in agreement with the input restraints. Therefore, the lower
         is the score, the more in agreement our model is with the imposed
        restraints.
        -------------
        max_steps: max number of steps to explore good-scoring solutions
        """
        # Set scoring function optimizer
        self.sf = IMP.core.RestraintsScoringFunction(self.pict_restraints, "sf")
        self.optimizer = IMP.core.ConjugateGradients(self.model)
        self.optimizer.set_scoring_function(self.sf)
        # Run a max steps of CG steepest descend
        # optimization and break if a minima is found
        score = self.sf.evaluate(True)
        self.optimizer.optimize(500)
        # for step in range(0, max_steps, 25):
        #     self.cg_scores.append((step, score))
        #     self.optimizer.optimize(step)
        #     if self.sf.evaluate(True) < score:
        #         score = self.sf.evaluate(True)
        #     else:
        #         break

    def get_bounding_box(self):
        """
        Return IMP bounding box
        """
        return self.bounding_box

    def get_particles(self):
        """
        Return particles dictionary
        """
        return self.particles

    def get_restraints(self):
        """
        Return list of harmonic pairwise restraints
        (PICT restraints).
        """
        return self.pict_restraints

    def get_cg_scores(self):
        """
        Return list of tuples (step, score)
        """
        return self.cg_scores

    def check_pict_restraints(self, df):
        """
        Check how many PICT restraints have been fulfilled
        after optimization
        ------------------
        df: dataframe with input PIC data
        """
        # Check if restraints have been fulfilled
        for row in df.itertuples():
            bait_coords = IMP.core.XYZ(self.particles[row.bait])
            prey_coords = IMP.core.XYZ(self.particles[row.prey])
            opt_distance = IMP.core.get_distance(bait_coords, prey_coords)
            if row.distance - (2 * row.distance_se) <= opt_distance <= row.distance + (2 * row.distance_se):
                self.fulfilled.append((f'{row.bait}-{row.prey}', 1))
            else:
                self.fulfilled.append((f'{row.bait}-{row.prey}', 0))

    def write_pdb(self, representation_df, out_file):
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
            pdbf.write(f'KEYWDS    IMP-score: {self.imp_score}, Acceptable restraints: {self.acceptable_restraints}%\n')
            for molecule in representation_df.itertuples():
                x, y, z = list(IMP.core.XYZ(self.particles[molecule.name]).get_coordinates())
                pdbf.write("{:>4}{:>7}{:>9}{:>2}{:>4}{:12.3f}{:8.3f}{:8.3f}{:>12}{:>13}\n".format(
                    "ATOM", molecule.Index + 1, "CA LYS", molecule.chain, molecule.Index + 1, x, y, z, "1.00 36.92", "C"
                ))

    def write_log(self, filename, tag_representation_df):
        """
        Write log (information) file for your model, to get
        track on the basic model information, restraints used,
        number of restraints fulfilled, and PDB information.
        ------------------
        filename: output.log file name
        tag_representation_df: dataframe with representation columns:
        ['name','chain','color'] where:
            - name: name of IMP particle.
            - chain: PDB chain to be assigned.
            - color: (optional) for chimeraX visualization

        """
        with open(filename, "w") as log:
            # Write model information (Score and % acceptance restraints)
            log.write('>Model_info:\n'
                      'score,in_vivo_%\n'
                      f'{self.imp_score},{self.acceptable_restraints}\n')
            log.write('#\n')
            # Write restraints
            log.write('>Pair_Restraints:\n'
                      'pair,is_fulfilled\n')
            for pair_r, val in self.fulfilled:
                log.write(f'{pair_r},{val}\n')
            log.write('#\n')
            # Write Scores
            log.write('>Scores:\n'
                      'steps,score\n')
            for stp, scr in self.cg_scores:
                log.write(f'{stp},{scr}\n')
            log.write('#\n')
            # Write PDB
            log.write('>PDB:\n')
            for molecule in tag_representation_df.itertuples():
                x, y, z = list(IMP.core.XYZ(self.particles[molecule.name]).get_coordinates())
                log.write("{:>4}{:>7}{:>9}{:>2}{:>4}{:12.3f}{:8.3f}{:8.3f}{:>12}{:>13}\n".format(
                    "ATOM", molecule.Index + 1, "CA LYS", molecule.chain, molecule.Index + 1, x, y, z, "1.00 36.92", "C"
                ))


def test_create_tags(verbose=params_2.verbose):
    """
        Test function to create tag models called 'Model_1'.
        It outputs the corresponding PDB and log files.
    """
    # Paths and Parameters
    input_path = params_2.input_path
    output_path = params_2.output_path
    # I/O files
    data_file = f"{input_path}/pict_data_cell.csv"
    tag_repr_file = f"{input_path}/tags_representation.csv"
    model_info_file = f"{output_path}/model_test.log"
    # Model Params
    tag_radius = params_2.tag_radius
    cg_steps = params_2.cg_steps
    bb_dimention = params_2.bb_dim

    # Check if input and output files and paths exists
    if not os.path.exists(input_path):
        raise FileNotFoundError
    if not os.path.exists(output_path):
        raise FileNotFoundError
    if not os.path.exists(data_file):
        raise FileNotFoundError
    if not os.path.exists(tag_repr_file):
        raise FileNotFoundError

    # Read data as dataframe
    data = pd.read_csv(data_file)
    tag_representation = pd.read_csv(tag_repr_file)
    # convert nm to angstrom
    data.loc[:, ['distance', 'distance_se', 'sigma', 'sigma_se']] = data.copy().loc[:,
                                                                    ['distance', 'distance_se',
                                                                     'sigma', 'sigma_se']].apply(lambda k: k * 10)
    # Molecule names for IMP
    mol_names = np.concatenate((data.bait.unique(), data.prey.unique()))
    # --------
    # Modeling
    # --------
    m = tag_model('Model_1')
    m.set_3D_bounding_box(bb_dimention)
    m.set_particles(mol_names, tag_radius)
    m.set_harmonic_restraints(data)
    m.optimize_cg(cg_steps)
    m.check_pict_restraints(data)
    m.write_pdb(tag_representation, f'{output_path}/model_test.pdb')
    m.write_log(model_info_file, tag_representation)

    if verbose:
        print(f'{m.model_name} created successfully!\n')


def main_create_tags(model_name, out_pdb=params_2.out_pdb,
                     verbose=params_2.verbose):
    """
        Main function to create tag models.
        It outputs the corresponding log files.
        -----------
        model_name: 'x' numer of the name "modelx"
        out_pdb: bool to write output PDB files.
        verbose: bool to verbose the process.
    """
    # Paths and Parameters
    input_path = params_2.input_path
    output_path = params_2.output_path
    # I/O files
    data_file = params_2.pict_data_file
    tag_repr_file = params_2.tag_repr_file
    tmp_dir = params_2.tmp_dir
    # Model Params
    tag_radius = params_2.tag_radius
    cg_steps = params_2.cg_steps
    bb_dim = params_2.bb_dim

    # Check if input and output files and paths exists
    if not os.path.exists(input_path):
        raise FileNotFoundError
    if not os.path.exists(output_path):
        raise FileNotFoundError
    if not os.path.exists(data_file):
        raise FileNotFoundError
    if not os.path.exists(tag_repr_file):
        raise FileNotFoundError
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    # Read data as dataframe
    data = pd.read_csv(data_file)

    # convert nm to angstrom
    data.loc[:, ['distance', 'distance_se', 'sigma', 'sigma_se']] = data.copy().loc[:,
                                                                    ['distance', 'distance_se',
                                                                     'sigma', 'sigma_se']].apply(lambda k: k * 10)
    # Molecule names for IMP
    mol_names = np.concatenate((data.bait.unique(), data.prey.unique()))
    # --------
    # Modeling
    # --------
    m = tag_model(model_name)
    m.set_3D_bounding_box(bb_dim)
    m.set_particles(mol_names, tag_radius)
    m.set_harmonic_restraints(data)
    m.optimize_cg(cg_steps)
    m.check_pict_restraints(data)
    # m.write_log(f"{tmp_dir}/{model_name}.log", tag_representation)
    if verbose:
        print(f'{m.model_name} created successfully!\n')

    return m


def get_best_scoring(tmp_dir):
    """
    Get best scoring models after running tags.py

    tmp_dir: tmp dir with .log files to get best scoring models
    """
    models = dict()
    for log_file in glob.glob(f'{tmp_dir}/model*.log'):
        with open(log_file, 'r') as file:
            for line in file:
                try:
                    if line.startswith('>Model_info:'):
                        file.readline()
                except FileNotFoundError:
                    pass


def rmsd(models):
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
    return np.array(rmsd_list)


def rmsd_clustering(sel_models, pdbs_dir, verbose=params_2.verbose):
    """
    RMSD-based clustering from selected custom PDBs
    """
    p = PDBParser(QUIET=True)
    if not os.path.exists(pdbs_dir):
        raise FileNotFoundError
    models = list()
    for model in sel_models:
        if type(model) == tag_model:
            pdb_file = f'{pdbs_dir}/{model.model_name}.pdb'
            models.append(p.get_structure(model.model_name, pdb_file))
        else:
            pdb_file = f'{pdbs_dir}/{model}.pdb'
            models.append(p.get_structure(model, pdb_file))
    if verbose:
        print(f"\n\nClustering with {len(models)} models...\n")

    # Calculating RMSD
    m = rmsd(models)
    if verbose:
        print(f'Mean RMSD = {np.mean(m[:, 0])}')

    # HIERARCHICAL CLUSTERING
    z = linkage(m, method='ward', metric='euclidean')
    if verbose:
        print("\nCophenetic correlation coefficient is {}".format(cophenet(z, pdist(m))[0]))

    # Write clustering assignments csv file
    assignments = fcluster(z,
                           t=3000,  # int(clusters_distance),
                           criterion='distance')  # threshold line to choose clusters
    labels = np.array([s.get_id() for s in models])
    if not os.path.exists(params_2.rmsd_dir):
        os.mkdir(params_2.rmsd_dir)
    pd.DataFrame({'models': labels, 'cluster': assignments}).to_csv(f"{params_2.rmsd_dir}/clustering.csv", index=False)

    return z


def plot_clustering(file,  z):
    """
    Plot RMSD-based clustering (dendrogram)
    """
    if not os.path.exists(params_2.rmsd_dir):
        os.mkdir(params_2.rmsd_dir)
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
    plt.savefig(f'{params_2.rmsd_dir}/{file}', dpi=72)
