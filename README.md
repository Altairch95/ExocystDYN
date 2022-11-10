----------------
SCRIPTS
---------------
For the modeling I use the following scripts:
	
	1) mc_tags_exocyst.py: main script for modeling using IMP.pmi functions.
	
	- I represent the fluorophores as beads of 1 residue (Lysine), and the exocyst subunits using the cryoEM model (Mei et al, 2018) and AlphaFold2 predictions for the missing regions. 
	- In vivo restraints (PICT) as Simple Harmonic distance restraints between fluorophores, defining a k from the stdev of each pairwise distance distribution. Modeling with different k values: k(sd), k=1, k=3, k=5, k=10
	- Linkers between fluorophores and proteins as Harmonic Upper Bound restraints with a k=1. Each fluorophore tags the termini of a individual exocyst subunit.
	- Connectivity restraint for each exocyst subunit (k=1)
	- Excluded Volume restraint for all components of the system (k=1)
	- Defining a bounding box to run the modeling and narrow down the possibilities. To define the size of the BB, I use the maximum distance restraint.
	
	- Sampling: 1000 MC frames x 50 steps/frame = 50.000 MC steps per run. Using 1 replica only. To use multiple replicas, use different cores with $ mpirun.
	
	Command to run several MC simulations:
	
		$parallel -j 10 python3 mc_tags_exocyst.py {} ::: {1..30} & --> generate 30 run_ folders.
	
	2) analysis_traj.py: analyse all the run_ simulations generated.
	
	Command: 
	
		$ analysis_traj.py -h
	
	- specify restraints to analyse.
	- Specify cores to use
	- Score-based clustering with HDBscan
	- Generates plot_clustering_scores.png
	- Generates plot_run_models_clusterX.pdf
	- Generates plot_scores_convergence_clusterX.pdf
	- Generates summary_hdbscan_clustering.dat
	
	3) extract_models.py: Extract good-scoring models from a score-based clustering.
	
	Command: 
	
		$ extract_models.py -h
		
	- Specify cluster and state to analyse.
	- Generates A_models_clustX_stateY.rmf3 and B_models_clustX_stateY.rmf3
	- Generates A_models_clustX_stateY.txt and B_models_clustX_stateY.txt
	
	4) check_in_vivo_restraints.py: check the % of in vivo restraints accomplished in all the frames for each RMF file.
	
	Command: 
	
		$ check_in_vivo_restraints.py -h
	
	- reference data in pict_restraints.csv
	- Generates pict_modelname_A.png and pict_modelname_B.png
	
	5) sampcon.sh: computes the model precision based on RMSD
	
	Command: 
	
		$ sampcon.sh 
		
	- Specify densities.txt 
	- Specify cluster and state to analyse
	- Generates clusters of precision
	- Generates clustering.log
	- Generates ChiSquare.pdf / Cluster_Population.pdf / Score_Dist.pdf / Top_Score_Conv.pdf and the respective csv and txt files
	
	6) align_rmf.py: align frames from RMF files A and B to the reference cluster determined with Sampcon, and compute the RMSD to the reference.
	
	Command: 
	
		$ align_rmf.py
	- Plots RMSD density distribution and mean RMSD 
	
	 process_aligned_rmf.py: computes the mean dispersion from each component centroid and plots the error bars
	 
	
	
	
	


