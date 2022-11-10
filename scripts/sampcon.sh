module load imp
#module load hdf5/1.8.17
module load python3/pyrmsd

export cl=2
export st=0

cp ../A_models_clust${cl}_${st}.txt ../scoresA.txt
cp ../B_models_clust${cl}_${st}.txt ../scoresB.txt
cp ../density_fluorophores.txt density.txt
nohup time python /home/altair/PycharmProjects/UCSF_colab/imp-sampcon/pyext/src/exhaust.py \
       -a -n fluorophores -p ../ -ra A_models_clust${cl}_${st}.rmf3 -rb \
       B_models_clust${cl}_${st}.rmf3 \
       -m cpu_omp -c 8 -g 2.0 -gp > clustering.log &


#nohup python /home/altair/PycharmProjects/UCSF_colab/imp-sampcon/pyext/src/exhaust.py \
#       -n DDI1_DDI2 -p ../ -ra A_models_clust${cl}_${st}_aligned.rmf3 -rb \
#       B_models_clust${cl}_${st}_aligned.rmf3 -d density.txt \
#       -m cpu_omp -c 8 -g 2.0 > clustering.log &
