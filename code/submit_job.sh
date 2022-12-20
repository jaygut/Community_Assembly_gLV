#!/bin/bash -l
#PBS -l nodes=1:ppn=1
#PBS -l walltime=72:00:00

cd $PBS_O_WORKDIR

REP=$1
#REP=100
INTERMODE=$2
#INTERMODE='compet_only'
#INTERMODE='compet_mutual'

module --force purge
#module load cluster/swalot
module load cluster/victini

bash run_simul_community_assembly.sh ${REP} ${INTERMODE}
