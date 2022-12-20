#!/bin/bash

#INTERMODE='compet_only'
INTERMODE='compet_mutual'

#CLUSTER='swalot'
CLUSTER='victini'

INITREP=21
ENDREP=40

for REP in $(seq $INITREP $ENDREP); do
    OUT="job_id${REP}.sh"
    echo -en "#!/bin/bash 

#PBS -l nodes=1:ppn=1

#PBS -l walltime=72:00:00

cd "\$PBS_O_WORKDIR"

module --force purge

module load cluster/${CLUSTER}

bash run_simul_community_assembly.sh ${REP} ${INTERMODE}" > ${OUT}

done


for REP in $(seq $INITREP $ENDREP); do
    OUT="job_id${REP}.sh"
    qsub ${OUT}
    sleep 2
done

