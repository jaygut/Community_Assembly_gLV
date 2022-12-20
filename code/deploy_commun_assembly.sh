#!/bin/bash

#Run on command line as:
#bash deploy_commun_assembly.sh compet_only
#OR
#bash deploy_commun_assembly.sh compet_mutual

for SIMREPL in {1..20}
    do
      FILE="commun_assembly_run${SIMREPL}.slurm"
      echo "#!/bin/bash

#SBATCH -J commun_assembly_jobs
#SBATCH -N 1
#SBATCH -n 1             #use site recommended # of cores
#SBATCH -p normal
#SBATCH -o commun_assembly.o%j
#SBATCH -e commun_assembly.e%j
#SBATCH -t 48:00:00
#SBATCH -A mega2014 
#------------------------------------------------------

julia run_community_assembly.jl ${SIMREPL} $1" > $FILE

sbatch $FILE

    done