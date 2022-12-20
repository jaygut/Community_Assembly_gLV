#!/bin/bash -l

###Use script to submit worker arrays, with 18/20 jobs (victini/swalot) being packed per array submitted to a given cluster
###The number of arrays submitted is determined by NUMCHUNKS

#module swap cluster/swalot
module swap cluster/victini

###Load worker node module
ml load worker/1.6.12-foss-2019a

###Invoke python to generate subranges indicating RepIDs
CHUNKLEN=18
NUMCHUNKS=10
array=`python -c "for i in list(range(0,($CHUNKLEN)*$NUMCHUNKS,($CHUNKLEN))):print(i)"`
for i in $array; do
    lb=$i
    lim=$((CHUNKLEN-1))
    up=$(( i + lim ))
#     echo $lb","$up
    wsub -t $lb-$up -batch run_worker.pbs
done


