#!/bin/bash -l
#Use script to submit a bunch of jobs. Bypassing worker node framework

INTERMODE='compet_only'
#INTERMODE='compet_mutual'

for REP in {11..30}; do
    qsub submit_job.sh $REP $INTERMODE
done
