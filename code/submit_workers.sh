#!/bin/bash

ml load worker/1.6.12-foss-2019a

for INTERMODE in 'compet_only' 'compet_mutual';
  do
    wsub -t 1-25 -batch run_worker.pbs ${INTERMODE}
  done

