#!/bin/bash -l
for f in *.run; do
    sbatch "$f"
    sleep 0.1
done