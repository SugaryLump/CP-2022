#!/usr/bin/env bash

scp -r src Makefile includes "$1"@search7edu2.di.uminho.pt:~/cp

ssh "$1"@s7edu2.di.uminho.pt "mkdir -p ~/cp && cd ~/cp;
make clean;
module load gcc/7.2.0
make k_means;
srun --partition=cpar --cpus-per-task=$3 perf stat -e L1-dcache-load-misses -M cpi /home/$1/cp/bin/k_means 10000000 $2 $3"