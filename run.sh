#!/usr/bin/env bash

scp -r src Makefile includes "$1"@search7edu2.di.uminho.pt:~/cp

ssh "$1"@s7edu2.di.uminho.pt "mkdir -p ~/cp && cd ~/cp;
make clean;
make k_means;
srun --partition=cpar perf stat -e L1-dcache-load-misses /home/$1/cp/bin/k_means"
