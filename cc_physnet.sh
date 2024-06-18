#!/bin/sh
#This is a jobscript to schedule a python script on an exclusive node.
#start the job at graphix.q
#its current working directory
#how much vmem it should use
#The maximum walltime
#The Jobname
#which gpu to use
#where to write the output
#logs into a directory
#$ -q graphix.q
#$ -cwd
#$ -l h_vmem=12G
#$ -l h_cpu=00:24:00
#$ -N "current_calculation"
#$ -l gpu_gen=2080ti
#$ -o $HOME/job_logs

module purge
module load anaconda3/2023.03
module load texlive/2022
module load ffmpeg/4.0.2
module load cuda/11.8.0
conda activate PyMMF
echo -e "\033[1;32m all modules purged, anaconda loaded, Anaconda Enviroment \033[1;31m $(conda info --envs | grep "*" | cut -d " " -f 1) \033[1;32m activated \033[0m"
echo -e "\033[1;34m current directory: \033[0m"
echo $PWD
echo -e "\033[1;34m files in current directory: \033[0m"
ls
# echo -e "\033[1;34m Current modules installed in Anaconda enviroment: \033[0m"
# conda list


echo -e "\033[1;34m CUDA toolkit version: \033[0m"
nvcc --version
echo -e "\033[1;34m Python version: \033[0m"
python --version
echo -e "\033[1;32m start \033[0m"
python current_calculation/current_calculation.py
echo -e "\033[1;36m fertig \033[0m"