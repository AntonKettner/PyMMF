# Forces the shell to use bash as the interpreter
#$ -S /bin/bash

# Sends the jobscript to the graphix.q
#$ -q graphix.q

# Tells the shell to use the current working directory
#$ -cwd

# Assigns ...GB of GPU memory to the job
#$ -l h_vmem=12G

# Sets the maximum walltime for the job 1h
#$ -l h_cpu=100:00:00

# Sets the name of the job
#$ -N "skyr_simulation"

# Sets the gpu type to use --> for A100 use the graphit.q
#$ -l gpu_gen=2080ti

# Dir of the joblog file
#$ -o $HOME/job_logs

# Request a parallel environment (PE) with a certain number of CPU slots.
# REMOVE THIS FOR SINGLE THREAD JOBS!
# Available PEs are: mpi (for treads across as few nodes as possible) or smp (for threads on the same node).
# #$ -pe mpi 4

module purge
module load anaconda3/2023.03
module load texlive/2022
module load ffmpeg/4.0.2
module load cuda/11.8.0
conda activate PyMMF_env_light
echo -e "all modules purged, anaconda loaded, Anaconda Enviroment $(conda info --envs | grep "*" | cut -d " " -f 1) activated"
echo "current directory:"
echo $PWD
echo "files in current directory:"
ls

# main part of the script --> executing the python scripts

echo "CUDA toolkit version:"
nvcc --version
echo "Python version:"
python --version
echo -e "\033[1;32m start \033[0m"
python skyrmion_simulation/main.py --sim_type "x_current"
echo -e "\033[1;36m fertig \033[0m"
