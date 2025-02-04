#!/bin/bash
#SBATCH --job-name=run_nsx
#SBATCH --partition=testp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=72:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Job id is $SLURM_JOBID"
echo "Job submission directory is : $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

############## RUN COMMANDS ###############

source /nlsasfs/home/precipitation/midhunm/Conda/bin/activate
conda activate tf2

python3 ${SLURM_SUBMIT_DIR}/main.py \
--epochs 500 \
--path ${SLURM_SUBMIT_DIR} \
--dpath ${SLURM_SUBMIT_DIR}/../../../DATASET/DATA_IND32 \
--prefix p07a


