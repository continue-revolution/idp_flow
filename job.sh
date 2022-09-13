#!/bin/bash
# The interpreter used to execute the script

#"#SBATCH" directives that convey submission options:

#SBATCH --job-name=train_idp
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=48:00:00
#SBATCH --account=tewaria0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

# The application(s) to execute along with its input arguments and options:

cd idp_flow/
conda init bash
conda activate nf
python -m experiments.train --start_mol=50