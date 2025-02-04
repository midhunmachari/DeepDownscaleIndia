############## RUN COMMANDS ###############

source /nlsasfs/home/precipitation/midhunm/Conda/bin/activate
conda activate tf2

python3 ${PWD}/main.py \
--epochs 2 \
--path ${PWD} \
--dpath ${PWD}/../../../DATASET/DATA_IND32 \
--prefix rnx
