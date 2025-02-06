# example command:
# nohup ./src/scripts/run.sh > nohup.out 2>&1 &

# Pretrain
CFG="r3d18"
SEED=10
python train.py -G=0 -C=$CFG seed=$SEED pretrain=True &
CFG="r3d34"
SEED=20
python train.py -G=0 -C=$CFG seed=$SEED pretrain=True &
CFG="r3d50"
SEED=30
python train.py -G=0 -C=$CFG seed=$SEED pretrain=True &
wait

CFG="r3d18"
SEED=10
python train.py -G=0 -C=$CFG fold=-1 weights_path="./data/models_pretrained/${CFG}_seed${SEED}_epoch19.pt" &
python train.py -G=0 -C=$CFG fold=-1 weights_path="./data/models_pretrained/${CFG}_seed${SEED}_epoch19.pt" &
wait
CFG="r3d34"
SEED=20
python train.py -G=0 -C=$CFG fold=-1 weights_path="./data/models_pretrained/${CFG}_seed${SEED}_epoch19.pt" &
python train.py -G=0 -C=$CFG fold=-1 weights_path="./data/models_pretrained/${CFG}_seed${SEED}_epoch19.pt" &
wait
CFG="r3d50"
SEED=30
python train.py -G=0 -C=$CFG fold=-1 weights_path="./data/models_pretrained/${CFG}_seed${SEED}_epoch19.pt" &
python train.py -G=0 -C=$CFG fold=-1 weights_path="./data/models_pretrained/${CFG}_seed${SEED}_epoch19.pt" &
wait