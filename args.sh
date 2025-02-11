BS=64


DATASET_NAME=$1
PRED=$2
# NGPU=1


SIGMA_MAX=80.0
SIGMA_MIN=0.002
SIGMA_DATA=0.5
COV_XY=0

SAMPLER=real-uniform

USE_16FP=True
ATTN_TYPE=flash
if [[ $DATASET_NAME == "cifar" ]]; then
    IMG_SIZE=32
    SAVE_ITER=20000
fi
    
if  [[ $PRED == "ve" ]]; then
    EXP+="_ve"
    COND=concat
elif  [[ $PRED == "vp" ]]; then
    EXP+="_vp"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
elif  [[ $PRED == "ve_simple" ]]; then
    EXP+="_ve_simple"
    COND=concat
elif  [[ $PRED == "vp_simple" ]]; then
    EXP+="_vp_simple"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
else
    echo "Not supported"
    exit 1
fi



if [[ $IMG_SIZE == 256 ]]; then
    BS=4
else
    BS=192
fi