conda activate ctm

DATASET_NAME=cifar
PRED=ve
CKPT=$3
source ./args.sh $DATASET_NAME $PRED

UNET=cbm_unet
DATA_DIR=/root/data/cifar10/
DATASET=cifar10
DATE=0212
USE_16FP=True
RESBLOCK_UPDOWN=True
IS_N2I=True

NUM_CH=64
NUM_RES_BLOCKS=3
ATTN=2,4

# IMG_SIZE=256
# BS=32
IMG_SIZE=32
BS=32

EXP="${DATE}/cbm/$PRED/cif10_${IMG_SIZE}"

SAVE_ITER=1000
# SAVE_ITER=10
FREQ_SAVE_ITER=2500

# NGPU=8
# CUDA_IDX="0,1,2,3,4,5,6,7"
# NGPU=4
# CUDA_IDX="4,5,6,7"
# CUDA_IDX="2,3,6,7"
NGPU=2
CUDA_IDX="0,1"
# NGPU=4
# CUDA_IDX="0,1,2,3"
# CUDA_IDX="4,5,6,7"

GLOBAL_BS=$((NGPU * BS))

# SCHEDULE_SAMPLER=real-uniform
# WEIGHT_SCHEDULE=bridge_karras
SCHEDULE_SAMPLER=ictlognormal_x0
# WEIGHT_SCHEDULE=cm_bridge_karras
WEIGHT_SCHEDULE=cm_bridge_karras_until_x0
LR=0.0001
EMA_RATE=0.9993

WANDB_OFFLINE=True
LOSS_NORM=ph

SOME_FLAGS="
--datasetname=${DATASET_NAME}
--data_dir=${DATA_DIR} 
--dataset=${DATASET} 
${CH_MULT:+ --channel_mult="${CH_MULT}"}
--num_workers=2
--sigma_data ${SIGMA_DATA} 
--sigma_max=${SIGMA_MAX} --sigma_min=${SIGMA_MIN} --cov_xy ${COV_XY}
--save_interval=$SAVE_ITER --debug=True 
--schedule_sampler=${SCHEDULE_SAMPLER} 
--start_scales=10 --end_scales=1280 --total_training_steps=250000
--target_ema_mode=fixed --scale_mode=ict --start_ema=0.0
--global_batch_size=$GLOBAL_BS
${CKPT:+ --resume_checkpoint="${CKPT}"} 
--is_n2i=$IS_N2I
--loss_norm=${LOSS_NORM}
"
WANDB_OFFLINE=True

WANDB_MODE=offline NCCL_P2P_DISABLE=1 CUDA_LAUNCH_BLOCKING=1 OMPI_MCA_opal_cuda_support=true CUDA_VISIBLE_DEVICES=${CUDA_IDX} mpiexec --allow-run-as-root -n $NGPU python scripts/cm_train.py --exp=$EXP \
 --attention_resolutions $ATTN --class_cond False --use_scale_shift_norm True \
  --dropout 0.1 --ema_rate $EMA_RATE --batch_size $BS \
   --image_size $IMG_SIZE --lr $LR --num_channels $NUM_CH --num_head_channels 64 \
    --num_res_blocks $NUM_RES_BLOCKS --resblock_updown $RESBLOCK_UPDOWN ${COND:+ --condition_mode="${COND}"} ${MICRO:+ --microbatch="${MICRO}"} \
     --pred_mode=$PRED \
    --use_fp16 $USE_16FP --weight_decay 0.0 --weight_schedule $WEIGHT_SCHEDULE \
     ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${SOME_FLAGS} --wandb_offline=$WANDB_OFFLINE

# WANDB_MODE=offline OMPI_MCA_opal_cuda_support=true CUDA_VISIBLE_DEVICES=${CUDA_IDX} mpiexec --allow-run-as-root -n $NGPU python scripts/ddbm_train.py --exp=$EXP \
#  --attention_resolutions $ATTN --class_cond False --use_scale_shift_norm True \
#   --dropout 0.1 --ema_rate 0.9999 --batch_size $BS \
#    --image_size $IMG_SIZE --lr 0.0001 --num_channels $NUM_CH --num_head_channels 64 \
#     --num_res_blocks $NUM_RES_BLOCKS --resblock_updown $RESBLOCK_UPDOWN ${COND:+ --condition_mode="${COND}"} ${MICRO:+ --microbatch="${MICRO}"} \
#      --pred_mode=$PRED  --schedule_sampler $SAMPLER ${UNET:+ --unet_type="${UNET}"} \
#     --use_fp16 $USE_16FP --attention_type $ATTN_TYPE --weight_decay 0.0 --weight_schedule bridge_karras \
#      ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"}  \
#       --data_dir=$DATA_DIR --dataset=$DATASET ${CH_MULT:+ --channel_mult="${CH_MULT}"} \
#       --num_workers=8  --sigma_data $SIGMA_DATA --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN --cov_xy $COV_XY \
#       --save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER --debug=False \
#       ${CKPT:+ --resume_checkpoint="${CKPT}"} 