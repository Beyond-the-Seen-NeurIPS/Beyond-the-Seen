#!/bin/bash

# custom config
DATA=/path/to/dataset/folder
TRAINER=CasPL_Beyond
TRAINER2=CasPL

DATASET=$1
WEIGHT=$2
SEED=$3

CFG=vit_b16_c2_ep20_batch4_8+8ctx_l12
SHOTS=16
COMMON_DIR=${DATASET}/shots_all/${TRAINER2}/${CFG}/seed1
MODEL_DIR=/path/to/teacher/first_stage/boosting_prompt/${COMMON_DIR}
DIR=/path/to/output/student/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}_mmdloss${WEIGHT}/seed${SEED}

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR}\
    --load-epoch 20 \
    --klloss \
    --mmdloss \
    --mmdloss_weight ${WEIGHT} \
    --dataset ${DATASET} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    KD.IS_TEST True \
    KD.FIRST_TRAIN False \
    KD.SECOND_TRAIN True \
    KD.DIR_NAME student"&"${SEED} \
    KD.RESULT_PATH /path/to/output/Beyond/caspl_beyond_base_to_novel/output_base_to_novel/ \
    KD.N_CTX_VISION 8 \
    KD.N_CTX_TEXT 8
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR}\
    --load-epoch 20 \
    --klloss \
    --mmdloss \
    --mmdloss_weight ${WEIGHT} \
    --dataset ${DATASET} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    KD.IS_TEST True \
    KD.FIRST_TRAIN False \
    KD.SECOND_TRAIN True \
    KD.DIR_NAME student"&"${SEED} \
    KD.RESULT_PATH /path/to/output/Beyond/caspl_beyond_base_to_novel/output_base_to_novel/ \
    KD.N_CTX_VISION 8 \
    KD.N_CTX_TEXT 8
fi