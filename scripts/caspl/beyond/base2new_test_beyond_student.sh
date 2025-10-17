#!/bin/bash

# custom config
DATA=/path/to/dataset/folder
TRAINER=CasPL_Beyond

DATASET=$1
WEIGHT=$2
SEED=$3

CFG=vit_b16_c2_ep20_batch4_8+8ctx_l12
SHOTS=16
LOADEP=20
SUB=new


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}_mmdloss${WEIGHT}/seed${SEED}
MODEL_DIR=/path/to/output/Beyond/caspl_beyond_base_to_novel/base_to_novel/student/base2new/train_base/${COMMON_DIR}
DIR=/path/to/output/Beyond/caspl_beyond_base_to_novel/base_to_novel/student/base2new/test_${SUB}/${COMMON_DIR}

if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    --mmdloss \
    --mmdloss_weight ${WEIGHT} \
    --dataset ${DATASET} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    KD.IS_TEST True \
    KD.FIRST_TRAIN False \
    KD.SECOND_TRAIN True \
    KD.DIR_NAME student"&"${SEED} \
    KD.RESULT_PATH /path/to/output/Beyond/caspl_beyond_base_to_novel/output_base_to_novel/ \
    KD.N_CTX_VISION 8 \
    KD.N_CTX_TEXT 8
else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    --mmdloss \
    --mmdloss_weight ${WEIGHT} \
    --dataset ${DATASET} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    KD.IS_TEST True \
    KD.FIRST_TRAIN False \
    KD.SECOND_TRAIN True \
    KD.DIR_NAME student"&"${SEED} \
    KD.RESULT_PATH /path/to/output/Beyond/caspl_beyond_base_to_novel/output_base_to_novel/ \
    KD.N_CTX_VISION 8 \
    KD.N_CTX_TEXT 8
fi