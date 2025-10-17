import os

dataset_list = [
    'caltech101', 'dtd', 'eurosat', 'fgvc_aircraft', 'food101', 'imagenet', 'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101'
]

klloss_weight = 1
mmdloss_weight = 0.1
k = 1
GPU_NUM=0

for DATASET in dataset_list:
    print(DATASET)

    for SEED in range(1, 4):
        cmd = f"CUDA_VISIBLE_DEVICES={GPU_NUM} bash scripts/caspl/beyond/base2new_train_beyond_student.sh \
              {DATASET} {klloss_weight} {mmdloss_weight} {k} {SEED}"
        os.system(cmd)
        cmd = f"CUDA_VISIBLE_DEVICES={GPU_NUM} bash scripts/caspl/beyond/base2new_test_beyond_student.sh \
              {DATASET} {klloss_weight} {mmdloss_weight} {k} {SEED}"
        os.system(cmd)















