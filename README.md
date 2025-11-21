# Gaussian-Based Instance-Adaptive Intensity Modeling for Point-Supervised Facial Expression Spotting

## Introduction

Official Pytorch Implementation of 'Gaussian-Based Instance-Adaptive Intensity Modeling for Point-Supervised Facial Expression Spotting' (ICLR2025)

Even though the SpotFormer has not been accepted yet, we decide to release the source code.

However, the current version has not been organized well, we will continue to organize it.

This is an improved version from the conference version.

## Train
``` bash
python main.py --lr 0.00002 --dataset-name CASME2 --path-dataset dataset/ --num-class 2 --use-model CO2  --max-iter 100 --interval 1  --dataset SampleDataset --weight_decay 0.1 --model-name CO2_3552 --seed 3552 --cuda 0 --dirname AAA --batch-size 1
```


## Validation
``` bash
python -W ignore validation_only.py --dataset-name CASME2 --path-dataset dataset/ --num-class 2 --use-model CO2  --max-iter 100  --dataset SampleDataset --model-name CO2_3552 --seed 42 --pretrained-ckpt ckptours02/best_CO2_3552 --batch-size 1 --cuda 1
```

## Acknowlegment
We implement our source code based on [CO2Net](https://github.com/harlanhong/MM2021-CO2-Net/tree/master).
