# Gaussian-Based Instance-Adaptive Intensity Modeling for Point-Supervised Facial Expression Spotting

## Introduction

Official Pytorch Implementation of 'Gaussian-Based Instance-Adaptive Intensity Modeling for Point-Supervised Facial Expression Spotting' (ICLR2025)

Although SpotFormer has not been officially accepted yet, we have decided to release the source code.

The current version is still somewhat disorganized, and we will continue to clean it up.

This is an improved version compared with the conference submission. [arxiv](https://arxiv.org/abs/2511.16952)

## Data preparation
Feature: [CAS(ME)^2 google drive](https://drive.google.com/file/d/1suimo6F15CKTQTFpx_jmwu5WRMJyt9Ht/view?usp=sharing)
[SAMM-LV google drive](https://drive.google.com/file/d/1utXkaNfs6lVBu9raRrB9npH2xl-912aD/view?usp=sharing)

After downloading the files, move them into the ```dataset/``` directory.


## Train
``` bash
python main.py --lr 0.00002 --dataset-name CASME2 --path-dataset dataset/ --num-class 2 --use-model CO2  --max-iter 100 --interval 1  --dataset SampleDataset --weight_decay 0.1 --model-name CO2_3552 --seed 3552 --cuda 0 --dirname AAA --batch-size 1
```


## Validation
``` bash
python -W ignore validation_only.py --dataset-name CASME2 --path-dataset dataset/ --num-class 2 --use-model CO2  --max-iter 100  --dataset SampleDataset --model-name CO2_3552 --seed 42 --pretrained-ckpt ckptours02/best_CO2_3552 --batch-size 1 --cuda 1
```


## Citation

```
@inproceedings{denggaussian,
  title={Gaussian-Based Instance-Adaptive Intensity Modeling for Point-Supervised Facial Expression Spotting},
  author={Deng, Yicheng and Hayashi, Hideaki and Nagahara, Hajime},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}

@article{denggaussianjournal,
  title={Point-Supervised Facial Expression Spotting with Gaussian-Based Instance-Adaptive Intensity Modeling},
  author={Deng, Yicheng and Hayashi, Hideaki and Nagahara, Hajime},
  journal={arXiv preprint arXiv:2511.16952},
  year={2025}
}

@article{deng2024spotformer,
  title={Spotformer: Multi-scale spatio-temporal transformer for facial expression spotting},
  author={Deng, Yicheng and Hayashi, Hideaki and Nagahara, Hajime},
  journal={arXiv preprint arXiv:2407.20799},
  year={2024}
}

@inproceedings{deng2024multi,
  title={Multi-scale spatio-temporal graph convolutional network for facial expression spotting},
  author={Deng, Yicheng and Hayashi, Hideaki and Nagahara, Hajime},
  booktitle={2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG)},
  pages={1--10},
  year={2024},
  organization={IEEE}
}
```

## Acknowlegment
We implement our source code based on [CO2Net](https://github.com/harlanhong/MM2021-CO2-Net/tree/master).
