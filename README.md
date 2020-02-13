## Requirements
- Pytorch 1.0.1.post2
- Python 3.6+
- DALI

## Usuage

Step1: Go into your project path

```bash
cd /userhome/project/pytorch_image_classification; 
```

Step2: Move data to memory

```bash
./script/data_to_memory.sh cifar10
./script/data_to_memory.sh imagenet
```
Step3: Start training
```bash
# Training with cifar10 DALI on different neural networks

python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --model_method manual --model_name MobileNetV2 --data_loader_type dali
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --model_method manual --model_name MobileNetV3Large --data_loader_type dali
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --model_method manual --model_name Resnet18 --data_loader_type dali

python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --model_method proxyless_NAS --model_name proxyless_gpu --data_loader_type dali 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --model_method proxyless_NAS --model_name proxyless_cpu --data_loader_type dali 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --model_method proxyless_NAS --model_name proxyless_mobile --data_loader_type dali 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --model_method proxyless_NAS --model_name proxyless_mobile_14 --data_loader_type dali 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --model_method proxyless_NAS --model_name ofa_595 --data_loader_type dali 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --model_method proxyless_NAS --model_name ofa_482 --data_loader_type dali
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --model_method proxyless_NAS --model_name ofa_398 --data_loader_type dali

# init channel 44 epoch 1800 dropout 0.7 will have a higher performance
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type dali --drop_path_prob 0.2 --aux_weight 0.4 --init_channels 36 --layers 20 --epochs 600 --model_method darts_NAS --model_name MDENAS
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type dali --drop_path_prob 0.2 --aux_weight 0.4 --init_channels 36 --layers 20 --epochs 600 --model_method darts_NAS --model_name DDPNAS_V1
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type dali --drop_path_prob 0.2 --aux_weight 0.4 --init_channels 36 --layers 20 --epochs 600 --model_method darts_NAS --model_name DDPNAS_V2
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type dali --drop_path_prob 0.2 --aux_weight 0.4 --init_channels 36 --layers 20 --epochs 600 --model_method darts_NAS --model_name DARTS_V1
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type dali --drop_path_prob 0.2 --aux_weight 0.4 --init_channels 36 --layers 20 --epochs 600 --model_method darts_NAS --model_name DARTS_V2


# Training with cifar10 Torch on different neural networks for high performance 

python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method manual --model_name MobileNetV2 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method manual --model_name MobileNetV3Large 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method manual --model_name Resnet18 


python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method proxyless_NAS --model_name proxyless_gpu 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method proxyless_NAS --model_name proxyless_cpu 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method proxyless_NAS --model_name proxyless_mobile 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method proxyless_NAS --model_name proxyless_mobile_14 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method proxyless_NAS --model_name ofa_595 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method proxyless_NAS --model_name ofa_482 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method proxyless_NAS --model_name ofa_398 

python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --drop_path_prob 0.2 --aux_weight 0.4 --model_method darts_NAS --model_name MDENAS
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --drop_path_prob 0.2 --aux_weight 0.4 --model_method darts_NAS --model_name DDPNAS_V1
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --drop_path_prob 0.2 --aux_weight 0.4 --model_method darts_NAS --model_name DDPNAS_V2
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --drop_path_prob 0.2 --aux_weight 0.4 --model_method darts_NAS --model_name DARTS_V1
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --drop_path_prob 0.2 --aux_weight 0.4 --model_method darts_NAS --model_name DARTS_V2


python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 1800 --init_channels 44 --batch_size 96 --drop_path_prob 0.2 --aux_weight 0.4 --model_method darts_NAS --model_name MDENAS
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 1800 --init_channels 44 --batch_size 96 --drop_path_prob 0.2 --aux_weight 0.4 --model_method darts_NAS --model_name DDPNAS_V1 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 1800 --init_channels 44 --batch_size 96 --drop_path_prob 0.2 --aux_weight 0.4 --model_method darts_NAS --model_name DDPNAS_V2
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 1800 --init_channels 44 --batch_size 96 --drop_path_prob 0.2 --aux_weight 0.4 --model_method darts_NAS --model_name DARTS_V1
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 1800 --init_channels 44 --batch_size 96 --drop_path_prob 0.2 --aux_weight 0.4 --model_method darts_NAS --model_name DARTS_V2

# Training with ImageNet torch on different neural networks

python train.py --dataset imagenet --data_path /userhome/temp_data/ImageNet --data_loader_type dali --drop_path_prob 0.2 --aux_weight 0.4 --init_channels 48 --layers 14 --epochs 300 --model_method darts_NAS --model_name MDENAS
python train.py --dataset imagenet --data_path /userhome/temp_data/ImageNet --data_loader_type dali --model_method proxyless_NAS --model_name proxyless_gpu 

python train.py --dataset imagenet --data_path /userhome/temp_data/ImageNet --data_loader_type torch --epochs 300 --auto_augmentation --drop_path_prob 0.2 --aux_weight 0.4 --init_channels 48 --layers 14 --model_method darts_NAS --model_name MDENAS
python train.py --dataset imagenet --data_path /userhome/temp_data/ImageNet --data_loader_type torch --epochs 300 --auto_augmentation  --model_method proxyless_NAS --model_name proxyless_gpu 
```

## Results
### ImageNet

|Model|Epoch|Dropout|LabelSmooth|FLOPs|Result|
|:----|:----:|:----:|:----:|:----:|:----:|
| MobileNetV2  | 150 | 0.0  | 0.1 |300.774 |71.67|
| MobileNetV3  | 150 | 0.0  | 0.1 |216.590 |72.93|
| proxyless_mobile_14  | 150 | 0.0  | 0.1 |580.883 |75.28|
| proxyless_mobile  | 150 | 0.0  | 0.1 |320.428 |73.41|
| proxyless_gpu  | 150 | 0.0  | 0.1 |465.260 |73.93|
| proxyless_cpu  | 150 | 0.0  | 0.1 |439.244 |74.15|
| ofa_595  | 150 | 0.0  | 0.1 |512.862 |75.59|
| ofa_482  | 150 | 0.0  | 0.1 |482.413 |75.36|
| ofa_398  | 150 | 0.0  | 0.1 |389.488 |74.61|
| my_600_cifar10  | 150 | 0.0  | 0.1 |570.014 |75.26|
| my_500_cifar10  | 150 | 0.0  | 0.1 |494.585 |74.99|
| my_400_cifar10  | 150 | 0.0  | 0.1 |395.348 |73.36|
| MobileNetV2  | 300 | 0.2  | 0.1 |300.774 |-|
| MobileNetV3  | 300 | 0.2  | 0.1 |216.590 |-|
| proxyless_mobile_14  | 300 | 0.2  | 0.1 |580.883 |-|
| proxyless_mobile  | 300 | 0.2  | 0.1 |320.428 |-|
| proxyless_gpu  | 300 | 0.2  | 0.1 |465.260 |-|
| proxyless_cpu  | 300 | 0.2  | 0.1 |439.244 |-|
| ofa_595  | 300 | 0.2  | 0.1 |512.862 |-|
| ofa_482  | 300 | 0.2  | 0.1 |482.413 |-|
| ofa_398  | 300 | 0.2  | 0.1 |389.488 |-|
| my_600_cifar10  | 300 | 0.2  | 0.1 |570.014 |-|
| my_500_cifar10  | 300 | 0.2  | 0.1 |494.585 |-|
| my_400_cifar10  | 300 | 0.2  | 0.1 |395.348 |-|

### CIFAR-10

|Model|Epoch|Dropout|LabelSmooth|FLOPs|Result|
|:----|:----:|:----:|:----:|:----:|:----:|
| MobileNetV2  | 300 | 0.0  | 0.1 | 6.125  |82.73|
| MobileNetV3Large  | 300 | 0.0  | 0.1 | 7.087  |83.00|
| Resnet18  | 300 | 0.0  | 0.1 |555.42  |93.59|
| my_400  | 300 | 0.0  | 0.1 |10.641  |86.30|
| my_500  | 300 | 0.0  | 0.1 |13.323  |86.35|
| my_600  | 300 | 0.0  | 0.1 |15.236  |85.82|
| ofa_398  | 300 | 0.0  | 0.1 |12.115  |84.02|
| ofa_482  | 300 | 0.0  | 0.1 |14.386  |85.37|
| ofa_595  | 300 | 0.0  | 0.1 |15.029  |85.72|
| proxyless_cpu  | 300 | 0.0  | 0.1 | 8.949  |82.85|
| proxyless_gpu  | 300 | 0.0  | 0.1 | 9.477  |80.99|
| proxyless_mobile  | 300 | 0.0  | 0.1 | 6.526  |81.28|
| proxyless_mobile_14  | 300 | 0.0  | 0.1 |11.836  |82.91|

## Experiments

### CIFAR-10
|Model|Epoch|Dropout|LabelSmooth|FLOPs|Result|
|:----|:----:|:----:|:----:|:----:|:----:|
| ofa__dataset_cifar10_width_multi_1.2_epochs_200_data_split_10_warm_up_epochs_0_lr_0.01_pruning_step_3:100  | 300 | 0.0  | 0.1 | 4.162  |79.74|
| ofa__dataset_cifar10_width_multi_1.2_epochs_200_data_split_10_warm_up_epochs_0_lr_0.01_pruning_step_3:200  | 300 | 0.0  | 0.1 | 7.889  |80.94|
| ofa__dataset_cifar10_width_multi_1.2_epochs_200_data_split_10_warm_up_epochs_0_lr_0.01_pruning_step_3:300  | 300 | 0.0  | 0.1 | 9.920  |84.08|
| ofa__dataset_cifar10_width_multi_1.2_epochs_200_data_split_10_warm_up_epochs_0_lr_0.01_pruning_step_3:400  | 300 | 0.0  | 0.1 |12.271  |85.55|
| ofa__dataset_cifar10_width_multi_1.2_epochs_200_data_split_10_warm_up_epochs_0_lr_0.01_pruning_step_3:500  | 300 | 0.0  | 0.1 |14.274  |85.37|
| ofa__dataset_cifar10_width_multi_1.2_epochs_200_data_split_10_warm_up_epochs_0_lr_0.01_pruning_step_3:600  | 300 | 0.0  | 0.1 |14.984  |85.67|
| proxyless__dataset_cifar10_width_multi_1.3_epochs_200_data_split_10_warm_up_epochs_0_lr_0.01_pruning_step_3:100  | 300 | 0.0  | 0.1 | 1.965  |79.62|
| proxyless__dataset_cifar10_width_multi_1.3_epochs_200_data_split_10_warm_up_epochs_0_lr_0.01_pruning_step_3:200  | 300 | 0.0  | 0.1 | 4.083  |83.75|
| proxyless__dataset_cifar10_width_multi_1.3_epochs_200_data_split_10_warm_up_epochs_0_lr_0.01_pruning_step_3:300  | 300 | 0.0  | 0.1 | 6.124  |82.54|
| proxyless__dataset_cifar10_width_multi_1.3_epochs_200_data_split_10_warm_up_epochs_0_lr_0.01_pruning_step_3:400  | 300 | 0.0  | 0.1 | 8.127  |84.87|
| proxyless__dataset_cifar10_width_multi_1.3_epochs_200_data_split_10_warm_up_epochs_0_lr_0.01_pruning_step_3:500  | 300 | 0.0  | 0.1 |10.148  |84.70|
| proxyless__dataset_cifar10_width_multi_1.3_epochs_200_data_split_10_warm_up_epochs_0_lr_0.01_pruning_step_3:600  | 300 | 0.0  | 0.1 |12.123  |85.51|
| ofa_cifar10_width_multi_1.2_epochs_1000_data_split_10_warm_up_epochs_0_lr_0.01_pruning_step_3:600 | 300 | 0.0 | 0.1 | 13.719 | 85.55 |
| ofa_cifar10_width_multi_1.2_epochs_1000_data_split_10_warm_up_epochs_0_lr_0.1_pruning_step_3:600 | 300 | 0.0 | 0.1 | 16.544 | 86.37 |
| ofa_cifar10_width_multi_1.2_epochs_1000_data_split_10_warm_up_epochs_0_lr_0.1_pruning_step_6:600 | 300 | 0.0 | 0.1 | 12.259 | 86.11 |
| ofa_cifar10_width_multi_1.2_epochs_1000_data_split_10_warm_up_epochs_0_lr_0.1_pruning_step_9:600 | 300 | 0.0 | 0.1 | 15.372 | 86.76 |
| ofa_cifar10_width_multi_1.2_epochs_1000_data_split_10_warm_up_epochs_10_lr_0.1_pruning_step_3:600 | 300 | 0.0 | 0.1 | 12.846 | 85.76 |
| ofa_cifar10_width_multi_1.2_epochs_1000_data_split_10_warm_up_epochs_10_lr_0.1_pruning_step_3_1:600 | 300 | 0.0 | 0.1 | 15.458 | 86.11 |
| ofa_cifar10_width_multi_1.2_epochs_1000_data_split_10_warm_up_epochs_10_lr_0.1_pruning_step_3_2:600 | 300 | 0.0 | 0.1 | 15.444 | 86.48 |
| ofa_cifar10_width_multi_1.2_epochs_1000_data_split_10_warm_up_epochs_10_lr_0.1_pruning_step_3_3:600 | 300 | 0.0 | 0.1 | 15.878 | 86.28 |
| ofa_cifar10_width_multi_1.2_epochs_1000_data_split_2_warm_up_epochs_0_lr_0.1_pruning_step_3:600 | 300 | 0.0 | 0.1 | 13.883 | 86.25 |
| ofa_cifar10_width_multi_1.2_epochs_1000_data_split_5_warm_up_epochs_0_lr_0.1_pruning_step_3:600 | 300 | 0.0 | 0.1 | 16.372 | 86.22 |
| ofa_cifar10_width_multi_1.3_epochs_1000_data_split_10_warm_up_epochs_0_lr_0.1_pruning_step_3:600 | 300 | 0.0 | 0.1 | 17.203 | 86.37 |
| ofa_cifar10_width_multi_1.4_epochs_1000_data_split_10_warm_up_epochs_0_lr_0.1_pruning_step_3:600 | 300 | 0.0 | 0.1 | 16.431 | 86.48 |
| EfficientNet_b0 | 300 | 0.0 | 0.1 | 8.475 | 80.93 |
| FBNet-C | 300 | 0.0 | 0.1 | 7.836 | 79.3 |
| proxyless__dataset_imagenet_width_multi_1.3_epochs_1000_data_split_10_warm_up_epochs_0_lr_0.01_pruning_step_3:600 | 300 | 0.0 | 0.1 | 12.129 | 84.48 |
