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
| proxyless_mobile_14  | 150 | 0.0  | 0.1 |580.883 |-|
| proxyless_mobile  | 150 | 0.0  | 0.1 |320.428 |73.41|
| proxyless_gpu  | 150 | 0.0  | 0.1 |465.260 |73.93|
| proxyless_cpu  | 150 | 0.0  | 0.1 |439.244 |-|
| ofa_595  | 150 | 0.0  | 0.1 |512.862 |-|
| ofa_482  | 150 | 0.0  | 0.1 |482.413 |-|
| ofa_398  | 150 | 0.0  | 0.1 |389.488 |-|
| my_600_cifar10  | 150 | 0.0  | 0.1 |570.014 |-|
| my_500_cifar10  | 150 | 0.0  | 0.1 |494.585 |-|
| my_400_cifar10  | 150 | 0.0  | 0.1 |395.348 |-|