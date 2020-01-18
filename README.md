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
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method proxyless_NAS --model_name proxyless_gpu 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method proxyless_NAS --model_name proxyless_cpu 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method proxyless_NAS --model_name proxyless_mobile 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method proxyless_NAS --model_name proxyless_mobile_14 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method proxyless_NAS --model_name ofa_595 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method proxyless_NAS --model_name ofa_482 
python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --data_loader_type torch --auto_augmentation --cutout_length 16 --epochs 600 --model_method proxyless_NAS --model_name ofa_398 

# Training with ImageNet torch on different neural networks
python train.py --dataset ImageNet --data_path /gdata/ImageNet2012 --data_loader_type dali --drop_path_prob 0.2 --aux_weight 0.4 --init_channels 48 --layers 14 --epochs 250 --model_method darts_NAS --model_name MDENAS
python train.py --dataset ImageNet --data_path /gdata/ImageNet2012 --data_loader_type dali --model_method proxyless_NAS --model_name proxyless_gpu 

python train.py --dataset ImageNet --data_path /userhome/temp_data/cifar10 --model_method proxyless_NAS --model_name proxyless_gpu

python train.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --model_method manual --model_name MobileNetV2

```
```bash
# High performance Training
nvidia-smi; 
cd /userhome/project/pytorch_image_classification; 
./script/data_to_memory.sh imagenet;
python train.py --auto_augmentation --dropout_rate 0.2 --data_path /userhome/temp_data/ImageNet --workers 16 --model_method proxyless_NAS --model_name ofa_398 
```
