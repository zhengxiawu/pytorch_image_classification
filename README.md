## Requirements
- Pytorch 1.0
- Python 3.6+
- DALI


```bash
# Quick Training
./script/data_to_memory.sh imagenet64
python train_dali.py --dataset ImageNet64 --data_path /userhome/temp_data/cifar10 --model_method proxyless_NAS --model_name proxyless_gpu

cd /userhome/project/pytorch_image_classification; 
./script/data_to_memory.sh cifar10;
python train_dali.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --model_method manual --model_name MobileNetV2

```
```bash
# High performance Training
nvidia-smi; 
cd /userhome/project/pytorch_image_classification; 
./script/data_to_memory.sh imagenet;
python train.py --auto_augmentation --dropout_rate 0.3 --data_path /userhome/temp_data/ImageNet --workers 16 --model_method proxyless_NAS --model_name ofa_398 

```
