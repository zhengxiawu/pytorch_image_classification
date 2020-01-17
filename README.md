## Requirements
- Pytorch 1.0
- Python 3.6+
- DALI
```bash
# Train A network with ImageNet64 memory data by dali
./script/data_to_memory.sh imagenet64
python train_dali.py --dataset ImageNet64 --data_path /userhome/temp_data/cifar10 --model_method proxyless_NAS --model_name proxyless_gpu

cd /userhome/project/pytorch_image_classification; ./script/data_to_memory.sh cifar10;python train_dali.py --dataset cifar10 --data_path /userhome/temp_data/cifar10 --model_method manual --model_name MobileNetV2



```
