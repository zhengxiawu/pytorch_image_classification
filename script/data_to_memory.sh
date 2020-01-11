#!/usr/bin/env bash
mount -t tmpfs -o size=30G tmpfs /userhome/temp_data
if [ "$1" == "imagenet64" ]; then
cp -r /userhome/data/ImageNet64 /userhome/temp_data/
elif [ "$1" == 'imagenet32' ];
then
cp -r /userhome/data/ImageNet32 /userhome/temp_data/
elif [ "$1" == 'imagenet16' ];
then
cp -r /userhome/data/ImageNet16 /userhome/temp_data/
elif [ "$1" == 'cifar10' ];
then
cp -r /userhome/data/cifar10 /userhome/temp_data/
fi