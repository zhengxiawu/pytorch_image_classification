""" Config class for search/augment """
import argparse
import os
import models.darts.genotypes as gt
from functools import partial
import torch
import time


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class AugmentConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Augment config")
        parser.add_argument('--name', default='')
        parser.add_argument('--dataset', default='CIFAR10', help='CIFAR10 / ImageNet64 / FashionMNIST')
        parser.add_argument('--data_path', required=False, default='/userhome/data/cifar10',
                            help='data path')
        parser.add_argument('--model_method', default='proxyless_NAS',)
        parser.add_argument('--model_name', default='proxyless_gpu', )
        parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--lr', type=float, default=0.1, help='lr for weights')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
        parser.add_argument('--grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=300, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=36)
        parser.add_argument('--layers', type=int, default=20, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--aux_weight', type=float, default=0, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        # parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')
        parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path prob')

        parser.add_argument('--genotype', default='', help='Cell genotype')
        parser.add_argument('--deterministic', type=bool, default=True, help='momentum')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))
        time_str = time.asctime(time.localtime()).replace(' ', '_')
        self.path = os.path.join('/userhome/project/pytorch_image_classification/expreiments', self.model_method + '_'
                                 + self.model_name + '_' + self.dataset + '_' + time_str)
        if len(self.genotype) > 1:
            self.genotype = gt.from_str(self.genotype)
        else:
            self.genotype = None
        self.gpus = parse_gpus(self.gpus)
