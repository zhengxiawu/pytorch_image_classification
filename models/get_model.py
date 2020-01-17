from models.manual import resnet
from collections import namedtuple
from models.proxyless.model_zoo import proxyless_cpu, proxyless_gpu, proxyless_mobile, proxyless_mobile_14
from models.ofa.model_zoo import OFA_595, OFA_482, OFA_398
from models.ofa.proxyless_nets import MobileNetV2
from models.ofa.mobilenet_v3 import MobileNetV3Large
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
NASNet = Genotype(
    normal=[
        [('sep_conv_5x5', 1), ('sep_conv_3x3', 0)],
        [('sep_conv_5x5', 0), ('sep_conv_3x3', 0)],
        [('avg_pool_3x3', 1), ('skip_connect', 0)],
        [('avg_pool_3x3', 0), ('avg_pool_3x3', 0)],
        [('sep_conv_3x3', 1), ('skip_connect', 1)],
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        [('sep_conv_5x5', 1), ('sep_conv_7x7', 0)],
        [('max_pool_3x3', 1), ('sep_conv_7x7', 0)],
        [('avg_pool_3x3', 1), ('sep_conv_5x5', 0)],
        [('skip_connect', 3), ('avg_pool_3x3', 2)],
        [('sep_conv_3x3', 2), ('max_pool_3x3', 1)],
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        [('avg_pool_3x3', 0), ('max_pool_3x3', 1)],
        [('sep_conv_3x3', 0), ('sep_conv_5x5', 2)],
        [('sep_conv_3x3', 0), ('avg_pool_3x3', 3)],
        [('sep_conv_3x3', 1), ('skip_connect', 1)],
        [('skip_connect', 0), ('avg_pool_3x3', 1)],
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        [('avg_pool_3x3', 0), ('sep_conv_3x3', 1)],
        [('max_pool_3x3', 0), ('sep_conv_7x7', 2)],
        [('sep_conv_7x7', 0), ('avg_pool_3x3', 1)],
        [('max_pool_3x3', 0), ('max_pool_3x3', 1)],
        [('conv_7x1_1x7', 0), ('sep_conv_3x3', 5)],
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[
        [('sep_conv_3x3', 1), ('sep_conv_3x3', 0)],
        [('skip_connect', 0), ('sep_conv_3x3', 1)],
        [('skip_connect', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_3x3', 0), ('skip_connect', 2)],
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        [('max_pool_3x3', 0), ('max_pool_3x3', 1)],
        [('skip_connect', 2), ('max_pool_3x3', 0)],
        [('max_pool_3x3', 0), ('skip_connect', 2)],
        [('skip_connect', 2), ('avg_pool_3x3', 0)]
    ],
    reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_3x3', 1), ('skip_connect', 0)],
        [('skip_connect', 0), ('dil_conv_3x3', 2)],
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        [('max_pool_3x3', 0), ('max_pool_3x3', 1)],
        [('skip_connect', 2), ('max_pool_3x3', 1)],
        [('max_pool_3x3', 0), ('skip_connect', 2)],
        [('skip_connect', 2), ('max_pool_3x3', 1)],
    ],
    reduce_concat=[2, 3, 4, 5])

MDENAS = Genotype(
    normal=[
        [('sep_conv_5x5', 1), ('sep_conv_3x3', 0)],
        [('skip_connect', 0), ('sep_conv_5x5', 1)],
        [('sep_conv_5x5', 3), ('sep_conv_3x3', 1)],
        [('dil_conv_5x5', 3), ('max_pool_3x3', 4)],
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 0), ('sep_conv_5x5', 1)],
        [('skip_connect', 0), ('skip_connect', 1)],
        [('sep_conv_3x3', 3), ('skip_connect', 2)],
        [('dil_conv_3x3', 3), ('sep_conv_5x5', 0)],
    ],
    reduce_concat=range(2, 6))
DDPNAS_1 = Genotype(
    normal=[
        [('sep_conv_5x5', 0), ('max_pool_3x3', 1)],
        [('sep_conv_3x3', 2), ('max_pool_3x3', 0)],
        [('dil_conv_5x5', 0), ('sep_conv_3x3', 1)],
        [('avg_pool_3x3', 2), ('max_pool_3x3', 4)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_3x3', 1), ('dil_conv_5x5', 0)],
        [('dil_conv_5x5', 1), ('sep_conv_5x5', 0)],
        [('sep_conv_3x3', 0), ('avg_pool_3x3', 3)],
        [('max_pool_3x3', 1), ('sep_conv_3x3', 0)]
    ],
    reduce_concat=range(2, 6))

DDPNAS_2 = Genotype(
    normal=[
        [('sep_conv_5x5', 0), ('skip_connect', 1)],
        [('dil_conv_5x5', 2), ('max_pool_3x3', 0)],
        [('sep_conv_5x5', 1), ('skip_connect', 2)],
        [('skip_connect', 1), ('sep_conv_3x3', 3)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_3x3', 2), ('avg_pool_3x3', 0)],
        [('avg_pool_3x3', 0), ('dil_conv_3x3', 1)],
        [('sep_conv_3x3', 0), ('max_pool_3x3', 4)]
    ],
    reduce_concat=range(2, 6))

DARTS_NAS_model_dict = {'MDENAS': MDENAS,
              'DDPNAS_V1': DDPNAS_1,
              'DDPNAS_V2': DDPNAS_2,
              'DARTS_V1': DARTS_V1,
              'DARTS_V2': DARTS_V2}

Proxyless_NAS_model_dict = {'proxyless_gpu': proxyless_gpu,
                            'proxyless_cpu': proxyless_cpu,
                            'proxyless_mobile': proxyless_mobile,
                            'proxyless_mobile_14': proxyless_mobile_14,
                            'ofa_595': OFA_595,
                            'ofa_482': OFA_482,
                            'ofa_398': OFA_398,
                            }

Manual_model_dict = {'Resnet18': resnet.ResNet18,
                     'MobileNetV2': MobileNetV2,
                     'MobileNetV3Large': MobileNetV3Large
                     }


def get_model(method, name):
    if method == 'darts_NAS':
        return DARTS_NAS_model_dict[name]
    elif method == 'manual':
        return Manual_model_dict[name]
    elif method == 'proxyless_NAS':
        return Proxyless_NAS_model_dict[name]
    else:
        raise NotImplementedError
