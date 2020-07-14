from models.manual import resnet
from collections import namedtuple
from models.proxyless.model_zoo import proxyless_cpu, proxyless_gpu, proxyless_mobile, proxyless_mobile_14
from models.ofa.model_zoo import OFA_595, OFA_482, OFA_398
from models.ofa.proxyless_nets import MobileNetV2
from models.ofa.mobilenet_v3 import MobileNetV3Large
from models.my_searched_model import MY_600, MY_500, MY_400
import pdb
from models.darts.genotypes import from_str
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

DDPNAS_3 = Genotype(
    normal=[
        [('sep_conv_5x5', 0), ('max_pool_3x3', 1)],
        [('skip_connect', 0), ('avg_pool_3x3', 2)],
        [('dil_conv_3x3', 1), ('sep_conv_5x5', 0)],
        [('max_pool_3x3', 0), ('dil_conv_5x5', 4)]],
    normal_concat=range(2, 6),
    reduce=[
        [('dil_conv_3x3', 0), ('max_pool_3x3', 1)],
        [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
        [('avg_pool_3x3', 0), ('avg_pool_3x3', 2)],
        [('max_pool_3x3', 1), ('max_pool_3x3', 3)]],
    reduce_concat=range(2, 6))

DDPNAS_V3_constraint_4 = Genotype(normal=[[('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], [('sep_conv_3x3', 0), ('skip_connect', 1)], [('sep_conv_3x3', 0), ('dil_conv_3x3', 3)], [('sep_conv_5x5', 0), ('avg_pool_3x3', 4)]], normal_concat=range(2, 6), reduce=[[('sep_conv_5x5', 1), ('skip_connect', 0)], [('sep_conv_5x5', 1), ('dil_conv_3x3', 0)], [('sep_conv_5x5', 1), ('max_pool_3x3', 0)], [('sep_conv_5x5', 1), ('avg_pool_3x3', 3)]], reduce_concat=range(2, 6))
# dynamic_SNG_V3 = Genotype(normal=[[('max_pool_3x3', 1), ('avg_pool_3x3', 0)], [('max_pool_3x3', 1), ('sep_conv_5x5', 2)], [('skip_connect', 2), ('max_pool_3x3', 3)], [('max_pool_3x3', 3), ('sep_conv_3x3', 0)]], normal_concat=range(2, 6), reduce=[[('sep_conv_5x5', 1), ('max_pool_3x3', 0)], [('max_pool_3x3', 1), ('max_pool_3x3', 2)], [('sep_conv_5x5', 1), ('sep_conv_5x5', 2)], [('sep_conv_3x3', 0), ('avg_pool_3x3', 1)]], reduce_concat=range(2, 6))
# BPE models
dynamic_SNG_V3 = Genotype(normal=[[('sep_conv_5x5', 0), ('sep_conv_5x5', 1)], [('max_pool_3x3', 0), ('skip_connect', 1)], [('sep_conv_3x3', 2), ('sep_conv_3x3', 1)], [('skip_connect', 4), ('sep_conv_5x5', 3)]], normal_concat=range(2, 6), reduce=[[('sep_conv_5x5', 1), ('dil_conv_3x3', 0)], [('skip_connect', 2), ('avg_pool_3x3', 0)], [('avg_pool_3x3', 1), ('dil_conv_5x5', 2)], [('sep_conv_3x3', 3), ('dil_conv_5x5', 2)]], reduce_concat=range(2, 6))

BPE_models = {
    'EA_BPE1': "Genotype(normal=[[('avg_pool_3x3', 0), ('skip_connect', 1)], [('skip_connect', 0), ('sep_conv_3x3', 1)], [('sep_conv_5x5', 0), ('dil_conv_5x5', 1)], [('sep_conv_3x3', 0), ('sep_conv_5x5', 2)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], [('sep_conv_3x3', 0), ('skip_connect', 2)], [('max_pool_3x3', 1), ('skip_connect', 2)]], reduce_concat=range(2, 6))",

    'EA_BPE2': "Genotype(normal=[[('skip_connect', 0), ('avg_pool_3x3', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)], [('dil_conv_3x3', 1), ('avg_pool_3x3', 2)], [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)]], normal_concat=range(2, 6), reduce=[[('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], [('dil_conv_5x5', 0), ('sep_conv_5x5', 2)], [('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], [('skip_connect', 1), ('sep_conv_3x3', 4)]], reduce_concat=range(2, 6))",

    'RL_BPE1': "Genotype(normal=[[('skip_connect', 0), ('dil_conv_3x3', 1)], [('sep_conv_3x3', 1), ('avg_pool_3x3', 2)], [('sep_conv_3x3', 0), ('max_pool_3x3', 3)], [('sep_conv_5x5', 0), ('avg_pool_3x3', 3)]], normal_concat=range(2, 6), reduce=[[('dil_conv_3x3', 0), ('sep_conv_5x5', 1)], [('sep_conv_5x5', 0), ('max_pool_3x3', 1)], [('dil_conv_3x3', 2), ('sep_conv_5x5', 3)], [('avg_pool_3x3', 0), ('sep_conv_3x3', 4)]], reduce_concat=range(2, 6))",

    'RL_BPE2': "Genotype(normal=[[('skip_connect', 0), ('avg_pool_3x3', 1)], [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_5x5', 0), ('skip_connect', 1)], [('avg_pool_3x3', 0), ('avg_pool_3x3', 2)]], normal_concat=range(2, 6), reduce=[[('sep_conv_5x5', 0), ('skip_connect', 1)], [('avg_pool_3x3', 1), ('sep_conv_3x3', 2)], [('dil_conv_5x5', 1), ('sep_conv_3x3', 3)], [('avg_pool_3x3', 1), ('max_pool_3x3', 4)]], reduce_concat=range(2, 6))",

    'DARTS_BPE1': "Genotype(normal=[[('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], [('sep_conv_3x3', 4), ('sep_conv_5x5', 3)]], normal_concat=range(2, 6), reduce=[[('dil_conv_3x3', 0), ('sep_conv_5x5', 1)], [('sep_conv_5x5', 1), ('sep_conv_5x5', 0)], [('sep_conv_5x5', 3), ('dil_conv_5x5', 2)], [('dil_conv_5x5', 4), ('sep_conv_5x5', 3)]], reduce_concat=range(2, 6))",

    'DARTS_BPE2': "Genotype(normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], [('sep_conv_3x3', 0), ('sep_conv_5x5', 3)], [('sep_conv_3x3', 4), ('sep_conv_3x3', 0)]], normal_concat=range(2, 6), reduce=[[('dil_conv_5x5', 1), ('sep_conv_5x5', 0)], [('avg_pool_3x3', 0), ('sep_conv_5x5', 1)], [('dil_conv_5x5', 3), ('sep_conv_5x5', 2)], [('sep_conv_3x3', 2), ('sep_conv_3x3', 1)]], reduce_concat=range(2, 6))",

    'RS_BPE1': "Genotype(normal=[[('sep_conv_5x5', 0), ('dil_conv_3x3', 1)], [('max_pool_3x3', 0), ('sep_conv_3x3', 1)], [('skip_connect', 3), ('sep_conv_5x5', 0)], [('sep_conv_3x3', 2), ('skip_connect', 0)]], normal_concat=range(2, 6), reduce=[[('dil_conv_5x5', 1), ('max_pool_3x3', 0)], [('sep_conv_3x3', 1), ('max_pool_3x3', 0)], [('dil_conv_3x3', 0), ('max_pool_3x3', 1)], [('dil_conv_3x3', 2), ('sep_conv_3x3', 0)]], reduce_concat=range(2, 6))",

    'RS_BPE2': "Genotype(normal=[[('skip_connect', 1), ('sep_conv_3x3', 0)], [('avg_pool_3x3', 2), ('skip_connect', 0)], [('sep_conv_3x3', 1), ('avg_pool_3x3', 3)], [('avg_pool_3x3', 0), ('dil_conv_3x3', 2)]], normal_concat=range(2, 6), reduce=[[('skip_connect', 1), ('dil_conv_3x3', 0)], [('max_pool_3x3', 1), ('dil_conv_3x3', 0)], [('dil_conv_5x5', 3), ('max_pool_3x3', 0)], [('skip_connect', 3), ('dil_conv_3x3', 2)]], reduce_concat=range(2, 6))"
}

DARTS_NAS_model_dict = {'MDENAS': MDENAS,
                        'DDPNAS_V1': DDPNAS_1,
                        'DDPNAS_V2': DDPNAS_2,
                        'DDPNAS_V3': DDPNAS_3,
                        'DDPNAS_V3_constraint_4': DDPNAS_V3_constraint_4,
                        'DARTS_V1': DARTS_V1,
                        'DARTS_V2': DARTS_V2,
                        'dynamic_SNG_V3': dynamic_SNG_V3,
                        'EA_BPE1': from_str(BPE_models['EA_BPE1']),
                        'EA_BPE2': from_str(BPE_models['EA_BPE2']),
                        'RL_BPE1': from_str(BPE_models['RL_BPE1']),
                        'RL_BPE2': from_str(BPE_models['RL_BPE2']),
                        'DARTS_BPE1': from_str(BPE_models['DARTS_BPE1']),
                        'DARTS_BPE2': from_str(BPE_models['DARTS_BPE2']),
                        'RS_BPE1': from_str(BPE_models['RS_BPE1']),
                        'RS_BPE2': from_str(BPE_models['RS_BPE2']),
                        }

Proxyless_NAS_model_dict = {'proxyless_gpu': proxyless_gpu,
                            'proxyless_cpu': proxyless_cpu,
                            'proxyless_mobile': proxyless_mobile,
                            'proxyless_mobile_14': proxyless_mobile_14,
                            'ofa_595': OFA_595,
                            'ofa_482': OFA_482,
                            'ofa_398': OFA_398,
                            }

My_model_dict = {'my_600': MY_600,
                 'my_500': MY_500,
                 'my_400': MY_400,
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
    elif method == 'my_model':
        return My_model_dict[name]
    else:
        raise NotImplementedError
