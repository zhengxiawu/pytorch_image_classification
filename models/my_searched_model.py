from models.ofa.proxyless_nets import ProxylessNASNets
from models.ofa.mobilenet_v3 import MobileNetV3
import json
import os
from functools import partial
import pdb

project_path = '/userhome/project/Auto_NAS_V2/experiment/dynamic_SNG_V3/ofa__epochs_200_data_split_10_warm_up_epochs_0_pruning_step_3_Wed_Jan_22_11:52:19_2020'


def my_specialized(num_classes=10, net_config=None, dropout_rate=0):
    assert net_config is not None, "Please input a network config"
    net_config_json = json.load(open(net_config, 'r'))
    net_config_json['classifier']['out_features'] = num_classes
    net_config_json['classifier']['dropout_rate'] = dropout_rate
    if net_config_json['name'] == ProxylessNASNets.__name__:
        net = ProxylessNASNets.build_from_config(net_config_json)
    elif net_config_json['name'] == MobileNetV3.__name__:
        net = MobileNetV3.build_from_config(net_config_json)
    else:
        raise ValueError('Not supported network type: %s' % net_config_json['name'])
    return net


MY_600 = partial(
    my_specialized,
    net_config=os.path.join(project_path,
                            "network_info/600.json"))

MY_500 = partial(
    my_specialized,
    net_config=os.path.join(project_path,
                            "network_info/500.json"))

MY_400 = partial(
    my_specialized,
    net_config=os.path.join(project_path,
                            "network_info/400.json"))


