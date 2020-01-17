from .proxyless_nets import ProxylessNASNets
from .mobilenet_v3 import MobileNetV3
import json, os
from functools import partial

project_path = os.getcwd()


def ofa_specialized(num_classes=10, net_config=None, dropout_rate=0):
    assert net_config is not None, "Please input a network config"
    net_config_json = json.load(open(net_config, 'r'))
    net_config_json['classifier']['out_features'] = num_classes
    net_config_json['classifier']['dropout_rate'] = dropout_rate
    if net_config['name'] == ProxylessNASNets.__name__:
        net = ProxylessNASNets.build_from_config(net_config)
    elif net_config['name'] == MobileNetV3.__name__:
        net = MobileNetV3.build_from_config(net_config)
    else:
        raise ValueError('Not supported network type: %s' % net_config['name'])
    return net


OFA_595 = partial(
    ofa_specialized,
    net_config=os.path.join(project_path,
                            "models/ofa/network_config/FLOPS595.config"))

OFA_482 = partial(
    ofa_specialized,
    net_config=os.path.join(project_path,
                            "models/ofa/network_config/FLOPS482.config"))

OFA_398 = partial(
    ofa_specialized,
    net_config=os.path.join(project_path,
                            "models/ofa/network_config/FLOPS398.config"))


