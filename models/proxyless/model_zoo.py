from functools import partial
import json

import torch

from models.proxyless.utils import download_url
from models.proxyless.nas_modules import ProxylessNASNets
import os
project_path = os.getcwd()


def proxyless_base(num_classes=10, pretrained=False, net_config=None, net_weight=None,
                   dropout_rate=0):
    assert net_config is not None, "Please input a network config"
    net_config_json = json.load(open(net_config, 'r'))
    net_config_json['classifier']['out_features'] = num_classes
    net_config_json['classifier']['dropout_rate'] = dropout_rate
    net = ProxylessNASNets.build_from_config(net_config_json)

    if 'bn' in net_config_json:
        net.set_bn_param(
            bn_momentum=net_config_json['bn']['momentum'],
            bn_eps=net_config_json['bn']['eps'])
    else:
        net.set_bn_param(bn_momentum=0.1, bn_eps=1e-3)

    if pretrained:
        assert net_weight is not None, "Please specify network weights"
        init_path = download_url(net_weight)
        init = torch.load(init_path, map_location='cpu')
        net.load_state_dict(init['state_dict'])

    return net


proxyless_cpu = partial(
    proxyless_base,
    net_config=os.path.join(project_path,
                            "models/proxyless/network_config/proxyless_cpu.config"),
    net_weight="https://hanlab.mit.edu/files/proxylessNAS/proxyless_cpu.pth")

proxyless_gpu = partial(
    proxyless_base,
    net_config=os.path.join(project_path,
                            "models/proxyless/network_config/proxyless_gpu.config"),
    net_weight="https://hanlab.mit.edu/files/proxylessNAS/proxyless_gpu.pth")

proxyless_mobile = partial(
    proxyless_base,
    net_config=os.path.join(project_path,
                            "models/proxyless/network_config/proxyless_mobile.config"),
    net_weight="https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile.pth")

proxyless_mobile_14 = partial(
    proxyless_base,
    net_config=os.path.join(project_path,
                            "models/proxyless/network_config/proxyless_mobile_14.config"),
    net_weight="https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile_14.pth")

# if __name__ == '__main__':
#     print(project_path)
#
#     print('hi')
#     pass
