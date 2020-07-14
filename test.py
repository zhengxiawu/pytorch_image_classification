""" Evaluate high accuracy model """
import os
import torch
import torch.nn as nn
import numpy as np
from config import BaseConfig, get_parser, parse_gpus
import models.darts.genotypes as gt
import time
import utils
from models.darts.augment_cnn import AugmentCNN, AugmentCNN_ImageNet
from models import get_model
from data import get_data
import flops_counter

project_path = "/userhome/project/pytorch_image_classification"


class EvaluateConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Augment config")
        parser.add_argument('--name', default='')
        parser.add_argument('--dataset', default='ImageNet',
                            help='imagenet / ImageNet56 / ImageNet112 / cifar10')
        parser.add_argument('--data_path', default='/gdata/ImageNet2012',
                            help='data path')
        parser.add_argument('--data_loader_type',
                            default='torch', help='torch/dali')
        parser.add_argument('--grad_clip', type=float,
                            default=0, help='gradient clipping for weights')
        parser.add_argument('--model_method', default='my_model_collection',)
        parser.add_argument('--model_name', default='my_model_collection', )
        parser.add_argument('--model_init', type=str,
                            default='he_fout', choices=['he_fin', 'he_fout'])

        parser.add_argument('--batch_size', type=int,
                            default=256, help='batch size')
        # parser.add_argument('--lr', type=float, default=0.05, help='lr for weights')
        # parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        # parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
        parser.add_argument('--label_smoothing', type=float, default=0.1)
        parser.add_argument('--no_decay_keys', type=str,
                            default='bn', choices=['None', 'bn', 'bn#bias'])

        parser.add_argument('--print_freq', type=int,
                            default=1, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        # parser.add_argument('--epochs', type=int, default=150, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=36)
        parser.add_argument('--layers', type=int,
                            default=20, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')

        parser.add_argument('--workers', type=int,
                            default=24, help='# of workers')
        parser.add_argument('--aux_weight', type=float,
                            default=0, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int,
                            default=0, help='cutout length')
        parser.add_argument('--auto_augmentation',  action='store_true',
                            default=False, help='using autoaugmentation')

        parser.add_argument('--bn_momentum', type=float, default=0.1)
        parser.add_argument('--bn_eps', type=float, default=1e-3)
        parser.add_argument('--sync_bn', action='store_true',
                            default=False, help='using sync_bn model')
        parser.add_argument('--dropout_rate', type=float, default=0)
        # parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')
        parser.add_argument('--drop_path_prob', type=float,
                            default=0, help='drop path prob')

        parser.add_argument('--genotype', default='', help='Cell genotype')
        parser.add_argument('--structure_path', default=None,
                            type=str, help='Config path')
        parser.add_argument('--deterministic', action='store_true',
                            default=False, help='using deterministic model')

        parser.add_argument('--pretrained', type=str,
                            default=False, help='load pretrained module')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))
        if self.data_loader_type == 'dali':
            if self.auto_augmentation or self.cutout_length > 0:
                print("DALI do not support Augmentation and Cutout!")
                exit()

        time_str = time.asctime(time.localtime()).replace(' ', '_')
        name_componment = [self.data_loader_type]
        if not self.model_method == 'darts_NAS':
            if self.aux_weight > 0 or self.drop_path_prob > 0:
                print("aux head and drop path only support for daats search space!")
                exit()
        else:
            name_componment += ['channels_' + str(self.init_channels), 'layers_' + str(self.layers),
                                'aux_weight_' + str(self.aux_weight), 'drop_path_prob_' + str(self.drop_path_prob)]

        # if self.dropout_rate > 0:
        #     name_componment.append('dropout_'+str(self.dropout_rate))
        # if self.auto_augmentation:
        #     name_componment.append('auto_augmentation_')
        # if self.label_smoothing > 0:
        #     name_componment.append('label_smoothing_' + str(self.label_smoothing))
#
        # if not self.no_decay_keys == 'None':
        #     name_componment.append('no_decay_keys_' + str(self.no_decay_keys))
        name_str = ''
        for i in name_componment:
            name_str += i + '_'
        name_str += time_str
        self.path = os.path.join(project_path, 'experiments', self.model_method,
                                 self.model_name, self.dataset, name_str)
        if len(self.genotype) > 1:
            self.genotype = gt.from_str(self.genotype)
        else:
            self.genotype = None
        self.gpus = parse_gpus(self.gpus)


def get_iterator_length(data_loader):
    _size = len(data_loader) if isinstance(data_loader, torch.utils.data.DataLoader) \
        else int(data_loader._size / data_loader.batch_size + 1)
    return _size


config = EvaluateConfig()

device = torch.device("cuda")


def main():
    print("evaluate start")

    # set default gpu device id
    # torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if config.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.benchmark = True

    # get data with meta info
    if config.data_loader_type == 'torch':
        input_size, input_channels, n_classes, train_data, valid_data = get_data.get_data(
            config.dataset, config.data_path, config.cutout_length,
            auto_augmentation=config.auto_augmentation)
        # train_loader = torch.utils.data.DataLoader(train_data,
        #                                            batch_size=config.batch_size,
        #                                            shuffle=True,
        #                                            num_workers=config.workers,
        #                                            pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_data,
                                                   batch_size=config.batch_size,
                                                   shuffle=False,
                                                   num_workers=config.workers,
                                                   pin_memory=False)
    elif config.data_loader_type == 'dali':
        input_size, input_channels, n_classes, train_data, valid_data = get_data.get_data_dali(
            config.dataset, config.data_path, batch_size=config.batch_size, num_threads=config.workers)
        # train_loader = train_data
        valid_loader = valid_data
    else:
        raise NotImplementedError

    use_aux = config.aux_weight > 0.
    if config.model_method == 'darts_NAS':
        if config.genotype is None:
            config.genotype = get_model.get_model(
                config.model_method, config.model_name)
        if 'imagenet' in config.dataset.lower():
            model = AugmentCNN_ImageNet(input_size, input_channels, config.init_channels, n_classes, config.layers,
                                        use_aux, config.genotype)
        else:
            model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                               use_aux, config.genotype)
    elif config.model_method == 'my_model_collection':
        from models.my_searched_model import my_specialized
        if config.structure_path is None:
            _ = config.model_name.split(':')
            net_config_path = os.path.join(project_path, 'models', 'my_model_collection',
                                           _[0], _[1] + '.json')
        else:
            net_config_path = config.structure_path
        # model = my_specialized(num_classes=n_classes, net_config=net_config_path,
        #                        dropout_rate=config.dropout_rate)
        model = my_specialized(num_classes=n_classes, net_config=net_config_path,
                               dropout_rate=0)
    else:
        model_fun = get_model.get_model(config.model_method, config.model_name)
        # model = model_fun(num_classes=n_classes, dropout_rate=config.dropout_rate)
        model = model_fun(num_classes=n_classes, dropout_rate=0)
    # load model
    ckpt = torch.load(config.pretrained)
    print(ckpt.keys())
    # for k in model:
    #     print(k)
    # return
    # set bn
    # model.set_bn_param(config.bn_momentum, config.bn_eps)
    for _key in list(ckpt['state_dict_ema'].keys()):
        if 'total_ops' in _key or 'total_params' in _key:
            del ckpt['state_dict_ema'][_key]
    model.load_state_dict(ckpt['state_dict_ema'])
    # model init
    # model.init_model(model_init=config.model_init)
    model.cuda()
    # model size
    total_ops, total_params = flops_counter.profile(
        model, [1, input_channels, input_size, input_size])
    print("Model size = {:.3f} MB".format(total_params))
    print("Model FLOPS with input {} = {:.3f} M".format(str([1, input_channels, input_size, input_size]),
                                                        total_ops))
    total_ops, total_params = flops_counter.profile(model, [1, 3, 224, 224])
    print(
        "Model FLOPS with input [1,3,224,224] {:.3f} M".format(total_ops))

    model = nn.DataParallel(model).to(device)
    # CRITERION
    if config.label_smoothing > 0:
        from utils import LabelSmoothLoss
        criterion = LabelSmoothLoss(
            smoothing=config.label_smoothing).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    best_top1 = validate(valid_loader, model, criterion, 0, 0)

    print("Final best Prec@1 = {:.4%}".format(best_top1))


@torch.no_grad()
def validate(valid_loader, model, criterion, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()
    _size = get_iterator_length(valid_loader)

    with torch.no_grad():
        for step, data in enumerate(valid_loader):
            if isinstance(valid_loader, torch.utils.data.DataLoader):
                X, y = data[0].cuda(non_blocking=True), data[1].to(
                    device, non_blocking=True)
            else:
                X = data[0]["data"].cuda(non_blocking=True)
                y = data[0]["label"].squeeze().long().cuda(non_blocking=True)
            N = X.size(0)

            if config.aux_weight > 0.:
                logits, _ = model(X)
            else:
                logits = model(X)
            loss = criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == _size-1:
                print(
                    "Valid: Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        step, _size-1, losses=losses,
                        top1=top1, top5=top5))

    # writer.add_scalar('val/loss', losses.avg, cur_step)
    # writer.add_scalar('val/top1', top1.avg, cur_step)
    # writer.add_scalar('val/top5', top5.avg, cur_step)
    if not isinstance(valid_loader, torch.utils.data.DataLoader):
        valid_loader.reset()

    print("Valid: Final Prec@1 {:.4%}".format(top1.avg))

    return top1.avg


if __name__ == '__main__':
    main()
