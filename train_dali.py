""" Training augmented model """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import AugmentConfig
import utils
from models.darts.augment_cnn import AugmentCNN
from models import get_model
from data import get_data
import flops_counter


config = AugmentConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "logger.log"))
config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")

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
    input_size, input_channels, n_classes, train_data, valid_data = get_data.get_data_dali(
        config.dataset, config.data_path, batch_size=config.batch_size, num_threads=config.workers)

    criterion = nn.CrossEntropyLoss().to(device)

    use_aux = config.aux_weight > 0.
    if config.model_method == 'darts_NAS':
        if config.genotype is None:
            config.genotype = get_model.get_model(config.model_method, config.model_name)
        model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                           use_aux, config.genotype)
    else:
        model_fun = get_model.get_model(config.model_method, config.model_name)
        model = model_fun(num_classes=n_classes, dropout_rate=config.dropout_rate)
    # model size
    total_ops, total_params = flops_counter.profile(model, [1, input_channels, input_size, input_size])
    logger.info("Model size = {:.3f} MB".format(total_params))
    logger.info("Model FLOPS = {:.3f} M".format(total_ops))
    model = nn.DataParallel(model, device_ids=config.gpus).to(device)
    # weights optimizer
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    best_top1 = 0.
    # training loop
    for epoch in range(config.epochs):
        lr_scheduler.step()
        if config.drop_path_prob > 0:
            drop_prob = config.drop_path_prob * epoch / config.epochs
            model.module.drop_path_prob(drop_prob)

        # training
        train(train_data, model, optimizer, criterion, epoch)

        # validation
        top1 = validate(valid_data, model, criterion, epoch)

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)

        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))


def train(train_loader, model, optimizer, criterion, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    writer.add_scalar('train/lr', cur_lr, epoch)

    model.train()

    for step, data in enumerate(train_loader):
        X = data[0]["data"].cuda(non_blocking=True)
        y = data[0]["label"].squeeze().long().cuda(non_blocking=True)
        N = X.size(0)
        optimizer.zero_grad()
        if config.aux_weight > 0.:
            logits, aux_logits = model(X)
            loss = criterion(logits, y)
            loss += config.aux_weight * criterion(aux_logits, y)
        else:
            logits = model(X)
            loss = criterion(logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), epoch)
        writer.add_scalar('train/top1', prec1.item(), epoch)
        writer.add_scalar('train/top5', prec5.item(), epoch)
    train_loader.reset()
    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, criterion, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, data in enumerate(valid_loader):
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

            if step % config.print_freq == 0:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, losses=losses,
                        top1=top1, top5=top5))
    valid_loader.reset()
    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/top1', top1.avg, epoch)
    writer.add_scalar('val/top5', top5.avg, epoch)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":

    main()
