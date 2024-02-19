# -*- coding: utf-8 -*-
# @Time    : 2021/6/7 11:59
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(
    os.path.abspath(os.path.join(root_dir, "MODELALG/CLS/ClsCollect/ClsCollectv0/"))
)

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from WORKFLOW.CLS.TableCls.v0.config import settings
from WORKFLOW.CLS.TableCls.v0.utils.make_dataloader import get_dataloader
from MODELALG.CLS.ClsCollect.ClsCollectv0.utils import (
    get_network,
    WarmUpLR,
    most_recent_folder,
    most_recent_weights,
    last_epoch,
    best_acc_weights,
)
from MODELALG.CLS.ClsCollect.ClsCollectv0.torch_utils import select_device


def train(epoch):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(angle_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(angle_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if "weight" in name:
                writer.add_scalar(
                    "LastLayerGradients/grad_norm2_weights", para.grad.norm(), n_iter
                )
            if "bias" in name:
                writer.add_scalar(
                    "LastLayerGradients/grad_norm2_bias", para.grad.norm(), n_iter
                )

        print(
            "Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}".format(
                loss.item(),
                optimizer.param_groups[0]["lr"],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(angle_training_loader.dataset),
            )
        )

        # update training loss for each iteration
        writer.add_scalar("Train/loss", loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print("epoch {} training time consumed: {:.2f}s".format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for images, labels in angle_val_loader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print("GPU INFO.....")
        print(torch.cuda.memory_summary(), end="")
    print("Evaluating Network.....")
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s".format(
            test_loss / len(angle_val_loader.dataset),
            correct.float() / len(angle_val_loader.dataset),
            finish - start,
        )
    )
    print()

    # add informations to tensorboard
    if tb:
        writer.add_scalar(
            "Test/Average loss", test_loss / len(angle_val_loader.dataset), epoch
        )
        writer.add_scalar(
            "Test/Accuracy", correct.float() / len(angle_val_loader.dataset), epoch
        )

    return correct.float() / len(angle_val_loader.dataset)


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=3 python train_ocr_angle.py --net mobilenetv2 --gpu
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, help="net type", default="shufflenetv2")
    parser.add_argument("--size", type=int, default=512, help="img size for dataloader")
    parser.add_argument("--gpu", type=str, default="1,2,3", help="use gpu or not")
    parser.add_argument("--b", type=int, default=18, help="batch size for dataloader")
    parser.add_argument("--warm", type=int, default=1, help="warm up training phase")
    parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument("--resume", default=False, help="resume training")
    args = parser.parse_args()

    net = get_network(args).to(select_device(args.gpu))
    net = torch.nn.DataParallel(net)

    # data preprocessing:
    angle_training_loader = get_dataloader(
        root_dir + "/DATASETS/CLS/TableCls/v0/train/", True, 8, args.b, args.size
    )
    angle_val_loader = get_dataloader(
        root_dir + "/DATASETS/CLS/TableCls/v0/val/", True, 8, args.b, args.size
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=settings.MILESTONES, gamma=0.2
    )  # learning rate decay
    iter_per_epoch = len(angle_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(
            os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT
        )
        if not recent_folder:
            raise Exception("no recent folder were found")

        checkpoint_path = os.path.join(
            settings.CHECKPOINT_PATH, args.net, recent_folder
        )

    else:
        checkpoint_path = os.path.join(
            settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW
        )

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    # since tensorboard can't overwrite old values
    # so the only way is to create a new tensorboard log
    writer = SummaryWriter(
        log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW)
    )
    # input_tensor = torch.Tensor(1, 3, 256, 256).cuda()
    # writer.add_graph(net, input_tensor)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, "{net}-{epoch}-{type}.pth")

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(
            os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
        )
        if best_weights:
            weights_path = os.path.join(
                settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights
            )
            print("found best acc weights file:{}".format(weights_path))
            print("load best training file to test acc...")
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print("best acc is {:0.2f}".format(best_acc))

        recent_weights_file = most_recent_weights(
            os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
        )
        if not recent_weights_file:
            raise Exception("no recent weights file were found")
        weights_path = os.path.join(
            settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file
        )
        print("loading weights file {} to resume training.....".format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(
            os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
        )

    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        # start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(
                net.state_dict(),
                checkpoint_path.format(net=args.net, epoch=epoch, type="best"),
            )
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(
                net.state_dict(),
                checkpoint_path.format(net=args.net, epoch=epoch, type="regular"),
            )

    writer.close()
