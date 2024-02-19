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
import torch
from WORKFLOW.CLS.TableCls.v0.utils.make_dataloader import get_dataloader
from MODELALG.CLS.ClsCollect.ClsCollectv0.utils import get_network
from MODELALG.CLS.ClsCollect.ClsCollectv0.torch_utils import select_device


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, help="net type", default="shufflenetv2")
    parser.add_argument("--size", type=int, default=512, help="img size for dataloader")
    # parser.add_argument('--weights', type=str, help='the weights file you want to test',
    #                     default=root_dir + '/MODEL/CLS/ClsCollect/ClsCollectv0/TableCls/20210629/best.pt')
    parser.add_argument(
        "--weights",
        type=str,
        help="the weights file you want to test",
        default=cur_dir
        + "/checkpoint/shufflenetv2/Wednesday_30_June_2021_18h_22m_07s/shufflenetv2-4-regular.pth",
    )
    parser.add_argument("--gpu", type=str, default="0", help="use gpu or not")
    parser.add_argument("--b", type=int, default=64, help="batch size for dataloader")
    args = parser.parse_args()

    net = get_network(args).to(select_device(args.gpu))
    net = torch.nn.DataParallel(net)

    angle_test_loader = get_dataloader(
        root_dir + "/DATASETS/CLS/TableCls/v0/test/", True, 8, args.b, args.size
    )
    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(angle_test_loader):
            # print('*******************', np.shape(image), label, type(label))
            # print('-------------------', np.array(image[0]))
            print(
                "iteration: {}\ttotal {} iterations".format(
                    n_iter + 1, len(angle_test_loader)
                )
            )

            if args.gpu:
                image = image.cuda()
                label = label.cuda()

            output = net(image)
            _, pred = output.topk(1, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            correct_1 += correct[:, :1].sum()

    print()
    print("Top 1 err: ", 1 - correct_1 / len(angle_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
