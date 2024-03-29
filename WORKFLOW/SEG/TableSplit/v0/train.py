import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(
    os.path.abspath(os.path.join(root_dir, "MODELALG/SEG/SPLERGESPLIT/SPLERGESPLITv0/"))
)

import argparse
import numpy as np
import torch
from termcolor import cprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from MODELALG.SEG.SPLERGESPLIT.SPLERGESPLITv0.libs.transforms import get_transform
from MODELALG.SEG.SPLERGESPLIT.SPLERGESPLITv0.libs.dataloader import SplitTableDataset
from MODELALG.SEG.SPLERGESPLIT.SPLERGESPLITv0.libs.model import SplitModel
from MODELALG.SEG.SPLERGESPLIT.SPLERGESPLITv0.libs.losses import split_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-img",
        "--images_dir",
        default=root_dir + "/DATASETS/SEG/TableSplit/v0/train/labels/table_images/",
        dest="train_images_dir",
        help="Path to training table images (generated by prepare_data.py).",
    )
    parser.add_argument(
        "-l",
        "--labels_dir",
        default=root_dir
        + "/DATASETS/SEG/TableSplit/v0/train/labels/table_split_labels/",
        dest="train_labels_dir",
        help="Path to labels for split model (generated by prepare_data.py).",
    )
    parser.add_argument(
        "-o",
        "--output_weight_path",
        default=root_dir
        + "/MODEL/SEG/SPLERGESPLIT/SPLERGESPLITv0/TableSplit/20220715/",
        dest="output_weight_path",
        help="Output folder path for model checkpoints and summary.",
    )
    parser.add_argument(
        "-e",
        "--num_epochs",
        type=int,
        dest="num_epochs",
        help="Number of epochs.",
        default=1000,
    )
    parser.add_argument(
        "-s",
        "--save_every",
        type=int,
        dest="save_every",
        help="Save checkpoints after given epochs",
        default=500,
    )
    parser.add_argument(
        "--log_every",
        type=int,
        dest="log_every",
        help="Print logs after every given steps",
        default=10,
    )
    # parser.add_argument(
    #     "--val_every",
    #     type=int,
    #     dest="val_every",
    #     help="perform validation after given steps",
    #     default=250,
    # )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        dest="learning_rate",
        help="learning rate",
        default=0.00075,
    )
    parser.add_argument(
        "--dr",
        "--decay_rate",
        type=float,
        dest="decay_rate",
        help="weight decay rate",
        default=0.95,
    )
    # parser.add_argument(
    #     "--vs",
    #     "--validation_split",
    #     type=float,
    #     dest="validation_split",
    #     help="validation split in data",
    #     default=0.1,
    # )

    configs = parser.parse_args()

    print("Train Images Directory:", configs.train_images_dir)
    print("Train Labels Directory:", configs.train_labels_dir)
    # print("Validation Split:", configs.validation_split)
    print("Output Weights Path:", configs.output_weight_path)
    print("Number of Epochs:", configs.num_epochs)
    print("Save Checkpoint Frequency:", configs.save_every)
    print("Display logs after steps:", configs.log_every)
    # print("Perform validation after steps:", configs.val_every)
    print("Batch Size:", 1)
    print("Learning Rate:", configs.learning_rate)
    print("Decay Rate:", configs.decay_rate)

    batch_size = 1
    learning_rate = configs.learning_rate

    MODEL_STORE_PATH = configs.output_weight_path

    train_images_path = configs.train_images_dir
    train_labels_path = configs.train_labels_dir

    cprint("Loading dataset...", "blue", attrs=["bold"])
    dataset = SplitTableDataset(
        os.getcwd(),
        train_images_path,
        train_labels_path,
        transforms=get_transform(train=True),
        fix_resize=False,
    )

    # split the dataset in train and test set
    # torch.manual_seed(1)
    # indices = torch.randperm(len(dataset)).tolist()

    # test_split = int(configs.validation_split * len(indices))

    # train_dataset = torch.utils.data.Subset(dataset, indices[test_split:])
    # val_dataset = torch.utils.data.Subset(dataset, indices[:test_split])

    # define training and validation data loaders
    train_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    # val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cprint("Creating split model...", "blue", attrs=["bold"])
    model = SplitModel().to(device)

    criterion = split_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=configs.decay_rate
    )

    num_epochs = configs.num_epochs

    # create the summary writer
    writer = SummaryWriter(os.path.join(MODEL_STORE_PATH, "summary"))

    # Train the model
    total_step = len(train_loader)

    print(27 * "=", "Training", 27 * "=")

    step = 0
    for epoch in range(num_epochs):
        for i, (images, targets, img_path, _, _) in enumerate(train_loader):
            images = images.to(device)

            model.train()
            # incrementing step
            step -= -1

            targets[0] = targets[0].long().to(device)
            targets[1] = targets[1].long().to(device)

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()

            # Run the forward pass
            outputs = model(images.to(device))
            loss, rpn_loss, cpn_loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            if (i + 1) % configs.log_every == 0:
                # writing loss to tensorboard
                writer.add_scalar(
                    "total loss train", loss.item(), (epoch * total_step + i)
                )
                writer.add_scalar(
                    "rpn loss train", rpn_loss.item(), (epoch * total_step + i)
                )
                writer.add_scalar(
                    "cpn loss train", cpn_loss.item(), (epoch * total_step + i)
                )
                cprint("Iteration: ", "green", attrs=["bold"], end="")
                print(step)
                cprint("Learning Rate: ", "green", attrs=["bold"], end="")
                print(lr_scheduler.get_lr()[0])
                print(
                    "Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, RPN Loss: {:.4f}, CPN Loss: {:.4f}".format(
                        epoch + 1,
                        num_epochs,
                        i + 1,
                        total_step,
                        loss.item(),
                        rpn_loss.item(),
                        cpn_loss.item(),
                    )
                )
                print("---")

            # if (i + 1) % configs.val_every == 0:
            #     print(26 * "~", "Validation", 26 * "~")
            #     model.eval()
            #     with torch.no_grad():
            #         val_loss_list = list()
            #         val_rpn_loss_list = list()
            #         val_cpn_loss_list = list()
            #
            #         for x, (val_images, val_targets, _, _, _) in enumerate(val_loader):
            #             val_targets[0] = val_targets[0].long().to(device)
            #             val_targets[1] = val_targets[1].long().to(device)
            #
            #             val_outputs = model(val_images.to(device))
            #             val_loss, val_rpn_loss, val_cpn_loss = criterion(
            #                 val_outputs, val_targets
            #             )
            #
            #             val_loss_list.append(val_loss.item())
            #             val_rpn_loss_list.append(val_rpn_loss.item())
            #             val_cpn_loss_list.append(val_cpn_loss.item())
            #
            #             print(
            #                 "Step [{}/{}], Val Loss: {:.4f}, RPN Loss: {:.4f}, CPN Loss: {:.4f}".format(
            #                     x + 1,
            #                     len(val_loader),
            #                     val_loss,
            #                     val_rpn_loss,
            #                     val_cpn_loss,
            #                 )
            #             )
            #
            #         avg_val_loss = np.mean(np.array(val_loss_list))
            #         avg_rpn_val_loss = np.mean(np.array(val_rpn_loss_list))
            #         avg_cpn_val_loss = np.mean(np.array(val_cpn_loss_list))
            #
            #         writer.add_scalar(
            #             "total loss val", avg_val_loss, (epoch * total_step + i)
            #         )
            #         writer.add_scalar(
            #             "rpn loss val", avg_rpn_val_loss, (epoch * total_step + i)
            #         )
            #         writer.add_scalar(
            #             "cpn loss val", avg_cpn_val_loss, (epoch * total_step + i)
            #         )
            #
            #     print(64 * "~")

            if (step + 1) % configs.save_every == 0:
                print("Saving model weights at iteration", step + 1)
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "iteration": step + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                        "rpn_loss": rpn_loss,
                        "cpn_loss": cpn_loss,
                    },
                    os.path.join(MODEL_STORE_PATH, "split_oriimg_model.pth"),
                )
            torch.cuda.empty_cache()
        lr_scheduler.step()
