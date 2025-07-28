import os
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from utils.losses import dice_loss
from dataloaders.la_heart import LAHeart, RandomCrop, RandomRotFlip, ToTensor
from test_util import test_all_case
from utils import ramps
import time


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="../data/2018LA_Seg_Training Set/",
    help="Name of Experiment",
)
parser.add_argument("--exp", type=str, default="sym_match", help="model_name")
parser.add_argument(
    "--max_iterations", type=int, default=9000, help="maximum epoch number to train"
)
parser.add_argument("--batch_size", type=int, default=2, help="batch_size per gpu")
parser.add_argument(
    "--base_lr", type=float, default=0.01, help="maximum epoch number to train"
)
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--label_num", type=int, default=16, help="label num")
parser.add_argument("--eta", type=float, default=0.3, help="weight to balance loss")
parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer")
parser.add_argument("--conf_thresh", type=float, default=0.75, help="conf_thresh")
parser.add_argument("--temperature", type=float, default=1, help="temperature")

parser.add_argument("--drop_rate_img", type=float, default=0.25, help="the drop rate")
parser.add_argument("--drop_rate_feat", type=float, default=0.5, help="the drop rate")
parser.add_argument(
    "--pert_type", type=str, default="alpdrop", help="the type of perturbation"
)
parser.add_argument("--loss_kd", action="store_true", help="whether use loss_kd")
parser.add_argument("--loss_mse", action="store_true", help="whether use loss_mse")
parser.add_argument(
    "--feat_pert_type",
    type=str,
    default="dropout",
    help="the type of feature perturbation",
)
parser.add_argument("--T", type=float, default=2.0, help="Temperature")
parser.add_argument(
    "--distill_loss", type=str, default="KL", help="distillation loss type"
)

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(","))
max_iterations = args.max_iterations
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
else:
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

patch_size = (112, 112, 80)
num_classes = 2

LABELED_ID_NUM = args.label_num  # 8 or 16
conf_thresh = args.conf_thresh
eta = args.eta
T = args.T

pervious_bset_dice = 0.0

DROP_TYPE_DICT = {
    "dropout": nn.Dropout3d,
    "alpdrop": nn.AlphaDropout,
    "featalpdrop": nn.FeatureAlphaDropout,
}

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + "/code"):
        shutil.rmtree(snapshot_path + "/code")

    shutil.copytree(
        ".", snapshot_path + "/code", shutil.ignore_patterns([".git", "__pycache__"])
    )

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    net = VNet(
        n_channels=1,
        n_classes=num_classes,
        normalization="batchnorm",
        has_dropout=False,
        pert_type=args.pert_type,
        drop_rate_feat=args.drop_rate_feat,
    )
    net = net.cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainset_u = LAHeart(
        base_dir=train_data_path,
        mode="train_u",
        num=80 - LABELED_ID_NUM,
        transform=transforms.Compose(
            [
                RandomRotFlip(),
                RandomCrop(patch_size),
                ToTensor(),
            ]
        ),
        id_path=f"../data/2018LA_Seg_Training Set/train_{LABELED_ID_NUM}_unlabel.list",
    )
    trainsampler_u = torch.utils.data.sampler.RandomSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size * 2,
        drop_last=True,
        sampler=trainsampler_u,
        worker_init_fn=worker_init_fn,
    )
    trainsampler_u_mix = torch.utils.data.sampler.RandomSampler(
        trainset_u, replacement=True
    )
    trainloader_u_mix = DataLoader(
        trainset_u,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size * 2,
        drop_last=True,
        sampler=trainsampler_u_mix,
        worker_init_fn=worker_init_fn,
    )

    trainset_l = LAHeart(
        base_dir=train_data_path,
        mode="train_l",
        num=80 - LABELED_ID_NUM,
        transform=transforms.Compose(
            [
                RandomRotFlip(),
                RandomCrop(patch_size),
                ToTensor(),
            ]
        ),
        id_path=f"../data/2018LA_Seg_Training Set/train_{LABELED_ID_NUM}_label.list",
    )
    trainsampler_l = torch.utils.data.sampler.RandomSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size * 2,
        drop_last=True,
        sampler=trainsampler_l,
        worker_init_fn=worker_init_fn,
    )

    net.train()
    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            net.parameters(),
            lr=base_lr,
            momentum=0.9,
            weight_decay=0.0002 if LABELED_ID_NUM == 16 else 0.0001,
        )
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0001)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.0001)
    else:
        raise NotImplementedError
    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} itertations per epoch".format(len(trainloader_l)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader_l) + 1
    print(f"All Epochs: {max_epoch}")
    lr_ = base_lr
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):

        time1 = time.time()
        net.train()

        for i_batch, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),
            (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _),
        ) in enumerate(zip(trainloader_l, trainloader_u, trainloader_u_mix)):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()

            with torch.no_grad():
                net.eval()
                pred_u_w_mix = net(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = (
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            )
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = (
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            )

            img_u_w_drop = DROP_TYPE_DICT[args.pert_type](args.drop_rate_img)(
                img_u_w.clone()
            )
            img_u_s1_drop = DROP_TYPE_DICT[args.pert_type](args.drop_rate_img)(
                img_u_s1.clone()
            )
            img_u_s2_drop = DROP_TYPE_DICT[args.pert_type](args.drop_rate_img)(
                img_u_s2.clone()
            )

            net.train()

            pred_x = net(img_x)

            preds_ws1s2, bottle_ws1s2 = net(
                torch.cat((img_u_w, img_u_s1, img_u_s2)), return_buttleneck=True
            )
            preds_ws1s2_drop, bottle_ws1s2_drop = net(
                torch.cat((img_u_w_drop, img_u_s1_drop, img_u_s2_drop)),
                return_buttleneck=True,
                feature_pert=True,
                feat_drop_func=DROP_TYPE_DICT[args.feat_pert_type],
            )

            pred_u_w, pred_u_s1, pred_u_s2 = preds_ws1s2.chunk(3)
            bottle_u_w, bottle_u_s1, bottle_u_s2 = bottle_ws1s2.chunk(3)

            pred_u_w_drop, pred_u_s1_drop, pred_u_s2_drop = preds_ws1s2_drop.chunk(3)
            bottle_u_w_drop, bottle_u_s1_drop, bottle_u_s2_drop = (
                bottle_ws1s2_drop.chunk(3)
            )

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            conf_u_s1 = pred_u_s1.softmax(dim=1).max(dim=1)[0].detach()
            mask_u_s1 = pred_u_s1.argmax(dim=1).detach()

            conf_u_s2 = pred_u_s2.softmax(dim=1).max(dim=1)[0].detach()
            mask_u_s2 = pred_u_s2.argmax(dim=1).detach()

            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]

            loss_x = (
                F.cross_entropy(pred_x, mask_x)
                + dice_loss(pred_x.softmax(dim=1)[:, 1, :, :, :], mask_x == 1)
            ) / 2.0

            loss_u_s1 = dice_loss(
                pred_u_s1.softmax(dim=1)[:, 1, :, :, :],
                mask_u_w_cutmixed1 == 1,
                ignore=(conf_u_w_cutmixed1 < conf_thresh).float(),
            )

            loss_u_s2 = dice_loss(
                pred_u_s2.softmax(dim=1)[:, 1, :, :, :],
                mask_u_w_cutmixed2 == 1,
                ignore=(conf_u_w_cutmixed2 < conf_thresh).float(),
            )

            if args.loss_kd:
                loss_u_w_drop = dice_loss(
                    pred_u_w_drop.softmax(dim=1)[:, 1, :, :, :],
                    mask_u_w == 1,
                    ignore=(conf_u_w < conf_thresh).float(),
                )

                loss_u_s1_drop = dice_loss(
                    pred_u_s1_drop.softmax(dim=1)[:, 1, :, :, :],
                    mask_u_w_cutmixed1 == 1,
                    ignore=(conf_u_w_cutmixed1 < conf_thresh).float(),
                )

                loss_u_s2_drop = dice_loss(
                    pred_u_s2_drop.softmax(dim=1)[:, 1, :, :, :],
                    mask_u_w_cutmixed2 == 1,
                    ignore=(conf_u_w_cutmixed2 < conf_thresh).float(),
                )
            else:
                loss_u_w_drop = 0
                loss_u_s1_drop = 0
                loss_u_s2_drop = 0

            if args.loss_mse:
                if args.distill_loss == "MSE":
                    loss_mse_u_w = F.mse_loss(
                        bottle_u_w.detach() / T, bottle_u_w_drop / T
                    )
                    loss_mse_u_s1 = F.mse_loss(
                        bottle_u_s1.detach() / T, bottle_u_s1_drop / T
                    )
                    loss_mse_u_s2 = F.mse_loss(
                        bottle_u_s2.detach() / T, bottle_u_s2_drop / T
                    )
                elif args.distill_loss == "KL":
                    loss_mse_u_w = F.kl_div(
                        F.log_softmax(bottle_u_w_drop / T, dim=1),
                        F.softmax(bottle_u_w.detach() / T, dim=1),
                    )
                    loss_mse_u_s1 = F.kl_div(
                        F.log_softmax(bottle_u_s1_drop / T, dim=1),
                        F.softmax(bottle_u_s1.detach() / T, dim=1),
                    )
                    loss_mse_u_s2 = F.kl_div(
                        F.log_softmax(bottle_u_s2_drop / T, dim=1),
                        F.softmax(bottle_u_s2.detach() / T, dim=1),
                    )
                elif args.distill_loss == "CE":
                    loss_mse_u_w = (
                        F.log_softmax(bottle_u_w_drop / T, dim=1)
                        * F.softmax(bottle_u_w.detach() / T, dim=1)
                    ).mean()
                    loss_mse_u_s1 = (
                        F.log_softmax(bottle_u_s1_drop / T, dim=1)
                        * F.softmax(bottle_u_s1.detach() / T, dim=1)
                    ).mean()
                    loss_mse_u_s2 = (
                        F.log_softmax(bottle_u_s2_drop / T, dim=1)
                        * F.softmax(bottle_u_s2.detach() / T, dim=1)
                    ).mean()
            else:
                loss_mse_u_w = 0
                loss_mse_u_s1 = 0
                loss_mse_u_s2 = 0

            loss_kd = (loss_u_w_drop + loss_u_s1_drop + loss_u_s2_drop) / 3.0
            loss_mse = (loss_mse_u_w + loss_mse_u_s1 + loss_mse_u_s2) / 3.0

            loss_u = loss_kd + loss_mse

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u * 0.5) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            conf_thresh = (
                args.conf_thresh
                + (1 - args.conf_thresh)
                * ramps.sigmoid_rampup(iter_num, max_iterations)
            ) * np.log(2)

            iter_num = iter_num + 1
            writer.add_scalar("lr", lr_, iter_num)
            writer.add_scalar("loss/loss_x", loss_x, iter_num)
            writer.add_scalar("loss/loss", loss, iter_num)
            writer.add_scalar("loss/loss_u_s1", loss_u_s1, iter_num)
            writer.add_scalar("loss/loss_u_s2", loss_u_s2, iter_num)
            writer.add_scalar("loss/loss_kd", loss_kd, iter_num)
            # writer.add_scalar("loss/loss_kl", loss_kl, iter_num)
            writer.add_scalar("conf/conf_thresh", conf_thresh, iter_num)

            lr_ = base_lr * (1 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            if iter_num % 50 == 0:
                image = (
                    img_x[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                )
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image("train/Image", grid_image, iter_num)

                outputs_soft = F.softmax(pred_x, 1)
                image = (
                    outputs_soft[0, 1:2, :, :, 20:61:10]
                    .permute(3, 0, 1, 2)
                    .repeat(1, 3, 1, 1)
                )
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image("train/Predicted_label", grid_image, iter_num)

                image = (
                    mask_x[0, :, :, 20:61:10]
                    .unsqueeze(0)
                    .permute(3, 0, 1, 2)
                    .repeat(1, 3, 1, 1)
                )
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image("train/Groundtruth_label", grid_image, iter_num)

            if iter_num > max_iterations:
                save_mode_path = os.path.join(
                    snapshot_path, "iter_" + str(iter_num) + ".pth"
                )
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                print("finish training, iter_num > max_iterations")
                break
            time1 = time.time()
        if iter_num > max_iterations:
            print("finish training")
            break

        if (epoch_num + 1) % 20 == 0 and False:
            # evals``
            net.eval()
            with torch.no_grad():
                with open(args.root_path + "./test.list", "r") as f:
                    image_list = f.readlines()
                image_list = [
                    args.root_path + item.replace("\n", "") + "/mri_norm2.h5"
                    for item in image_list
                ]

                dice, jc, hd, asd = test_all_case(
                    net,
                    image_list,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    stride_xy=18,
                    stride_z=4,
                    save_result=False,
                    test_save_path=None,
                )

                if dice > pervious_bset_dice:
                    pervious_bset_dice = dice
                    save_mode_path = os.path.join(snapshot_path, "best_model.pth")
                    torch.save(net.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

    save_mode_path = os.path.join(
        snapshot_path, "iter_" + str(max_iterations + 1) + ".pth"
    )
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
