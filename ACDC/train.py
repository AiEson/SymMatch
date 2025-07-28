import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from dataset.acdc import ACDCDataset
from model.unet import UNet
from util.classes import CLASSES
from util.utils import (
    AverageMeter,
    count_params,
    init_log,
    DiceLoss,
)
from util.dist_helper import setup_distributed
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


parser = argparse.ArgumentParser(
    description="SymMatch: Symmetric Bi-Scale Matching with Self-Knowledge Distillation in Semi-Supervised Medical Image Segmentation"
)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--labeled-id-path", type=str, required=True)
parser.add_argument("--unlabeled-id-path", type=str, required=True)
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--port", default=None, type=int)
parser.add_argument("--eta", default=0.3, type=float)
parser.add_argument(
    "--drop_rate_img", type=float, default=0.25, help="the image drop rate"
)
parser.add_argument(
    "--drop_rate_feat", type=float, default=0.5, help="the feature drop rate"
)
parser.add_argument(
    "--pert_type", type=str, default="alpdrop", help="the type of image perturbation"
)
parser.add_argument(
    "--feat_pert_type",
    type=str,
    default="dropout",
    help="the type of feature perturbation",
)
parser.add_argument("--loss_kd", action="store_true", help="whether use loss_kd")
parser.add_argument("--loss_mse", action="store_true", help="whether use loss_mse")
parser.add_argument("--T", type=float, default=2.0, help="Temperature for distillation")
parser.add_argument(
    "--distill_loss", type=str, default="KL", help="distillation loss type"
)

DROP_TYPE_DICT = {
    "dropout": nn.Dropout2d,
    "alpdrop": nn.AlphaDropout,
    "featalpdrop": nn.FeatureAlphaDropout,
}


def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    conf_thresh = cfg["conf_thresh"]

    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    model = UNet(in_chns=1, class_num=cfg["nclass"], drop_rate_feat=args.drop_rate_feat)
    if rank == 0:
        logger.info("Total params: {:.1f}M\n".format(count_params(model)))

    optimizer = SGD(model.parameters(), cfg["lr"], weight_decay=0.0001, momentum=0.9)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=False,
    )

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg["nclass"])

    trainset_u = ACDCDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_u",
        cfg["crop_size"],
        args.unlabeled_id_path,
    )
    trainset_l = ACDCDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_l",
        cfg["crop_size"],
        args.labeled_id_path,
        nsample=len(trainset_u.ids),
    )
    valset = ACDCDataset(cfg["dataset"], cfg["data_root"], "val")

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=cfg["batch_size"] * 3,
        drop_last=True,
        sampler=trainsampler_l,
    )
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=cfg["batch_size"] * 3,
        drop_last=True,
        sampler=trainsampler_u,
    )

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=3,
        drop_last=False,
        sampler=valsampler,
        pin_memory_device=f"cuda:{local_rank}",
    )

    total_iters = len(trainloader_u) * cfg["epochs"]
    previous_best = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        checkpoint = torch.load(os.path.join(args.save_path, "latest.pth"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]

        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

    for epoch in range(epoch + 1, cfg["epochs"]):
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}, TH: {:.4f}".format(
                    epoch, optimizer.param_groups[0]["lr"], previous_best, conf_thresh
                )
            )

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_u = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        for i, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),
        ) in enumerate(loader):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()

            # Apply image-level perturbations
            img_u_w_drop = DROP_TYPE_DICT[args.pert_type](args.drop_rate_img)(
                img_u_w.clone()
            )
            img_u_s1_drop = DROP_TYPE_DICT[args.pert_type](args.drop_rate_img)(
                img_u_s1.clone()
            )
            img_u_s2_drop = DROP_TYPE_DICT[args.pert_type](args.drop_rate_img)(
                img_u_s2.clone()
            )

            model.train()

            pred_x = model(img_x)

            # Teacher network (left side) - no feature perturbation
            preds_ws1s2, bottle_ws1s2 = model(
                torch.cat((img_u_w, img_u_s1, img_u_s2)), return_buttleneck=True
            )

            # Student network (right side) - with feature perturbation
            preds_ws1s2_drop, bottle_ws1s2_drop = model(
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

            loss_x = (
                criterion_ce(pred_x, mask_x)
                + criterion_dice(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float())
            ) / 2.0

            loss_u_s1 = criterion_dice(
                (pred_u_s1).softmax(dim=1),
                mask_u_w.unsqueeze(1).float(),
                ignore=(conf_u_w < conf_thresh).float(),
            )

            loss_u_s2 = criterion_dice(
                (pred_u_s2).softmax(dim=1),
                mask_u_w.unsqueeze(1).float(),
                ignore=(conf_u_w < conf_thresh).float(),
            )

            # SymMatch Bi-scale Distillation Loss
            if args.loss_kd:
                loss_u_w_drop = criterion_dice(
                    pred_u_w_drop.softmax(dim=1),
                    mask_u_w.unsqueeze(1).float(),
                    ignore=(conf_u_w < conf_thresh).float(),
                )

                loss_u_s1_drop = criterion_dice(
                    pred_u_s1_drop.softmax(dim=1),
                    mask_u_w.unsqueeze(1).float(),
                    ignore=(conf_u_w < conf_thresh).float(),
                )

                loss_u_s2_drop = criterion_dice(
                    pred_u_s2_drop.softmax(dim=1),
                    mask_u_w.unsqueeze(1).float(),
                    ignore=(conf_u_w < conf_thresh).float(),
                )
            else:
                loss_u_w_drop = 0
                loss_u_s1_drop = 0
                loss_u_s2_drop = 0

            # Feature-level distillation loss (L^f in the paper)
            if args.loss_mse:
                if args.distill_loss == "MSE":
                    loss_mse_u_w = F.mse_loss(
                        bottle_u_w.detach() / args.T, bottle_u_w_drop / args.T
                    )
                    loss_mse_u_s1 = F.mse_loss(
                        bottle_u_s1.detach() / args.T, bottle_u_s1_drop / args.T
                    )
                    loss_mse_u_s2 = F.mse_loss(
                        bottle_u_s2.detach() / args.T, bottle_u_s2_drop / args.T
                    )
                elif args.distill_loss == "KL":
                    loss_mse_u_w = F.kl_div(
                        F.log_softmax(bottle_u_w_drop / args.T, dim=1),
                        F.softmax(bottle_u_w.detach() / args.T, dim=1),
                    )
                    loss_mse_u_s1 = F.kl_div(
                        F.log_softmax(bottle_u_s1_drop / args.T, dim=1),
                        F.softmax(bottle_u_s1.detach() / args.T, dim=1),
                    )
                    loss_mse_u_s2 = F.kl_div(
                        F.log_softmax(bottle_u_s2_drop / args.T, dim=1),
                        F.softmax(bottle_u_s2.detach() / args.T, dim=1),
                    )
                elif args.distill_loss == "CE":
                    loss_mse_u_w = (
                        -(
                            F.log_softmax(bottle_u_w_drop / args.T, dim=1)
                            * F.softmax(bottle_u_w.detach() / args.T, dim=1)
                        )
                        .sum(dim=1)
                        .mean()
                    )
                    loss_mse_u_s1 = (
                        -(
                            F.log_softmax(bottle_u_s1_drop / args.T, dim=1)
                            * F.softmax(bottle_u_s1.detach() / args.T, dim=1)
                        )
                        .sum(dim=1)
                        .mean()
                    )
                    loss_mse_u_s2 = (
                        -(
                            F.log_softmax(bottle_u_s2_drop / args.T, dim=1)
                            * F.softmax(bottle_u_s2.detach() / args.T, dim=1)
                        )
                        .sum(dim=1)
                        .mean()
                    )
            else:
                loss_mse_u_w = 0
                loss_mse_u_s1 = 0
                loss_mse_u_s2 = 0

            # Prediction-level loss (L^p in the paper)
            loss_kd = (loss_u_w_drop + loss_u_s1_drop + loss_u_s2_drop) / 3.0
            # Feature-level loss (L^f in the paper)
            loss_mse = (loss_mse_u_w + loss_mse_u_s1 + loss_mse_u_s2) / 3.0

            # Total unlabeled loss (L^u = L^p + L^f)
            loss_u = loss_kd + loss_mse

            # Final loss following SymMatch formulation
            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u * 0.5) / 2.0

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_u.update(loss_u.item())

            mask_ratio = (conf_u_w >= conf_thresh).sum() / conf_u_w.numel()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i

            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            # SymMatch uses fixed confidence threshold, no dynamic threshold updating

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar(
                    "train/loss_s", (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters
                )
                writer.add_scalar("train/loss_u", loss_u.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss u: {:.3f}, Mask ratio: {:.3f}".format(
                        i,
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_s.avg,
                        total_loss_u.avg,
                        total_mask_ratio.avg,
                    )
                )

        model.eval()

        dice_class = [0] * 3

        with torch.no_grad():
            for img, mask in valloader:
                img, mask = img.cuda(), mask.cuda()

                h, w = img.shape[-2:]
                img = F.interpolate(
                    img,
                    (cfg["crop_size"], cfg["crop_size"]),
                    mode="bilinear",
                    align_corners=False,
                )

                img = img.permute(1, 0, 2, 3)

                pred = model(img)

                pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
                pred = pred.argmax(dim=1).unsqueeze(0)

                for cls in range(1, cfg["nclass"]):
                    inter = ((pred == cls) * (mask == cls)).sum().item()
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    dice_class[cls - 1] += 2.0 * inter / union

        dice_class = [dice * 100.0 / len(valloader) for dice in dice_class]
        mean_dice = sum(dice_class) / len(dice_class)

        if rank == 0:
            for cls_idx, dice in enumerate(dice_class):
                logger.info(
                    "***** Evaluation ***** >>>> Class [{:} {:}] Dice: "
                    "{:.2f}".format(cls_idx, CLASSES[cfg["dataset"]][cls_idx], dice)
                )
            logger.info(
                "***** Evaluation ***** >>>> MeanDice: {:.2f}\n".format(mean_dice)
            )

            writer.add_scalar("eval/MeanDice", mean_dice, epoch)
            for i, dice in enumerate(dice_class):
                writer.add_scalar(
                    "eval/%s_dice" % (CLASSES[cfg["dataset"]][i]), dice, epoch
                )

        is_best = mean_dice > previous_best
        previous_best = max(mean_dice, previous_best)
        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))


if __name__ == "__main__":
    main()
