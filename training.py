"""
This scripts perform a basic training pipeline. It takes 4 CLI arguments:

 - dataset (yellow): dataset to train on #TODO: add public datasets option?
 - loss (Dice, BCE, Focal, FT, Combined or CombinedFT): loss function
 - seed: initialization for data split and network weights/training cycle
 - gpu_id: id of the gpu to use for training

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-06-28
License: Apache License 2.0
"""

import sys
import inspect
from pathlib import Path

SCRIPT_PATH = inspect.getfile(inspect.currentframe())
FLUOCELLS_PATH = Path(SCRIPT_PATH).absolute()

sys.path.append(str(FLUOCELLS_PATH))

import os
import json
from random import randint
from fastai.vision.all import *
from fastai.losses import FocalLossFlat
from fastai.torch_basics import set_seed
from fluocells.config import (
    REPO_PATH,
    DATA_PATH,
    DATA_PATH_g,
    DATA_PATH_y,
    DATA_PATH_r,
    METADATA,
    MODELS_PATH,
)
from fluocells.models import cResUnet, c_resunet
from fluocells.models._losses import FocalTverskyLoss, CombinedLoss, CombinedFTLoss

import argparse

parser = argparse.ArgumentParser(description="Run a basic training pipeline")

# Add the dataset argument
parser.add_argument(
    "dataset",
    type=str,
    choices=["yellow"],
    help="Dataset to train on: yellow",
)

# Add the loss argument
parser.add_argument(
    "--loss",
    type=str,
    choices=["Dice", "BCE", "Focal", "FT", "Combined", "CombinedFT"],
    default="Dice",
    help="Loss function. Either Dice, (Weighted) Binary Cross Entropy, Focal, Focal Tversky, Combined or CombinedFT  (default: 'Dice')",
)

# Add the loss weights
parser.add_argument(
    "--loss_weights",
    type=str,
    choices=["balanced", "overcrowd", "cellvit"],
    default="cellvit",
    help="""Configuration of weight for BCE, Dice and Focal/FocalTversky loss terms (used only with `Combined*` lossed). Options:
    - balanced:
    - overcrowd:
    - cellvit:
    (default: 'cellvit').""",
)

# Add the W_CELL argument
parser.add_argument(
    "--w_cell",
    type=int,
    default=1,
    help="Weight for cell class in case loss is BCE (default: 200)",
)

# Add the seed argument
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Seed value, random if not set (default: None)",
)

# Add the gpu_id argument
parser.add_argument(
    "--gpu_id", type=int, choices=[0, 1, 2, 3], default=0, help="GPU ID (default: 0)"
)

args = parser.parse_args()

# torch.set_printoptions(precision=10)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


def label_func(p):
    return Path(str(p).replace("images", "ground_truths/masks"))


DATASET = args.dataset
SEED = args.seed

# EXPERIMENT CONFIGURATION
# reproducibility
if SEED is None:
    SEED = randint(0, 100)
VAL_PCT = 0.2

# data and augmentation params
EPOCHS_SCRATCH = 200
# EPOCHS_FINETUNE = 50
BS = 32
CROP_SIZE = 512
# RESIZE = 512
MAX_LIGHT = 0.1
ZOOM_DELTA = 0.1  # min_zoom = 1 - ZOOM_DELTA; max_zoom = 1 + ZOOM_DELTA
MAX_ROTATION_ANGLE = 15.0

# model params
N_IN, N_OUT = 16, 2
PRETRAINED = False

# optimizer params
if args.loss == "Dice":
    LOSS_FUNC = DiceLoss(axis=1, smooth=1e-06, reduction="mean", square_in_union=False)
    LOSS_NAME = "Dice"
elif args.loss == "BCE":
    # optimizer params
    W_CELL, W_BKGD = args.w_cell, 1
    LOSS_FUNC = CrossEntropyLossFlat(axis=1, weight=torch.Tensor([W_BKGD, W_CELL]))
    LOSS_NAME = f"BCE_wcell={W_CELL}"
elif args.loss == "Focal":
    GAMMA = 2.0
    LOSS_FUNC = FocalLossFlat(axis=1, gamma=GAMMA, weight=None, reduction="mean")
    LOSS_NAME = f"Focal_gamma={GAMMA}"
elif args.loss == "FT":
    GAMMA = 2.0
    LOSS_FUNC = FocalTverskyLoss()
    LOSS_NAME = f"FT_default"
elif args.loss == "Combined":
    weights_config = {
        "balanced": (0.3, 0.3, 0.4),
        "overcrowd": (0.2, 0.5, 0.3),
        "cellvit": (0.5, 0.2, 0.5),
    }
    WEIGHTS = weights_config[args.loss_weights]
    LOSS_FUNC = CombinedLoss(
        axis=1,
        w_bce=WEIGHTS[0],
        w_dice=WEIGHTS[1],
        w_focal=WEIGHTS[2],
    )
    LOSS_NAME = f"Combined_weights={WEIGHTS}"
elif args.loss == "CombinedFT":
    weights_config = {
        "balanced": (0.3, 0.3, 0.4),
        "overcrowd": (0.2, 0.5, 0.3),
        "cellvit": (0.5, 0.2, 0.5),
    }
    WEIGHTS = weights_config[args.loss_weights]
    LOSS_FUNC = CombinedFTLoss(
        axis=1,
        w_bce=WEIGHTS[0],
        w_dice=WEIGHTS[1],
        w_focal=WEIGHTS[2],
    )
    LOSS_NAME = f"CombinedFT_weights={WEIGHTS}"


LR = None
OPT, OPT_NAME = partial(Adam, lr=LR), "Adam"
MONIT_SCORE, MIN_DELTA, PATIENCE_ES = "dice", 0.005, 20  # early stopping
FACTOR, PATIENCE_LR = 1.2, 4  # scheduling learning rate


hyperparameter_defaults = dict(
    # reproducibility
    seed=SEED,
    val_pct=VAL_PCT,
    # dataloader
    epochs_scratch=EPOCHS_SCRATCH,
    # epochs_finetune=EPOCHS_FINETUNE,
    batch_size=BS,
    crop_size=CROP_SIZE,
    # resize=RESIZE,
    angle=MAX_ROTATION_ANGLE,
    zoom_delta=ZOOM_DELTA,
    max_light=MAX_LIGHT,
    # model
    n_in=N_IN,
    n_out=N_OUT,
    pretrained=PRETRAINED,
    # optimizer
    loss_func=LOSS_FUNC,
    loss_name=LOSS_NAME,
    lr=LR,
    opt=OPT,
    opt_name=OPT_NAME,
    monit_dict={"score": MONIT_SCORE, "min_delta": MIN_DELTA, "patience": PATIENCE_ES},
    lr_monit_dict={"factor": FACTOR, "patience": PATIENCE_LR},
)

cfg = namedtuple("Config", hyperparameter_defaults.keys())(**hyperparameter_defaults)

EXP_NAME = f"{DATASET}_{SEED}_{LOSS_NAME}"
LOG_PATH = REPO_PATH / "logs" / EXP_NAME
LOG_PATH.mkdir(exist_ok=True, parents=True)

model_path = f"{MODELS_PATH / EXP_NAME}"

print(f"Running experiment: {EXP_NAME}")
print(f"Log path set to: {LOG_PATH}")
print(f"Model weights will be stored in: {model_path}")


def main(dataset, gpu_id, cfg):
    torch.set_printoptions(precision=10)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # edit dataset folder here: DATA_PATH_g --> green; DATA_PATH_y --> yellow; DATA_PATH_r --> red
    dataset_path = globals()[f"DATA_PATH_{dataset[0]}"]

    trainval_path = dataset_path / "trainval" / "images"
    trainval_fnames = [fn for fn in trainval_path.iterdir()]

    # augmentation
    tfms = [
        IntToFloatTensor(div_mask=255.0),  # need masks in [0, 1] format
        # RandomCrop(cfg.crop_size),
        *aug_transforms(
            # size=cfg.resize,  # resize
            max_lighting=cfg.max_light,
            p_lighting=0.5,  # luminosity variation
            min_zoom=1 - cfg.zoom_delta,
            max_zoom=1 + cfg.zoom_delta,  # zoom
            max_warp=0,  # distorsion
            max_rotate=cfg.angle,  # rotation
        ),
    ]

    set_seed(cfg.seed)
    # splitter
    splitter = RandomSplitter(valid_pct=cfg.val_pct)  # , seed=SEED)

    # dataloader
    dls = SegmentationDataLoaders.from_label_func(
        DATA_PATH,
        fnames=trainval_fnames,
        label_func=label_func,
        bs=cfg.batch_size,
        splitter=splitter,
        item_tfms=RandomCrop(cfg.crop_size),
        batch_tfms=tfms,
        device="cuda",
        num_workers=10,
    )

    # initialize model
    arch = "c-ResUnet"
    cresunet = c_resunet(
        arch=arch, n_features_start=cfg.n_in, n_out=cfg.n_out, pretrained=cfg.pretrained
    )

    # callbacks
    CBS = [
        SaveModelCallback(
            monitor=cfg.monit_dict["score"],
            min_delta=cfg.monit_dict["min_delta"],
            fname="model",
            # fname=f"model_{EXP_NAME}",
            every_epoch=False,
            at_end=False,
            with_opt=True,
            reset_on_fit=False,
        ),
        # CSVLogger(fname=f"history_{EXP_NAME}.csv", append=True), # for some reason it doesn't work
        CSVLogger,
    ]

    # learner
    learn = Learner(
        dls,
        model=cresunet,
        loss_func=cfg.loss_func,
        metrics=[Dice(), JaccardCoeff(), foreground_acc],
        cbs=CBS,
        path=LOG_PATH,
        model_dir=model_path,
    )

    print(
        f"Logs save path: {learn.path}\nModel save path: {learn.path / learn.model_dir}"
    )

    print(learn.summary())

    # suggested learning rate if not set
    if cfg.lr is None:
        res = learn.lr_find()
        print("Suggested starting learning rate at:\t", res.valley)

        cfg._replace(lr=res.valley)
        cfg._replace(opt=partial(Adam, lr=cfg.lr))
        hyperparameter_defaults["lr"] = res.valley

    with open(LOG_PATH / f"config.json", "w") as f:
        # with open(LOG_PATH / f"config_{EXP_NAME}.json", "w") as f:
        json.dump(
            {
                k: v
                for k, v in hyperparameter_defaults.items()
                if k not in ["opt", "loss_func"]
            },
            f,
        )

    learn.fit_one_cycle(n_epoch=cfg.epochs_scratch, lr_max=cfg.lr, cbs=CBS)


if __name__ == "__main__":
    main(args.dataset, args.gpu_id, cfg)
