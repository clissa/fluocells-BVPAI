"""
This script tests model generalization when used for inference on new data.

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-07-16
License: Apache License 2.0
"""

import sys
import inspect
from pathlib import Path

SCRIPT_PATH = inspect.getfile(inspect.currentframe())
FLUOCELLS_PATH = Path(SCRIPT_PATH).parent.absolute()

sys.path.append(str(FLUOCELLS_PATH))

from fastai.vision.all import *
from fluocells.config import (
    REPO_PATH,
    DATA_PATH,
    DATA_PATH_g,
    DATA_PATH_y,
    DATA_PATH_r,
    METADATA,
    MODELS_PATH,
)
from fluocells.utils.data import post_process
from fluocells.models import cResUnet, c_resunet
from fluocells.utils.metrics import eval_prediction


torch.set_printoptions(precision=10)

import argparse

parser = argparse.ArgumentParser(description="Run a basic training pipeline")

# Add the dataset argument
parser.add_argument(
    "dataset",
    type=str,
    choices=["green", "yellow", "red"],
    help="Dataset to train on: green, yellow, or red",
)

# Add the experiment argument
parser.add_argument(
    "experiment",
    type=str,
    help="Name of the experiment folder. Needed for setup of input/output paths",
)

# Add the threshold argument
parser.add_argument(
    "--bin_thresh",
    type=float,
    default=0.5,
    help="Threshold for heatmap binarization (default: 0.5)",
)

# Add the hole_size argument
parser.add_argument(
    "--smooth_disk",
    type=int,
    default=0,
    help="Size of disk used to smoothing object contours (default: 0)",
)

# Add the hole_size argument
parser.add_argument(
    "--max_hole",
    type=int,
    default=50,
    help="Maximum hole size to fill (default: 50)",
)

# Add the min_size argument
parser.add_argument(
    "--min_size",
    type=int,
    default=200,
    help="Minimum allowed object size. Smaller objects are removed (default: 200)",
)

# Add the max_filt argument
parser.add_argument(
    "--max_dist",
    type=int,
    default=30,
    help="Max filter argument in ndimage.maximum_filter(default: 30)",
)

# Add the footprint argument
parser.add_argument(
    "--fp",
    type=int,
    default=40,
    help="Footprint argument in peak_local_maxi (default: 40)",
)

# Add the iou threshold argument
parser.add_argument(
    "--iou_thresh",
    type=float,
    default=0.5,
    help="Threshold used as IoU overlapping to determine true positives (default: 0.5)",
)

# Add the footprint argument
parser.add_argument(
    "--prox_thresh",
    type=int,
    default=40,
    help="Threshold used for centers matching to determine true positives (default: 40)",
)

# Add the device argument
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "cuda"],
    default="cpu",
    help="Device to use to get model prediction (affect dataloader and model) (default: 'cpu')",
)


def label_func(p):
    return Path(str(p).replace("rawimages", "groundtruth"))


def instance_to_semantic_mask(x: Image.Image):
    x = np.array(x)
    x[x > 0] = 1
    return PILMask.create(x)


BS = 1
DATASET = "S-BSST265"
SEED = 47
VAL_PCT = 0.2
dataset_path = REPO_PATH / "public_datasets" / DATASET / "dataset"

set_seed(SEED)

# model params
N_IN, N_OUT = 16, 2
PRETRAINED = False

# optimizer params
LOSS_FUNC, LOSS_NAME = (
    DiceLoss(axis=1, smooth=1e-06, reduction="mean", square_in_union=False),
    "Dice",
)

EXP_NAME = "yellow_5_FT_default"
log_path = REPO_PATH / "logs" / EXP_NAME
model_path = MODELS_PATH / EXP_NAME
results_path = REPO_PATH / "results" / DATASET / EXP_NAME
results_path.mkdir(exist_ok=True, parents=True)

DEVICE = "cpu"
def label_func(p):
    return Path(str(p).replace("rawimages", "groundtruth"))


trainval_path = dataset_path / "rawimages" # edit dataset folder here: DATA_PATH_g --> green; DATA_PATH_y --> yellow; DATA_PATH_r --> red 

# read train/valid/test split dataframe
trainval_fnames = [fn for fn in trainval_path.iterdir()]

# augmentation
tfms = [
    # IntToFloatTensor(div_mask=255.),  # need masks in [0, 1] format
    Resize((1024, 1360), method=ResizeMethod.Pad, pad_mode="zeros")
]

# splitter
splitter = RandomSplitter(valid_pct=VAL_PCT)

# dataloader
dls = SegmentationDataLoaders.from_label_func(
    DATA_PATH, fnames=trainval_fnames, label_func=label_func,
    bs=BS,
    splitter=splitter,
    item_tfms=tfms,
    device=DEVICE 
)

test_dl = dls.test_dl(trainval_fnames[:2], with_labels=True, tfms=tfms)


# set up Learner
from fluocells.models import cResUnet, c_resunet


torch.set_printoptions(precision=10)

arch = "c-ResUnet"
# pretrained=True would load Morelli et al. 2021 weights. We add new pretrained weights after
cresunet = c_resunet(
    arch=arch, n_features_start=N_IN, n_out=N_OUT, pretrained=PRETRAINED # this would load Morelli et al. 2022
)
# cresunet = cResUnet(cfg.n_in, cfg.n_out)

learn = Learner(dls, model=cresunet, loss_func=LOSS_FUNC,
                metrics=[Dice(), JaccardCoeff(), foreground_acc],
                path=log_path , 
                model_dir=model_path,
                )  

print(
    f'Logs save path: {learn.path}\nModel save path: {learn.path / learn.model_dir}')

print(f"Loading pesi da: {model_path}")

learn.load(model_path / 'model') # model.pth
learn.eval()

# compute metrics
C = 1
metrics_df = pd.DataFrame(
    {}, columns="TP_iou FP_iou FN_iou TP_prox FP_prox FN_prox".split(" ")
)
for i, b in enumerate(tqdm(test_dl)):
    image_name = test_dl.items[i].name
    img, mask = b
    heatmap = (
        learn.model(img).squeeze().permute(1, 2, 0)[:, :, C].detach().to("cpu")
    )

    # convert to matplotlib format
    img = img.squeeze().permute(1, 2, 0).to("cpu")
    thresh_image = np.squeeze(
        (heatmap.numpy() > postproc_cfg.bin_thresh).astype("uint8")
    )
    post_proc_mask = post_process(
        thresh_image,
        smooth_disk=postproc_cfg.smooth_disk,
        max_hole_size=postproc_cfg.max_hole,
        min_object_size=postproc_cfg.min_size,
        max_filter_size=postproc_cfg.max_dist,
        footprint=postproc_cfg.fp,
    )
