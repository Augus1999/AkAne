# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Useful tools (hopefully :)
"""
import os
import re
import glob
import logging
from pathlib import Path
from typing import Optional, List, Dict, Iterable
import torch
import torch.nn as nn
import torch.optim as op
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from .graph import gather

# set 'num_workers' varibale in torch.utils.data.DataLoader
os.environ["NUM_WORKER"] = "4"
if hasattr(os, "sched_getaffinity"):
    os.environ["NUM_WORKER"] = f"{len(os.sched_getaffinity(0))}"
# set inference batch-size
os.environ["INFERENCE_BATCH_SIZE"] = "20"


def collate(batch: List) -> Dict[str, Tensor]:
    """
    Padding the data in one batch into the same size

    :param batch: a list of data (one batch)
    :return: batched {"mol": mol, "label": label}
    """
    mol = [i["mol"] for i in batch]
    label = [i["label"] for i in batch]
    token_input = [i["token_input"] for i in label]
    token_label = [i["token_label"] for i in label]
    max_len = max([len(w) for w in token_input])
    token_input = [
        F.pad(i, (0, max_len - len(i)), value=0)[None, :] for i in token_input
    ]
    token_label = [
        F.pad(i, (0, max_len - len(i)), value=0)[None, :] for i in token_label
    ]
    mol = gather(batch=mol)
    token_input = torch.cat(token_input, dim=0)
    token_label = torch.cat(token_label, dim=0)
    label_ = {"token_input": token_input, "token_label": token_label}
    if "property" in batch[0]["label"]:
        property = [i["property"] for i in label]
        max_len = max([len(w) for w in property])
        property = [F.pad(i, (0, max_len - len(i)), value=0)[None, :] for i in property]
        property = torch.cat(property, dim=0)
        label_["property"] = property
    return {"mol": mol, "label": label_}


def train(
    model: nn.Module,
    data: Iterable,
    mode: str = "predict",
    n_epochs: int = 800,
    batch_size: int = 5,
    load: Optional[str] = None,
    log_dir: Optional[str] = None,
    work_dir: str = "workdir",
    save_every: Optional[int] = None,
) -> None:
    """
    Trian the network.

    :param model: model for training
    :param data: training data
    :param mode: training mode selected from
                 'autoencoder', 'predict', 'classify', and 'diffusion'
    :param n_epochs: training epochs size
    :param batch_size: batch size
    :param load: load from an existence state file <file>
    :param log_dir: where to store log file <file>
    :param work_dir: where to store model state_dict <path>
    :param save_every: store checkpoint every 'save_every' epochs
    :return: None
    """
    assert mode in ("autoencoder", "predict", "classify", "diffusion")
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    logging.basicConfig(
        filename=log_dir,
        format=" %(asctime)s %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )
    start_epoch: int = 0
    data_size = len(data)
    loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=int(os.environ["NUM_WORKER"]),
    )
    train_size = len(loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    if hasattr(model, "encoder"):
        model.encoder.to(device)
        if mode == "autoencoder":
            model.decoder.to(device)
        if mode == "predict" or mode == "classify":
            model.readout.to(device)
        if mode == "diffusion":
            model.dit.to(device)
    else:
        model.to(device)
    optimizer = op.Adam(model.parameters(), lr=1e-6, amsgrad=False)
    scheduler = op.lr_scheduler.CyclicLR(
        optimizer=optimizer,
        base_lr=1e-6,
        max_lr=1e-5 if mode == "autoencoder" else 1e-4,
        step_size_up=1000,
        step_size_down=None if mode == "autoencoder" else n_epochs * train_size - 1000,
        cycle_momentum=False,
    )
    logging.info(f"using hardware {device}")
    logging.info(f"loaded {data_size} data in the dataset")
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.debug(f"{n_param} trainable parameters")
    if load:
        with open(load, mode="rb") as sf:
            state_dic = torch.load(sf, map_location=device)
        keys = {"nn", "opt", "epoch", "scheduler"}
        if keys & set(state_dic.keys()) == keys:
            scheduler.load_state_dict(state_dict=state_dic["scheduler"])
            start_epoch: int = state_dic["epoch"]
            optimizer.load_state_dict(state_dict=state_dic["opt"])
        model.pretrained(load)
        logging.info(f'loaded state from "{load}"')
    logging.info("start training")
    for epoch in range(start_epoch, n_epochs):
        running_loss = 0.0
        for d in loader:
            mol, label = d["mol"], d["label"]
            for i in mol:
                mol[i] = mol[i].to(device)
            if mode == "predict":
                property = label["property"].to(device)
                if hasattr(model, "predict_train_step"):
                    loss = model.predict_train_step(mol, property)
                else:
                    y = model(mol)["prediction"]
                    label_mask = (property != torch.inf).float()
                    property = property.masked_fill(property == torch.inf, 0)
                    loss = F.mse_loss(y * label_mask, property, reduction="mean")
            if mode == "classify":
                property = label["property"].to(device, torch.long)
                if hasattr(model, "classify_train_step"):
                    loss = model.classify_train_step(mol, property)
                else:
                    y = model(mol)["prediction"]
                    loss = F.cross_entropy(y, property.reshape(-1))
            if mode == "autoencoder":
                token_input = label["token_input"].to(device)
                token_label = label["token_label"].to(device)
                loss = model.autoencoder_train_step(mol, token_input, token_label)
            if mode == "diffusion":
                property = label["property"].to(device)
                loss = model.diffusion_train_step(mol, property)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            scheduler.step()  # update scheduler after optimised
        logging.info(f"epoch: {epoch + 1} loss: {running_loss / train_size}")
        if save_every:
            if (epoch + 1) % save_every == 0:
                chkpt_idx = str(epoch + 1).zfill(len(str(n_epochs)))
                if hasattr(model, "encoder"):
                    nn = {
                        "encoder": model.encoder.state_dict(),
                        "decoder": model.decoder.state_dict(),
                    }
                    if mode == "predict" or mode == "classify":
                        nn["readout"] = model.readout.state_dict()
                    if mode == "diffusion":
                        nn["dit"] = model.dit.state_dict()
                else:
                    nn = model.state_dict()
                state = {
                    "nn": nn,
                    "opt": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                }
                torch.save(state, Path(work_dir) / f"state-{chkpt_idx}.pth")
                logging.info(f"saved checkpoint state-{chkpt_idx}.pth")
    if hasattr(model, "encoder"):
        nn = {
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
        }
        if mode == "predict" or mode == "classify":
            nn["readout"] = model.readout.state_dict()
        if mode == "diffusion":
            nn["dit"] = model.dit.state_dict()
    else:
        nn = model.state_dict()
    torch.save({"nn": nn}, Path(work_dir) / "trained.pt")
    logging.info("saved state!")
    logging.info("finished")


@torch.no_grad()
def test(
    model: nn.Module,
    data: Iterable,
    mode: str = "regression",
    load: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Test the trained network.

    :param model: model for testing
    :param data: data
    :param mode: testing mode chosen from 'regression' and 'classification'
    :param load: load from an existence state file <file>
    :param log_dir: where to store log file <file>
    :return: MAE & RMSE & MAPE of property
    """
    logging.basicConfig(
        filename=log_dir,
        format=" %(asctime)s %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )
    data_size = len(data)
    loader = DataLoader(
        data, int(os.environ["INFERENCE_BATCH_SIZE"]), collate_fn=collate
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"using hardware {device}")
    logging.info(f"loaded {data_size} data in the dataset")
    if load:
        model = model.pretrained(file=load)
    model.eval()
    if hasattr(model, "encoder"):
        model.encoder.to(device)
        model.readout.to(device)
    else:
        model.to(device)
    logging.info(f'loaded state from "{load}"')
    tae, tse, tape, scale = 0, 0, 0, 0
    predict_cls, label_cls = [], []
    for d in loader:
        mol, label = d["mol"], d["label"]
        for i in mol:
            mol[i] = mol[i].to(device)
        property = label["property"]
        if mode == "regression":
            property = property.to(device)
            label_mask = (property != torch.inf).float()
            property_ = property.masked_fill(property == torch.inf, 0)
            if hasattr(model, "inference"):
                out_property = model.inference(mol)["prediction"] * label_mask
            else:
                out_property = model(mol)["prediction"] * label_mask
            l1 = (out_property - property_).abs().sum(dim=0).detach()
            l2 = (out_property - property_).pow(2).sum(dim=0).detach()
            l3 = ((out_property - property_).abs() / property).sum(dim=0).detach()
            tae += l1.to("cpu").numpy()
            tse += l2.to("cpu").numpy()
            tape += l3.to("cpu").numpy()
            scale += label_mask.sum(dim=0).detach().to("cpu").numpy()
        if mode == "classification":
            if hasattr(model, "inference"):
                out_property = model.inference(mol)["prediction"]
            else:
                out_property = model(mol)["prediction"]
            out_property = F.softmax(out_property, dim=-1)
            predict_cls.append(out_property.detach().to("cpu"))
            label_cls.append(property.flatten())
    if mode == "regression":
        MAE = list(tae / scale)
        RMSE = list((tse / scale) ** 0.5)
        MAPE = list(100 * tape / scale)
        logging.info(f"MAE: {MAE}, RMSE: {RMSE}, MAPE: {MAPE}")
        return {"MAE": MAE, "RMSE": RMSE, "MAPE": MAPE}
    if mode == "classification":
        predict_cls = torch.cat(predict_cls, dim=0).numpy()
        label_cls = torch.cat(label_cls, dim=-1).numpy()
        roc_auc = roc_auc_score(label_cls, predict_cls[:, 1])
        precision, recall, _ = precision_recall_curve(label_cls, predict_cls[:, 1])
        prc_auc = auc(recall, precision)
        logging.info(f"ROC-AUC: {roc_auc}, PRC-AUC: {prc_auc}")
        return {"ROC-AUC": roc_auc, "PRC-AUC": prc_auc}


def find_recent_checkpoint(workdir: str) -> Optional[str]:
    """
    Find the most recent checkpoint file in the work dir. \n
    The name of checkpoint file is like state-abcdef.pth

    :param workdir: the directory where the checkpoint files stored <path>
    :return: the file name
    """
    load: Optional[str] = None
    if os.path.exists(workdir):
        cps = list(glob.glob(str(Path(workdir) / r"*.pth")))
        if cps:
            cps.sort(key=lambda x: int(os.path.basename(x).split(".")[0].split("-")[1]))
            load = cps[-1]
    return load


def extract_log_info(log_name: str = "training.log") -> Dict[str, List]:
    """
    Extract training loss from training log file.

    :param log_name: log file name <file>
    :return: dict["epoch": epochs, "loss": loss]
    """
    info = {"epoch": [], "loss": []}
    with open(log_name, mode="r", encoding="utf-8") as f:
        lines = f.read()
    loss_info = re.findall(r"epoch: (\d+) loss: (\d+(.\d+(e|E)?(-|\+)?\d+)?)", lines)
    if loss_info:
        for i in loss_info:
            epoch = int(i[0])
            loss = float(i[1])
            info["epoch"].append(epoch)
            info["loss"].append(loss)
    return info
