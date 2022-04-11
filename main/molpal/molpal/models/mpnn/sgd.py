from typing import Dict, Tuple

import ray.util.sgd.v2 as sgd
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import trange

from main.molpal.molpal.models import mpnn
from main.molpal.molpal.models.chemprop.data.data import construct_molecule_batch
from main.molpal.molpal.models.chemprop.nn_utils import NoamLR


def train_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    device: torch.device,
    uncertainty: str = "none",
):
    model.train()

    losses = []
    num_samples = 0

    for i, batch in enumerate(train_loader):
        # batch_info = {"batch_idx": batch_idx}

        step_results = train_step(
            batch,
            model,
            criterion,
            optimizer,
            scheduler,
            device,
            uncertainty,
        )

        losses.append(step_results["loss"])
        num_samples += step_results["num_samples"]

    loss = torch.stack(losses).mean().item()

    return {"loss": loss, "num_samples": num_samples}


def train_step(
    batch: Tuple,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    device: torch.device,
    uncertainty: str = "none",
):
    componentss, targets = batch

    optimizer.zero_grad()

    componentss = [
        [
            X.to(device, non_blocking=True) if isinstance(X, torch.Tensor) else X
            for X in components
        ]
        for components in componentss
    ]

    mask = torch.tensor([[bool(y) for y in ys] for ys in targets], device=device)
    targets = torch.tensor([[y or 0 for y in ys] for ys in targets], device=device)
    class_weights = torch.ones(targets.shape, device=device)

    preds = model(componentss)

    if uncertainty == "mve":
        pred_means = preds[:, 0::2]
        pred_vars = preds[:, 1::2]

        loss = criterion(pred_means, pred_vars, targets)
    else:
        loss = criterion(preds, targets) * class_weights * mask

    loss = loss.sum() / mask.sum()

    loss.backward()
    optimizer.step()
    scheduler.step()

    return {"loss": loss, "num_samples": len(targets)}


def validate_epoch(
    val_loader: DataLoader,
    model: nn.Module,
    metric: nn.Module,
    device: torch.device,
    uncertainty: str = "none",
):
    model.eval()

    losses = []
    num_samples = 0

    for i, batch in enumerate(val_loader):
        # batch_info = {"batch_idx": batch_idx}

        step_results = validate_step(batch, model, metric, device, uncertainty)

        losses.append(step_results["loss"])
        num_samples += step_results["num_samples"]

    loss = torch.cat(losses).mean().item()

    return {"loss": loss, "num_samples": num_samples}


def validate_step(
    batch: Tuple,
    model: nn.Module,
    metric: nn.Module,
    device: torch.device,
    uncertainty: str = "none",
):
    componentss, targets = batch

    model = model
    metric = metric

    componentss = [
        [
            X.to(device, non_blocking=True) if isinstance(X, torch.Tensor) else X
            for X in components
        ]
        for components in componentss
    ]
    targets = torch.tensor(targets, device=device)

    with torch.no_grad():
        preds = model(componentss)

    if uncertainty == "mve":
        preds = preds[:, 0::2]

    loss = metric(preds, targets)
    return {"loss": loss, "num_samples": len(targets)}


def train_func(config: Dict):
    model = config["model"]
    train_data = config["train_data"]
    val_data = config["val_data"]
    uncertainty = config["uncertainty"]
    dataset_type = config["dataset_type"]
    metric = config.get("metric", "rmse")
    batch_size = config.get("batch_size", 50)
    warmup_epochs = config.get("warmup_epochs", 2.0)
    max_epochs = config.get("max_epochs", 50)
    num_lrs = 1
    init_lr = config.get("init_lr", 1e-4)
    max_lr = config.get("max_lr", 1e-3)
    final_lr = config.get("final_lr", 1e-4)
    ncpu = config.get("ncpu", 1)

    device = torch.device(
        f"cuda:{sgd.local_rank()}" if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    model = model.to(device)
    model = DistributedDataParallel(
        model, device_ids=[sgd.local_rank()] if torch.cuda.is_available() else None
    )

    # print(list(model.parameters())[1])

    train_loader = DataLoader(
        train_data,
        batch_size,
        sampler=DistributedSampler(train_data),
        num_workers=ncpu,
        collate_fn=construct_molecule_batch,
    )
    val_loader = DataLoader(
        val_data,
        batch_size,
        sampler=DistributedSampler(val_data),
        num_workers=ncpu,
        collate_fn=construct_molecule_batch,
    )

    optimizer = Adam(model.parameters(), init_lr, weight_decay=0)
    scheduler = NoamLR(
        optimizer=optimizer,
        warmup_epochs=[warmup_epochs],
        total_epochs=[max_epochs] * num_lrs,
        steps_per_epoch=len(train_data) / batch_size + 1,
        init_lr=[init_lr],
        max_lr=[max_lr],
        final_lr=[final_lr],
    )
    criterion = mpnn.utils.get_loss_func(dataset_type, uncertainty)
    metric = {
        "mse": lambda X, Y: F.mse_loss(X, Y, reduction="none"),
        "rmse": lambda X, Y: torch.sqrt(F.mse_loss(X, Y, reduction="none")),
    }[metric]

    with trange(
        max_epochs, desc="Training", unit="epoch", dynamic_ncols=True, leave=True
    ) as bar:
        for _ in bar:
            train_res = train_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                device,
                uncertainty,
            )
            val_res = validate_epoch(val_loader, model, metric, device, uncertainty)

            train_loss = train_res["loss"]
            val_loss = val_res["loss"]

            bar.set_postfix_str(
                f"train_loss={train_loss:0.3f} | val_loss={val_loss:0.3f} "
            )

    return model.module.to("cpu")
