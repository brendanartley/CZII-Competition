import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from monai.inferers import sliding_window_inference

import pandas as pd
import numpy as np
from tqdm import tqdm
import gc

from src.data.utils import (
    get_dataset,
    get_dataloader,
)

from src.models.utils import (
    get_model,
    ModelEMA,
)

from src.modules.utils import (
    get_optimizer,
    get_scheduler,
    batch_to_device,
    calc_grad_norm,
    flatten_dict,
    save_weights,
)

from src.logging.utils import get_logger
from src.modules.metrics import score
from src.modules.ccl import mask2centroids

def score_predictions(cfg, all_preds, val_ds, val_metrics):
    # Mask -> Centroids
    submission= []
    for i, idx in enumerate(val_ds.idxs):
        submission.append(
            mask2centroids(
                all_preds[i],
                experiment= idx, 
                log= True,
                )
            )
    if len(submission) == 0:
        return val_metrics
    submission= pd.concat(submission)

    # Score
    solution= pd.read_csv("./data/raw/labels.csv")
    solution= solution[solution["experiment"].isin(val_ds.idxs)]

    metric_score= score(
        solution= solution,
        submission= submission,
    )
    val_metrics["val"]= val_metrics["val"] | metric_score

    return val_metrics

def run_eval(model, val_ds, val_dl, val_metrics, cfg):
    model.eval()

    progress_bar = tqdm(range(len(val_dl)), disable=cfg.no_tqdm)
    val_itr = iter(val_dl)
    val_acc= 0
    val_loss= None
    val_outputs= {
        "logits": []
    }
    i= 0
    all_preds= []

    with torch.no_grad():
        
        # Only predict on center
        pz, py, px= cfg.patch_size
        roi_weight_map= torch.zeros((pz, py, px)).cuda()
        roi_weight_map[(pz//4):-(pz//4), (py//4):-(py//4), (px//4):-(px//4)]= 1.0

        for itr in progress_bar:
            i+=1
            data= next(val_itr)
            batch = batch_to_device(data, cfg.device)

            with autocast(cfg.device.type):
                
                # Pad input for roi_weight_map
                batch["input"]= F.pad(batch["input"], (32, 32, 32, 32, 8, 8))

                # Sliding window
                preds = sliding_window_inference(
                    inputs= batch["input"],
                    roi_size= cfg.patch_size,
                    predictor= model,
                    roi_weight_map= roi_weight_map,
                    **vars(cfg.infer_cfg)
                )
                preds= preds[:, :, 8:-8, 32:-32, 32:-32]

                # Loss
                val_metrics["val"]["loss"]= model.loss_fn(
                    input= preds,
                    target= batch["target"],
                ).item()

                # Activation + CPU
                preds= F.softmax(preds, dim=1)
                preds= preds[0].cpu().numpy()
                all_preds.append(preds)

        # Score preds
        val_metrics= score_predictions(cfg, all_preds, val_ds, val_metrics)

    return val_metrics

def train(cfg):

    # Dev run
    if cfg.fast_dev_run:
        cfg.epochs= 1

    # Load logger
    logger= get_logger(cfg)

    # Data
    print("-"*10)
    print("Dataset: {}".format(cfg.dataset_type))
    train_ds= get_dataset(cfg, mode="train")
    train_dl= get_dataloader(train_ds, cfg, mode="train")

    val_ds= get_dataset(cfg, mode="val")
    val_dl= get_dataloader(val_ds, cfg, mode="val")

    # Model/optimizer/scheduler
    model, emb_dim = get_model(cfg)
    model.to(cfg.device)

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg, n_steps=len(train_dl)*cfg.epochs)

    if cfg.mixed_precision:
        scaler = GradScaler(cfg.device.type)
    else:
        scaler = None

    # Training Loop
    train_metrics= {"train": {}, "lr": None, "epoch": None}
    val_metrics= {"val": {}}
    i= 0
    epoch= 0
    total_grad_norm = None    
    total_grad_norm_after_clip = None
    ema_model = None
    optimizer.zero_grad()

    if cfg.train:
        for epoch in range(cfg.epochs):
            train_metrics["epoch"] = epoch

            losses= []
            grad_norms= []
            grad_norms_clipped= []
            gc.collect()

            progress_bar = tqdm(range(len(train_dl)), disable=cfg.no_tqdm)
            tr_itr = iter(train_dl)

            if cfg.ema == True and epoch == 0:
                print("Starting EMA..")
                ema_model= ModelEMA(model, decay=cfg.ema_decay)
            
            for itr in progress_bar:
                i += 1
                data= next(tr_itr)
                batch= batch_to_device(data, cfg.device)

                model= model.train()

                # Forward Pass
                if cfg.mixed_precision:
                    with autocast(cfg.device.type):
                        output= model(batch)
                else:
                    output= model(batch)
                loss= output["loss"]
                losses.append(loss.item())

                # Backward pass
                if cfg.mixed_precision:
                    scaler.scale(loss).backward()
                    if i % cfg.grad_accumulation == 0:
                        if (cfg.track_grad_norm) or (cfg.grad_clip > 0):
                            scaler.unscale_(optimizer)
                        if cfg.track_grad_norm:
                            total_grad_norm = calc_grad_norm(model.parameters(), cfg.grad_norm_type)
                            if total_grad_norm is not None:
                                grad_norms.append(total_grad_norm.item())
                        if cfg.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        if cfg.track_grad_norm:
                            total_grad_norm_after_clip = calc_grad_norm(model.parameters(), cfg.grad_norm_type)
                            if total_grad_norm_after_clip is not None:
                                grad_norms_clipped.append(total_grad_norm_after_clip.item())
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if i % cfg.grad_accumulation == 0:
                        if cfg.track_grad_norm:
                            total_grad_norm = calc_grad_norm(model.parameters())
                            if total_grad_norm is not None:
                                grad_norms.append(total_grad_norm.item())
                        if cfg.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        if cfg.track_grad_norm:
                            total_grad_norm_after_clip = calc_grad_norm(model.parameters(), cfg.grad_norm_type)
                            if total_grad_norm_after_clip is not None:
                                grad_norms_clipped.append(total_grad_norm_after_clip.item())
                        optimizer.step()
                        optimizer.zero_grad() 

                if ema_model is not None:
                    ema_model.update(model)

                if scheduler is not None:
                    scheduler.step()

                # Train Logging
                if cfg.local_rank == 0 and i % cfg.logging_steps == 0:
                    train_metrics["train"]["loss"]= np.mean(losses[-10:])
                    train_metrics["lr"]= cfg.opt_cfg.lr if scheduler is None else scheduler.get_last_lr()[0]

                    if cfg.track_grad_norm:
                        train_metrics["grad_norm"] = np.mean(grad_norms[-10:])
                        train_metrics["grad_norm_clipped"] = np.mean(grad_norms_clipped[-10:])
            
                    progress_bar.set_postfix(flatten_dict(train_metrics | val_metrics))
                    logger.log(train_metrics, commit=False)
                    
                if cfg.fast_dev_run:
                    break

            # Run eval
            if cfg.local_rank == 0 and epoch != cfg.epochs-1:
                if ema_model is not None:
                    val_metrics= run_eval(ema_model.module, val_ds, val_dl, val_metrics, cfg)
                else:
                    val_metrics= run_eval(model, val_ds, val_dl, val_metrics, cfg)

                progress_bar.set_postfix(flatten_dict(train_metrics | val_metrics))
                logger.log(val_metrics, commit=True)

            # Weights checkpoint
            if epoch % cfg.epochs_checkpoint == 0 and \
               epoch + 1 >= cfg.epochs_checkpoint_start and \
               epoch + 1 != cfg.epochs and \
               cfg.save_weights:
                if ema_model is not None:
                    save_weights(ema_model.module, cfg, epoch)
                else:
                    save_weights(model, cfg, epoch)

    # Weights checkpoint
    if cfg.save_weights:
        if ema_model is not None:
            save_weights(ema_model.module, cfg, epoch)
        else:
            save_weights(model, cfg, epoch)

    # Final eval
    if ema_model is not None:
        val_metrics= run_eval(ema_model.module, val_ds, val_dl, val_metrics, cfg)
    else:
        val_metrics= run_eval(model, val_ds, val_dl, val_metrics, cfg)
    logger.log(val_metrics, commit=True)
    print(val_metrics)

    logger.finish()
    return
