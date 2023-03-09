import os

import torch
from tqdm import tqdm

from utils.utils import get_lr

from nets.deeplabv3_training import (CE_Loss, Dice_loss, Focal_Loss,
                                     weights_init)

from utils_seg.utils import get_lr
from utils_seg.utils_metrics import f_score

from utils.multitaskloss import MultiTaskLossWrapper


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, loss_history_seg, eval_callback, eval_callback_seg, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, dice_loss, focal_loss, cls_weights, num_class_seg, local_rank=0):
    total_loss_det = 0
    total_loss_seg = 0
    total_f_score = 0

    val_loss_det = 0
    val_loss_seg = 0
    val_f_score = 0

    total_loss = 0
    val_total_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets, radars, pngs, seg_labels = batch[0], batch[1], batch[2], batch[3], batch[4]

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                radars = radars.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                seg_labels = seg_labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs, outputs_seg = model_train(images, radars)

            if focal_loss:
                loss_seg = Focal_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)
            else:
                loss_seg = CE_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)

            if dice_loss:
                main_dice = Dice_loss(outputs_seg, seg_labels)
                loss_seg = loss_seg + main_dice

            # ----------------------#
            #   计算损失
            # ----------------------#
            loss_det = yolo_loss(outputs, targets)

            mtl = MultiTaskLossWrapper(task_num=2)
            total_loss = mtl(loss_seg, loss_det)

            with torch.no_grad():
                train_f_score = f_score(outputs_seg, seg_labels)

            # ----------------------#
            #   反向传播
            # ----------------------#
            total_loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs, outputs_seg = model_train(images, radars)

                if focal_loss:
                    loss_seg = Focal_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)
                else:
                    loss_seg = CE_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)

                if dice_loss:
                    main_dice = Dice_loss(outputs_seg, seg_labels)
                    loss_seg = loss_seg + main_dice

                # ----------------------#
                #   calculate loss
                # ----------------------#
                loss_det = yolo_loss(outputs, targets)

                mtl = MultiTaskLossWrapper(task_num=2)
                total_loss = mtl(loss_seg, loss_det)

                with torch.no_grad():
                    train_f_score = f_score(outputs_seg, seg_labels)

            # ----------------------#
            #   back-propagation
            # ----------------------#
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        total_loss_det += loss_det.item()
        total_loss_seg += loss_seg.item()
        total_loss += total_loss_det + total_loss_seg
        total_f_score += train_f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'detection loss': total_loss_det / (iteration + 1),
                                'segmentation loss': total_loss_seg / (iteration + 1),
                                'total loss': total_loss / (iteration + 1),
                                'f score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets, radars, pngs, seg_labels = batch[0], batch[1], batch[2], batch[3], batch[4]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                radars = radars.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                seg_labels = seg_labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs, outputs_seg = model_train(images, radars)

            if focal_loss:
                loss_seg = Focal_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)
            else:
                loss_seg = CE_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)

            if dice_loss:
                main_dice = Dice_loss(outputs_seg, seg_labels)
                loss_seg = loss_seg + main_dice

            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs_seg, seg_labels)

            # ----------------------#
            #   计算损失
            # ----------------------#
            loss_value = yolo_loss(outputs, targets)
            loss_value_seg = loss_seg
            val_f_score += _f_score.item()

        val_loss_det += loss_value.item()
        val_loss_seg += loss_value_seg.item()
        val_total_loss = val_loss_det + val_loss_seg

        if local_rank == 0:
            pbar.set_postfix(**{'detection val_loss': val_loss_det / (iteration + 1),
                                'segmentation val_loss': val_loss_seg / (iteration + 1),
                                'val loss': val_total_loss / (iteration + 1),
                                'f_score': val_f_score / (iteration + 1),
                                })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss_det / epoch_step, val_loss_det / epoch_step_val)
        loss_history_seg.append_loss(epoch + 1, total_loss_seg / epoch_step, val_loss_seg / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        eval_callback_seg.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss Det: %.3f  || Val Loss Seg: %.3f' % ((total_loss / epoch_step,
                                                                                  val_loss_det / epoch_step_val,
                                                                                 val_loss_seg / epoch_step_val)))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-det_val_loss%.3f-seg_val_loss%.3f.pth" % (
            epoch + 1, val_total_loss / epoch_step, val_loss_det / epoch_step_val, val_loss_seg / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_total_loss / epoch_step_val) <= min(loss_history.val_loss) + min(loss_history_seg.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))