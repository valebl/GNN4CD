import torch
import torch.nn as nn
import numpy as np
import time
from utils.metrics import AverageMeter, accuracy_binary_one, accuracy_binary_one_classes
from utils.tools import write_log

#-----------------------------------------------------
#---------------------- TRAIN ------------------------
#-----------------------------------------------------


class Trainer(object):

    def __init__(self):
        super(Trainer, self).__init__()

    def _create_plots_C(self, y_pred, y, t, train_mask, graph, accelerator, step, args):
        pass

    def _create_plots_R_Rall(self, y_pred, y, t, train_mask, graph, accelerator, step, args):
        pass

    #--- CLASSIFIER (C)
    def train_C(self, model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args,
                        epoch_start, alpha=0.75, gamma=2, log_freq=5):
        
        write_log(f"\nStart training the classifier.", args, accelerator, 'a')

        step = 0
        
        for epoch in range(epoch_start, epoch_start+args.epochs):
            
            model.train()
            write_log(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}", args, accelerator, 'a')

            # Define objects to track meters durng training
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            acc_class0_meter = AverageMeter()
            acc_class1_meter = AverageMeter()

            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()
            val_acc_class0_meter = AverageMeter()
            val_acc_class1_meter = AverageMeter()

            start = time.time()

            for graph in dataloader_train:
                
                optimizer.zero_grad()             
                y_pred = model(graph).squeeze()

                train_mask = graph["high"].train_mask      
                y = graph['high'].y

                # Apply mask
                y_pred, y = y_pred[train_mask], y[train_mask]

                loss = loss_fn(y_pred, y, alpha, gamma, reduction='mean')
                
                accelerator.backward(loss)
                optimizer.step()
                step += 1
                              
                acc = accuracy_binary_one(y_pred, y)
                acc_class0, acc_class1 = accuracy_binary_one_classes(y_pred, y)

                loss_meter.update(val=loss.item(), n=y_pred.shape[0])
                acc_meter.update(val=acc.item(), n=y_pred.shape[0])
                acc_class0_meter.update(val=acc_class0.item(), n=(y==0).sum().item())
                acc_class1_meter.update(val=acc_class1.item(), n=(y==1).sum().item())

                accelerator.log({'epoch':epoch, 'accuracy iteration': acc_meter.val, 'loss avg': loss_meter.avg,
                                 'accuracy avg': acc_meter.avg,'accuracy class0 avg': acc_class0_meter.avg, 'accuracy class1 avg': acc_class1_meter.avg}, step=step)
                
            end = time.time()

            # End of epoch --> write log and save checkpoint
            accelerator.log({'epoch':epoch, 'loss epoch': loss_meter.avg,'accuracy epoch': acc_meter.avg,
                             'accuracy class0 epoch': acc_class0_meter.avg, 'accuracy class1 epoch': acc_class1_meter.avg}, step=step)
            
            write_log(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}; "
                      + f"acc: {acc_meter.avg:.4f}; acc class 0: {acc_class0_meter.avg:.4f}; acc class 1: {acc_class1_meter.avg:.4f}.", args, accelerator, 'a')

            accelerator.save_state(output_dir=args.output_path+f"checkpoint_{epoch}/", safe_serialization=False)
            torch.save({"epoch": epoch}, args.output_path+f"checkpoint_{epoch}/epoch")

            # VALIDATION
            model.eval()

            if epoch%log_freq==0:
                y_pred_list = []
                y_list = []
                train_mask_list = []
                t_list = []
                
            with torch.no_grad():
                for graph in dataloader_val:

                    y_pred = model(graph).squeeze()
                    train_mask = graph['high'].train_mask
                    y = graph['high'].y

                    # Validation metrics for 1GPU
                    loss = loss_fn(y_pred[train_mask], y[train_mask], alpha, gamma, reduction="mean")

                    acc_class0, acc_class1 = accuracy_binary_one_classes(y_pred, y)
                    acc = accuracy_binary_one(y_pred, y)

                    # Update AverageMeter
                    val_loss_meter.update(val=loss.item(), n=y_pred.shape[0])
                    val_acc_meter.update(val=acc.item(), n=y_pred.shape[0])
                    val_acc_class0_meter.update(val=acc_class0.item(), n=(y==0).sum().item())
                    val_acc_class1_meter.update(val=acc_class1.item(), n=(y==1).sum().item())

                    accelerator.log({'epoch':epoch, 'val loss iteration': val_loss_meter.val, 'val loss avg': val_loss_meter.avg,
                                     'lr': np.mean(lr_scheduler.get_last_lr())}, step=step)
                
                    if epoch%5==0:
                        # Gather from all processes for metrics
                        t = graph.t
                        y_pred, y, train_mask, t = accelerator.gather((
                            y_pred.unsqueeze(0), y.unsqueeze(0), train_mask.unsqueeze(0), t))
                
                        # nodes, time
                        y_pred_list.append(torch.atleast_2d(y_pred)) # time, nodes
                        y_list.append(torch.atleast_2d(y))
                        train_mask_list.append(torch.atleast_2d(train_mask))
                        t_list.append(torch.atleast_2d(t))

                ###### PLOTS ######
                # TODO -> implement this function
                if epoch%5==0:
                    t = torch.cat(t_list, dim=1).squeeze()
                    y_pred = torch.cat(y_pred_list, dim=0).swapaxes(0,1)
                    y = torch.cat(y_list, dim=0).swapaxes(0,1)
                    train_mask = torch.cat(train_mask_list, dim=0).swapaxes(0,1)
                    self._create_plots_C(y_pred, y, t, train_mask, graph, accelerator, step, args)

            if lr_scheduler is not None:
                lr_scheduler.step()
           
            accelerator.log({'epoch':epoch, 'val loss avg': val_loss_meter.avg,
                             'val accuracy': val_acc_meter.avg,
                             'val accuracy class0': val_acc_class0_meter.avg,
                             'val accuracy class1': val_acc_class1_meter.avg
                             }, step=step)
                
    #--- REGRESSOR (either R or Rall)
    def train_R_Rall(self, model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=0, log_freq=5):
        
        write_log(f"\nStart training the regressor.", args, accelerator, 'a')

        step = 0

        MSELoss = nn.MSELoss()
        
        for epoch in range(epoch_start, epoch_start+args.epochs):

            model.train()
            write_log(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}", args, accelerator, 'a')
            
            # Define objects to track meters
            loss_meter = AverageMeter()            
            loss_term1_meter = AverageMeter()
            loss_term2_meter = AverageMeter()

            val_loss_meter = AverageMeter()
            val_loss_term1_meter = AverageMeter()
            val_loss_term2_meter = AverageMeter()

            start = time.time()
            
            # TRAIN
            for graph in dataloader_train:

                optimizer.zero_grad()
                y_pred = model(graph).squeeze()

                train_mask = graph['high'].train_mask
                y = graph['high'].y

                # Apply mask
                y_pred, y = y_pred[train_mask], y[train_mask]

                w = graph['high'].w
                w = w[train_mask]

                loss_mse = MSELoss(y_pred, y)
                loss_qmse = loss_fn(y_pred, y, w)
                loss = loss_mse + args.alpha * loss_qmse
                
                accelerator.backward(loss)
                optimizer.step()
                step += 1
                
                loss_meter.update(val=loss.item(), n=y_pred.shape[0])
                loss_term1_meter.update(val=loss_mse.item(), n=y_pred.shape[0])
                loss_term2_meter.update(val=loss_qmse.item(), n=y_pred.shape[0])
                
                accelerator.log({'epoch':epoch, 'train loss iteration': loss_meter.val, 'train loss avg': loss_meter.avg,
                                'train mse loss avg': loss_term1_meter.avg, 'train quantized loss avg': loss_term2_meter.avg
                                }, step=step)
            end = time.time()

            accelerator.log({'epoch':epoch, 'train loss avg': loss_meter.avg,
                                'train mse loss avg': loss_term1_meter.avg, 'train quantized loss avg': loss_term2_meter.avg,
                                'lr': np.mean(lr_scheduler.get_last_lr())
                                }, step=step)
            
            write_log(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds." +
                      f"Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}. ", args, accelerator, 'a')
                    
            accelerator.save_state(output_dir=args.output_path+f"checkpoint_{epoch}/", safe_serialization=False)
            torch.save({"epoch": epoch}, args.output_path+f"checkpoint_{epoch}/epoch")

            # VALIDATION
            # Validation is performed on all the validation dataset at once
            model.eval()

            if epoch%log_freq==0:
                y_pred_list = []
                y_list = []
                train_mask_list = []
                t_list = []

            with torch.no_grad():    
                for graph in dataloader_val:
                    
                    y_pred = model(graph).squeeze()
                    train_mask = graph['high'].train_mask
                    y = graph['high'].y
                    w = graph['high'].w
                    loss_mse = MSELoss(y_pred[train_mask].squeeze(), y[train_mask])
                    loss_qmse = loss_fn(y_pred[train_mask].squeeze(), y[train_mask], w[train_mask])
                    loss = loss_mse + args.alpha * loss_qmse
                    
                    val_loss_meter.update(val=loss.item(), n=y_pred.shape[0])
                    val_loss_term1_meter.update(val=loss_mse.item(), n=y_pred.shape[0])
                    val_loss_term2_meter.update(val=loss_qmse.item(), n=y_pred.shape[0])

                    accelerator.log({'epoch':epoch, 'val loss iteration': val_loss_meter.val, 'val loss avg': val_loss_meter.avg
                        }, step=step)
                    
                    if epoch%5==0:
                        # Gather from all processes for metrics
                        t = graph.t
                        y_pred, y, train_mask, t = accelerator.gather((
                            y_pred.unsqueeze(0), y.unsqueeze(0), train_mask.unsqueeze(0), t))

                        # nodes, time
                        y_pred_list.append(torch.atleast_2d(y_pred)) # time, nodes
                        y_list.append(torch.atleast_2d(y))
                        train_mask_list.append(torch.atleast_2d(train_mask))
                        t_list.append(torch.atleast_2d(t))

                ###### PLOTS ######
                # TODO -> Implement function to plot
                if epoch%5==0:
                    t = torch.cat(t_list, dim=1).squeeze()
                    y_pred = torch.cat(y_pred_list, dim=0).swapaxes(0,1)
                    y = torch.cat(y_list, dim=0).swapaxes(0,1)
                    train_mask = torch.cat(train_mask_list, dim=0).swapaxes(0,1)
                    self._create_plots_R_Rall(y_pred, y, t, train_mask, graph, accelerator, step, args)

            if "quantized_loss" in args.loss_fn:
                accelerator.log({'epoch':epoch, 'val loss avg': val_loss_meter.avg,
                                 'val mse loss avg': val_loss_term1_meter.avg, 'val qmse loss avg': val_loss_term2_meter.avg
                                }, step=step)
            else:
                accelerator.log({'epoch':epoch, 'val loss avg': val_loss_meter.avg,
                                }, step=step)
                    
            if lr_scheduler is not None:
                lr_scheduler.step()

#-----------------------------------------------------
#----------------------- TEST ------------------------
#-----------------------------------------------------


class Tester(object):

    def test(self, model, dataloader, args, accelerator=None):
        model.eval()
        step = 0 

        pr = []
        times = []
        with torch.no_grad():    
            for graph in dataloader:

                t = graph.t
                times.append(t)
                
                # Regressor
                y_pred = model(graph)
                if args.model_type == "R" or args.model_type == "Rall":
                    y_pred = torch.where(torch.isfinite(torch.expm1(y_pred)), torch.expm1(y_pred), np.nan)
                elif args.model_type == "C":
                    y_pred = torch.where(y_pred < 0, 1, 0)
                pr.append(y_pred)
                
                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        pr = torch.stack(pr)
        times = torch.stack(times)

        return pr, times

    def test_RC(self, model_R, model_C, dataloader, args, accelerator=None):
        model_R.eval()
        model_C.eval()
        step = 0 

        pr_R = []
        pr_C = []
        times = []
        with torch.no_grad():    
            for graph in dataloader:

                t = graph.t
                times.append(t)
                
                # Regressor
                y_pred_R = model_R(graph)
                y_pred_R = torch.where(torch.isfinite(torch.expm1(y_pred_R)), torch.expm1(y_pred_R), np.nan)
                pr_R.append(y_pred_R)
                
                y_pred_C = model_C(graph)
                y_pred_C = torch.where(y_pred_C < 0, 1, 0)
                pr_C.append(y_pred_C)
                
                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        pr_R = torch.stack(pr_R)
        pr_C = torch.stack(pr_C)
        times = torch.stack(times)

        return pr_R, pr_C, times
