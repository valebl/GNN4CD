import torch
import numpy as np
import time
import wandb
from utils.metrics import AverageMeter, accuracy_binary_one, accuracy_binary_one_classes
from utils.tools import write_log
from utils.plots import create_zones, plot_maps, plot_pdf, plot_diurnal_cycles
import matplotlib.pyplot as plt


#-----------------------------------------------------
#---------------------- TRAIN ------------------------
#-----------------------------------------------------


class Trainer(object):

    def __init__(self):
        super(Trainer, self).__init__()

    #--- CLASSIFIER (C)
    def train_cl(self, model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args,
                        epoch_start, alpha=0.75, gamma=2):
        
        write_log(f"\nStart training the classifier.", args, accelerator, 'a')

        step = 0
        
        for epoch in range(epoch_start, epoch_start+args.epochs):
            
            model.train()
            write_log(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}", args, accelerator, 'a')

            # Define objects to track meters durng training
            all_loss_meter = AverageMeter()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            acc_class0_meter = AverageMeter()
            acc_class1_meter = AverageMeter()

            start = time.time()

            for graph in dataloader_train:
                
                optimizer.zero_grad()             
                y_pred = model(graph).squeeze()

                train_mask = graph["high"].train_mask      
                y = graph['high'].y   

                # Gather from all processes for metrics
                all_y_pred, all_y, all_train_mask = accelerator.gather((y_pred, y, train_mask))

                # Apply mask
                y_pred, y = y_pred[train_mask], y[train_mask]
                all_y_pred, all_y = all_y_pred[all_train_mask], all_y[all_train_mask]
                
                loss = loss_fn(y_pred, y, alpha, gamma, reduction='mean')
                all_loss = loss_fn(all_y_pred, all_y, alpha, gamma, reduction='mean')
                
                accelerator.backward(loss)
                optimizer.step()
                step += 1
                
                loss_meter.update(val=loss.item(), n=y_pred.shape[0])   
                all_loss_meter.update(val=all_loss.item(), n=all_y_pred.shape[0])   
                
                acc = accuracy_binary_one(all_y_pred, all_y)
                acc_class0, acc_class1 = accuracy_binary_one_classes(all_y_pred, all_y)

                acc_meter.update(val=acc.item(), n=all_y_pred.shape[0])
                acc_class0_meter.update(val=acc_class0.item(), n=(all_y==0).sum().item())
                acc_class1_meter.update(val=acc_class1.item(), n=(all_y==1).sum().item())

                accelerator.log({'epoch':epoch, 'accuracy iteration': acc_meter.val, 'loss avg': all_loss_meter.avg,
                                 'loss avg (1GPU)': loss_meter.avg, 'accuracy avg': acc_meter.avg,
                                 'accuracy class0 avg': acc_class0_meter.avg, 'accuracy class1 avg': acc_class1_meter.avg}, step=step)
                
            end = time.time()

            # End of epoch --> write log and save checkpoint
            accelerator.log({'epoch':epoch, 'loss epoch': all_loss_meter.avg, 'loss epoch (1GPU)': loss_meter.avg,  'accuracy epoch': acc_meter.avg,
                             'accuracy class0 epoch': acc_class0_meter.avg, 'accuracy class1 epoch': acc_class1_meter.avg}, step=step)
            write_log(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {all_loss_meter.sum:.4f} - average: {all_loss_meter.avg:.10f}; "
                      + f"acc: {acc_meter.avg:.4f}; acc class 0: {acc_class0_meter.avg:.4f}; acc class 1: {acc_class1_meter.avg:.4f}.", args, accelerator, 'a')

            accelerator.save_state(output_dir=args.output_path+f"checkpoint_{epoch}/", safe_serialization=False)
            torch.save({"epoch": epoch}, args.output_path+f"checkpoint_{epoch}/epoch")

            # Perform the validation step
            model.eval()

            y_pred_val = []
            y_val = []
            train_mask_val = []
            t = []
                
            with torch.no_grad():
                for graph in dataloader_val:
                    # Append the data for the current epoch
                    y_pred_val.extend(model(graph,inference=True)) # num_nodes, time
                    graph = graph.to_data_list()
                    [train_mask_val.append(graph_i["high"].train_mask) for graph_i in graph]
                    [y_val.append(graph_i['high'].y) for graph_i in graph]
                    [t.append(graph_i.t) for graph_i in graph]

                # Create tensors
                train_mask_val = torch.stack(train_mask_val, dim=-1).squeeze().swapaxes(0,1) # time, nodes
                y_pred_val = torch.stack(y_pred_val, dim=-1).squeeze().swapaxes(0,1)
                y_val = torch.stack(y_val, dim=-1).squeeze().swapaxes(0,1)
                t = torch.stack(t, dim=-1).squeeze()

                # Validation metrics for 1GPU
                loss_val_1gpu = loss_fn(y_pred_val[train_mask_val], y_val[train_mask_val], alpha, gamma, reduction="mean")

                # Gather from all processes for metrics
                y_pred_val, y_val, train_mask_val = accelerator.gather((y_pred_val, y_val, train_mask_val))

                # Apply mask
                y_pred_val, y_val = y_pred_val[train_mask_val], y_val[train_mask_val]

                # Compute metrics on all validation dataset            
                loss_val = loss_fn(y_pred_val, y_val, alpha, gamma, reduction="mean")

                acc_class0_val, acc_class1_val = accuracy_binary_one_classes(y_pred_val, y_val)
                acc_val = accuracy_binary_one(y_pred_val, y_val)
            
            if lr_scheduler is not None:
                lr_scheduler.step()
           
            accelerator.log({'epoch':epoch, 'validation loss': loss_val.item(), 'validation loss (1GPU)': loss_val_1gpu.item(),
                             'validation accuracy': acc_val.item(),
                             'validation accuracy class0': acc_class0_val.item(),
                             'validation accuracy class1': acc_class1_val.item(),
                             'lr': np.mean(lr_scheduler.get_last_lr())}, step=step)
                
    #--- REGRESSOR (either R or Rall)
    def train_reg(self, model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=0):
        
        write_log(f"\nStart training the regressor.", args, accelerator, 'a')

        step = 0
        
        for epoch in range(epoch_start, epoch_start+args.epochs):

            model.train()
            write_log(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}", args, accelerator, 'a')
            
            # Define objects to track meters
            loss_meter = AverageMeter()
            all_loss_meter = AverageMeter()
            
            loss_term1_meter = AverageMeter()
            loss_term2_meter = AverageMeter()

            start = time.time()
            
            # TRAIN
            for i, graph in enumerate(dataloader_train):

                optimizer.zero_grad()
                y_pred = model(graph).squeeze()

                train_mask = graph['high'].train_mask
                y = graph['high'].y

                # Gather from all processes for metrics
                all_y_pred, all_y, all_train_mask = accelerator.gather((y_pred, y, train_mask))

                # Apply mask
                y_pred, y = y_pred[train_mask], y[train_mask]
                all_y_pred, all_y = all_y_pred[all_train_mask], all_y[all_train_mask]

                w = graph['high'].w
                all_w =accelerator.gather((w))
                w = w[train_mask]
                all_w = all_w[all_train_mask]
                

                loss, _, _ = loss_fn(y_pred, y, w)
                all_loss, loss_term1, loss_term2 = loss_fn(all_y_pred, all_y, all_w)
                
                accelerator.backward(loss)
                optimizer.step()
                step += 1
                
                # Log values to wandb
                loss_meter.update(val=loss.item(), n=y_pred.shape[0])    
                all_loss_meter.update(val=all_loss.item(), n=all_y_pred.shape[0])
                
                loss_term1_meter.update(val=loss_term1.item(), n=all_y_pred.shape[0])
                loss_term2_meter.update(val=loss_term2.item(), n=all_y_pred.shape[0])
                    
                accelerator.log({'epoch':epoch, 'loss iteration': loss_meter.val, 'loss avg': loss_meter.avg, 'loss all avg': all_loss_meter.avg}, step=step)

            end = time.time()

            accelerator.log({'epoch':epoch, 'train loss (1GPU)': loss_meter.avg, 'train loss': all_loss_meter.avg,
                                'train mse loss': loss_term1_meter.avg, 'train quantized loss': loss_term2_meter.avg}, step=step)

            write_log(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds." +
                      f"Loss - total: {all_loss_meter.sum:.4f} - average: {all_loss_meter.avg:.10f}. ", args, accelerator, 'a')
                    
            accelerator.save_state(output_dir=args.output_path+f"checkpoint_{epoch}/", safe_serialization=False)
            torch.save({"epoch": epoch}, args.output_path+f"checkpoint_{epoch}/epoch")

            # VALIDATION
            # Validation is performed on all the validation dataset at once
            model.eval()

            y_pred_val = []
            y_val = []
            train_mask_val = []
            t = []

            w_val = []

            with torch.no_grad():    
                for graph in dataloader_val:
                    # Append the data for the current epoch
                    y_pred_val.extend(model(graph,inference=True)) # num_nodes, time
                    graph = graph.to_data_list()
                    [train_mask_val.append(graph_i["high"].train_mask) for graph_i in graph]
                    [y_val.append(graph_i['high'].y) for graph_i in graph]
                    [t.append(graph_i.t) for graph_i in graph]
                    [w_val.append(graph_i['high'].w) for graph_i in graph]

                # Create tensors
                train_mask_val = torch.stack(train_mask_val, dim=-1).squeeze().swapaxes(0,1) # time, nodes
                y_pred_val = torch.stack(y_pred_val, dim=-1).squeeze().swapaxes(0,1)
                y_val = torch.stack(y_val, dim=-1).squeeze().swapaxes(0,1)
                t = torch.stack(t, dim=-1).squeeze()
                w_val = torch.stack(w_val, dim=-1).squeeze().swapaxes(0,1)

                # Log validation metrics for 1GPU
                loss_val_1gpu,  _, _ = loss_fn(y_pred_val.flatten()[train_mask_val.flatten()],
                                                y_val.flatten()[train_mask_val.flatten()],
                                                w_val.flatten()[train_mask_val.flatten()])

                # Gather from all processes for metrics
                y_pred_val, y_val, train_mask_val, t = accelerator.gather((y_pred_val, y_val, train_mask_val, t))

                # nodes, time
                y_pred_val, y_val, train_mask_val = y_pred_val.swapaxes(0,1), y_val.swapaxes(0,1), train_mask_val.swapaxes(0,1) # nodes, time

                w_val = accelerator.gather((w_val))
                w_val = w_val.swapaxes(0,1)
                
                # Apply mask
                y_pred_val, y_val = y_pred_val[train_mask_val], y_val[train_mask_val]
                    
                w_val = w_val[train_mask_val]
                loss_val, loss_term1_val, loss_term2_val = loss_fn(y_pred_val.flatten(), y_val.flatten(), w_val.flatten())

            if lr_scheduler is not None:
                lr_scheduler.step()
            
            accelerator.log({'epoch':epoch, 'validation loss (1GPU)': loss_val_1gpu.item(), 'validation loss': loss_val.item(),
                                'validation mse loss': loss_term1_val.item(),'validation quantized loss': loss_term2_val.item(),
                                'lr': np.mean(lr_scheduler.get_last_lr())}, step=step)


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
                    y_pred = torch.expm1(y_pred)
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

    def test_RC(self, model_R, model_C dataloader, args, accelerator=None):
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
                y_pred_R = torch.expm1(y_pred_R)
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
