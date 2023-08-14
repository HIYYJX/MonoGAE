import os
import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint

from utils import misc


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 loss,
                 model_name):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detr_loss = loss
        self.model_name = model_name
        self.output_dir = os.path.join('./' + cfg['save_path'], model_name)
        self.tester = None
        
    def train(self, train_loader, test_loader, train_sampler, test_sampler, distributed, local_rank):
        start_epoch = self.epoch
        best_result = self.best_result
        best_epoch = self.best_epoch
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            np.random.seed(np.random.get_state()[1][0] + epoch)
            if distributed:
                train_sampler.set_epoch(epoch)
                test_sampler.set_epoch(epoch)
       
            self.train_one_epoch(epoch, train_loader, local_rank)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0 and local_rank == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                if self.cfg['save_all']:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint_epoch_%d' % self.epoch)
                else:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint')
                save_checkpoint(
                    get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),
                    ckpt_name)

                '''
                if self.tester is not None:
                    self.logger.info("Test Epoch {}".format(self.epoch))
                    self.tester.inference()
                    cur_result = self.tester.evaluate()
                    if cur_result > best_result:
                        best_result = cur_result
                        best_epoch = self.epoch
                        ckpt_name = os.path.join(self.output_dir, 'checkpoint_best')
                        save_checkpoint(
                            get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),
                            ckpt_name)
                    self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))
                '''
        # self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

        return None
    
    def softget(self,weighted_depth,outputs_coord):
        h,w = weighted_depth.shape[-2], weighted_depth.shape[-1]
        pts_x = outputs_coord[:,0]  
        pts_y = outputs_coord[:,1]
        
        pts_x_low = pts_x.floor().long()
        pts_x_high = pts_x.ceil().long() 
        pts_y_low = pts_y.floor().long() 
        pts_y_high = pts_y.ceil().long()
        rop_lt = weighted_depth[..., pts_y_low, pts_x_low]
        rop_rt = weighted_depth[..., pts_y_low, pts_x_low]
        rop_ld = weighted_depth[..., pts_y_high, pts_x_low]
        rop_rd = weighted_depth[..., pts_y_high, pts_x_high]
        
        rop_t = (1 - pts_x + pts_x_low) * rop_lt + (1 - pts_x_high + pts_x) * rop_rt
        rop_d = (1 - pts_x + pts_x_low) * rop_ld + (1 - pts_x_high + pts_x) * rop_rd
        rop = (1 - pts_y + pts_y_low) * rop_t + (1 - pts_y_high + pts_y) * rop_d
        return rop
        
    def to_local_rank(inputs, local_rank):
        if isinstance(inputs, list):
            return [to_local_rank(x, local_rank) for x in inputs]
        elif isinstance(inputs, dict):
            return {k: to_local_rank(v, local_rank) for k, v in inputs.items()}
        else:
            if isinstance(inputs, int) or isinstance(inputs, float) \
                    or isinstance(inputs, str):
                return inputs
            return inputs.cuda(local_rank, non_blocking=True)

    def train_one_epoch(self, epoch, train_loader, local_rank):
        torch.set_grad_enabled(True)
        if local_rank == 0:
            print(">>>>>>> Epoch:", str(epoch) + ":")
        progress_bar = tqdm.tqdm(total=len(train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, (inputs, calibs, targets, info,random_flip_flag) in enumerate(train_loader):
            inputs = inputs.cuda(local_rank, non_blocking=True)
            calibs = calibs.cuda(local_rank, non_blocking=True)
            for key in targets.keys():
                targets[key] = targets[key].cuda(local_rank, non_blocking=True)
            img_sizes = targets['img_size']
            depthmaps = targets['depthmaps'].cuda(local_rank, non_blocking=True)
            gt_denorms = targets['denorms'].cuda(local_rank, non_blocking=True)
            ori_denorms = targets['ori_denorms'].cuda(local_rank, non_blocking=True)
            random_flip_flag = random_flip_flag.cuda(local_rank, non_blocking=True)    
            targets = self.prepare_targets(targets, inputs.shape[0])
        
            self.model.train()
            self.model.zero_grad()
            self.optimizer.zero_grad()
            outputs = self.model(inputs, calibs, targets, img_sizes, ori_denorms, random_flip_flag)
            detr_losses_dict = self.detr_loss(outputs, targets,depthmaps, gt_denorms) # depthmaps 8,58,32,1  gt_denorms 8,58,32,4

            weight_dict = self.detr_loss.weight_dict
            detr_losses_dict_weighted = [detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict.keys() if k in weight_dict]
            detr_losses = sum(detr_losses_dict_weighted)

            detr_losses_dict = misc.reduce_dict(detr_losses_dict)
            detr_losses_dict_log = {}
            detr_losses_log = 0
            for k in detr_losses_dict.keys():
                if k in weight_dict:
                    detr_losses_dict_log[k] = (detr_losses_dict[k] * weight_dict[k]).item()
                    detr_losses_log += detr_losses_dict_log[k]
            detr_losses_dict_log["loss_detr"] = detr_losses_log

            flags = [True] * 5
            if batch_idx % 30 == 0 and local_rank == 0:
                print("%s: %.2f, " %("loss_detr", detr_losses_dict_log["loss_detr"]))
                for key, val in detr_losses_dict_log.items():
                    if key == "loss_detr":
                        continue
                    if "0" in key or "1" in key or "2" in key or "3" in key or "4" in key or "5" in key:
                        if flags[int(key[-1])]:
                            print("")
                            flags[int(key[-1])] = False
                    print("%s: %.2f, " %(key, val), end="")

            torch.distributed.barrier()
            detr_losses.backward()
            self.optimizer.step()
            progress_bar.update()
            torch.cuda.empty_cache()
        progress_bar.close()

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']

        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
            targets_list.append(target_dict)
            
        return targets_list

