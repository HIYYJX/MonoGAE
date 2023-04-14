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
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 loss,
                 model_name):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
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
     

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            resume_model_path = os.path.join(self.output_dir, "checkpoint.pth")
            assert os.path.exists(resume_model_path)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(
                model=self.model.to(self.device),
                optimizer=self.optimizer,
                filename=resume_model_path,
                map_location=self.device,
                logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info("Loading Checkpoint... Best Result:{}, Best Epoch:{}".format(self.best_result, self.best_epoch))
        
    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        best_result = self.best_result
        best_epoch = self.best_epoch
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
                    
       
            self.train_one_epoch(epoch)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                if self.cfg['save_all']:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint_epoch_%d' % self.epoch)
                else:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint')
                save_checkpoint(
                    get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),
                    ckpt_name)

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

            progress_bar.update()

        self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

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
        
    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.model.train()
        print(">>>>>>> Epoch:", str(epoch) + ":")

        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, (inputs, calibs, targets, info,random_flip_flag) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device) # 8 3 x 4
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)
            img_sizes = targets['img_size']
            depthmaps = targets['depthmaps'].to(self.device)  # 8,58,32,1
            gt_denorms = targets['denorms'].to(self.device) # 8,58,32,4
            ori_denorms = targets['ori_denorms'].to(self.device) # 8,58,32,4
            random_flip_flag = random_flip_flag.to(self.device) # 8,1
            #print(random_flip_flag)
            
            '''
            outputs_coord = targets[ 'boxes_3d']
         
            outputs_depth = targets[ 'depth']
            #print(outputs_coord.shape) torch.Size([1, 50, 6])
            size = np.array([928,512])
            size = torch.tensor(size).to(self.device)
            pad = np.array([28,11])
            pad =torch.tensor(pad).to(self.device)
            pad2 = np.array([2,1])
            pad2 =torch.tensor(pad2).to(self.device)
            outputs_coord[..., :2] = (outputs_coord[..., :2]*size-pad)/16+pad2
            batch = outputs_coord.shape[0]
            depths = []
            for b in range(batch):
                weighted_depth = depthmaps[b]  
                weighted_depth = weighted_depth.squeeze(-1)
                weighted_depth = weighted_depth.transpose(1,0)
                #print(weighted_depth.shape) #torch.Size([32,58])
                coord = outputs_coord[b] 
                #print(coord[:,:2])
                #print(coord.shape)#torch.Size( 50, 6])
                depth_map = self.softget(weighted_depth,coord)
                depth_map = depth_map.unsqueeze(-1)
                #print(depth_map)
                #print(depth_map.shape)torch.Size([50])
                depths.append(depth_map)
            depths = torch.stack(depths) 
            #print(depths.shape) torch.Size([8, 50, 1])
             
             
   
                
            #outputs_coord[..., :2] = outputs_coord[..., :2]-pad
            #outputs_coord[..., :2] = outputs_coord[..., :2]/16
            #outputs_coord[..., :2] = outputs_coord[..., :2]+pad2
            mapsize = np.array([58,32])
            mapsize =torch.tensor(mapsize).to(self.device)
            #print(outputs_coord[..., :2][1])
            outputs_coord[..., :2] = outputs_coord[..., :2]/mapsize
            #print(outputs_coord[..., :2].shape)torch.Size([8, 50, 2])
            #print(outputs_depth)
            outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2).detach()
            #print(outputs_center3d.shape) torch.Size([1, 50, 1, 2]) 
        
            depthmap = depthmaps.squeeze(-1)
            depthmap = depthmap.transpose(2,1)
            depth_map = F.grid_sample(
                depthmap.unsqueeze(1),
                outputs_center3d,
                mode='bilinear',
                align_corners=True).squeeze(1)
             
            #print(depth_map.shape)torch.Size([1, 50, 1])
            #print(depth_map)
            ''' 
            
            targets = self.prepare_targets(targets, inputs.shape[0])
        
            # train one batch
            self.optimizer.zero_grad()
            #print(inputs.shape)
            outputs = self.model(inputs, calibs, targets, img_sizes,ori_denorms,random_flip_flag)
            #print(outputs['pred_logits'].shape)
            detr_losses_dict = self.detr_loss(outputs, targets,depthmaps,gt_denorms) # depthmaps 8,58,32,1  gt_denorms 8,58,32,4

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
            if batch_idx % 30 == 0:
                print("----", batch_idx, "----")
                print("%s: %.2f, " %("loss_detr", detr_losses_dict_log["loss_detr"]))
                for key, val in detr_losses_dict_log.items():
                    if key == "loss_detr":
                        continue
                    if "0" in key or "1" in key or "2" in key or "3" in key or "4" in key or "5" in key:
                        if flags[int(key[-1])]:
                            print("")
                            flags[int(key[-1])] = False
                    print("%s: %.2f, " %(key, val), end="")
                print("")
                print("")

            detr_losses.backward()
            self.optimizer.step()

            progress_bar.update()
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
        #targets_list.append(targets['depthmaps'])  
        #print(targets['depthmaps'].shape)   torch.Size([8, 58, 32, 1])
            
        return targets_list

