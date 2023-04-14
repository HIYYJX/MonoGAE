import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerEncoder, TransformerEncoderLayer
import numpy as np

class DepthPredictor(nn.Module):

    def __init__(self, model_cfg):
        """
        Initialize depth predictor and depth encoder
        Args:
            model_cfg [EasyDict]: Depth classification network config
        """
        super().__init__()
        depth_num_bins = int(model_cfg["num_depth_bins"]) #80 270
        depth_min = float(model_cfg["depth_min"])
        depth_max = float(model_cfg["depth_max"]) #60 200
        self.depth_max = depth_max
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)

        # Create modules
        d_model = model_cfg["hidden_dim"]
        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model))
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        self.upsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))

        self.depth_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU())

        self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1))
        self.offpre = nn.Conv2d(d_model, 4, kernel_size=(1, 1))
        depth_encoder_layer = TransformerEncoderLayer(
            d_model, nhead=8, dim_feedforward=256, dropout=0.1)

        self.depth_encoder = TransformerEncoder(depth_encoder_layer, 1)

        self.depth_pos_embed = nn.Embedding(int(self.depth_max)+1, 256)
    
    
    def get_GPdepthmap(self, P, denorms,GPdepthmap,flip,imagesize):
        for i in range(int(denorms.shape[0])): # 31
            for j in range(int(denorms.shape[1])): #55
                #denorm = denorms[i,j]
                w1 = P[0,0]/35.2
                w11 = P[0,2]/35.2
                w2 = P[1,1]/35.2
                w22 = P[1,2]/35.2
                d = denorms[i,j]
                if flip==True:
                    jj = imagesize[0]/35.2-j
                else:
                    jj =j
                W = torch.tensor([[w1,0,w11-jj],[0,w2,w22-i],[d[0],d[1],d[2]]],requires_grad=True)
               
                #print(W) #(3, 3)
                result = torch.tensor([0,0,-d[3]]).reshape(-1,1)
                W_inv = torch.inverse(W) 
                #print(W_inv.shape)
                #print(torch.mm(W_inv,W))
                #print(W_inv.requires_grad)
                vvxyz = torch.mm(W_inv,result)
                #print(vvxyz[0,2])
                GPdepthmap[i,j]=vvxyz[2,0]
              
        return GPdepthmap
    
    def GPdepthmapFunc(self, P,denorms,GPdepthmap, flip, imagesize): # 31,55,4
     
        w1 = P[0, 0] / 35.2
        w11 = P[0, 2] / 35.2
        w2 = P[1, 1] / 35.2
        w22 = P[1, 2] / 35.2
        denorms = denorms.permute(1,0,2)
        ii = torch.arange(denorms.shape[0])#[0,55]

        j = torch.arange(denorms.shape[1]) 
        if flip == True:
            ii = imagesize[0] / 35.2 - ii

        W = torch.ones((55, 31, 3, 3))
        W[:,:,:2,:2] = torch.tensor([[w1, 0], [0, w2]])
        #print((w11 - ii).shape)(55,)
        #print((w11 - ii)[:, np.newaxis].shape)(55, 1)
        #print((w22 - j)[np.newaxis, :].shape)(1,31)
        W[:, :, 0, 2] = (w11 - ii).unsqueeze(-1) # (55, 1)
        W[:, :, 1, 2] = (w22 - j).unsqueeze(0) # (1,31)
        W[:, :, 2, :] = denorms[:, :, :3]
        W_inv = torch.inverse(W) 
        result = torch.zeros((55, 31, 3, 3))
        result[:, :, 2, 0] = -1 * denorms[:, :, 3]

        vvxyz = torch.matmul(W_inv, result)
        GPdepthmap[:,:,0] = vvxyz[:,:,2,0]
        
        return GPdepthmap
    
    def forward(self, feature, mask, pos,ori_denorms,calibs,random_flip_flag,img_sizes):
       
        assert len(feature) == 4

        # foreground depth map
        src_16 = self.proj(feature[1])
        src_32 = self.upsample(F.interpolate(feature[2], size=src_16.shape[-2:]))
        src_8 = self.downsample(feature[0])
        src = (src_8 + src_16 + src_32) / 3
        #print(src.shape)torch.Size([8, 256, 32, 58]) 
        src = self.depth_head(src)
        denorms_offset = self.offpre(src) #8,4,32,58
        denorms_offset = denorms_offset.permute(0,2,3,1) #8,4,32,58 --> 8,32,58,4
        ori_denorms = ori_denorms.permute(0,2,1,3) #8,58,32,4 --> 8,32,58,4
        #pre_denorms = 0.1*(0.05*denorms_offset+0.95*denorms_offset.detach()) + ori_denorms #8,32,58,4 跟图像一一对应
        pre_denorms =  denorms_offset  + ori_denorms #8,32,58,4 跟图像一一对应
        #f_denorms = pre_denorms.permute(0,3,1,2)#8,4,32,58
        #Features = torch.cat([src,f_denorms],dim = 1) #8,260,32,58
        depth_logits = self.depth_classifier(src) # 8 271 32 58
        #pre_denorms = ori_denorms
        img_pre_denorms = pre_denorms[:,1:33,2:57,:] # 8,31,55,4
        #print(img_pre_denorms.shape)torch.Size([8, 31, 55, 4])
        #print(pre_denorms.shape) #torch.Size([8, 32, 58, 4])
        P = calibs
        depthmaps = torch.zeros((8,32, 58, 1)).to(self.device)
        GPdepthmaps = torch.zeros((8,55, 31, 1)).to(self.device)
   
        #print(P.shape) torch.Size([8, 3, 4])
        #print(random_flip_flag.shape)torch.Size([8])
        #print(img_sizes.shape)torch.Size([8, 2])
        ''' 
        for i in range(len(pre_denorms)):
            GPdepthmap = self.GPdepthmapFunc(P[i],img_pre_denorms[i],GPdepthmaps[i],random_flip_flag[i],img_sizes[i])
            #print(GPdepthmap.shape) torch.Size([31, 55, 1]) 55,31,1
            GPdepthmap = GPdepthmap.permute(1,0,2)  # 31,55,1
            depthmaps[i][1: 1+int(GPdepthmap.shape[0]), 2:2+int(GPdepthmap.shape[1])] = GPdepthmap
        '''
        #print(GPdepthmaps.shape) # torch.Size([8, 32, 58, 1])
        weighted_depth =depthmaps.squeeze(-1) # [8, 32, 58 # 由预测出的denorm推理的深度图
    
        #depth_probs = F.softmax(depth_logits, dim=1)
        #weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)

        # depth embeddings with depth positional encodings
        B, C, H, W = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos = pos.flatten(2).permute(2, 0, 1)

        depth_embed = self.depth_encoder(src, mask, pos)
        depth_embed = depth_embed.permute(1, 2, 0).reshape(B, C, H, W)

        #depth_pos_embed_ip = self.interpolate_depth_embed(weighted_depth)
        #depth_embed = depth_embed + depth_pos_embed_ip

        return depth_logits, depth_embed, pre_denorms, weighted_depth

    def interpolate_depth_embed(self, depth):
        depth = depth.clamp(min=0, max=self.depth_max)
        pos = self.interpolate_1d(depth, self.depth_pos_embed)
        pos = pos.permute(0, 3, 1, 2)
        return pos

    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings - 1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta
