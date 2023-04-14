import numpy as np
import torch
import torch.nn as nn
from lib.datasets.utils import class2angle
from utils import box_ops
import math

padsize = np.array([28.,11.],dtype=np.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def affine_transform(pt, t):
   
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    #new_pt = torch.cat((pt, torch.ones(1).to(device=device)), dim=0)
    #new_pt = new_pt.unsqueeze(-1)
        # expand project points as [N, 3, 1]
    #new_pt=torch.matmul(t, new_pt)
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def softget(weighted_depth,outputs_coord):
        h,w = weighted_depth.shape[-2], weighted_depth.shape[-1] # 32 58
        pts_x =outputs_coord [0]  
        pts_y = outputs_coord[ 1]
        
        pts_x_low = math.floor(pts_x ) 
        pts_x_high =math.ceil(pts_x )  
        pts_y_low = math.floor(pts_y )  
        pts_y_high =math.ceil(pts_y)   
        pts_x_low = np.clip(pts_x_low, a_min=0,a_max = w-1) 
        pts_x_high = np.clip(pts_x_high, a_min=0,a_max = w-1) 
        pts_y_low = np.clip(pts_y_low, a_min=0,a_max = h-1) 
        pts_y_high =np.clip(pts_y_high, a_min=0,a_max = h-1) 
 
        rop_lt = weighted_depth[..., pts_y_low, pts_x_low]
        rop_rt = weighted_depth[..., pts_y_low, pts_x_high]
        rop_ld = weighted_depth[..., pts_y_high, pts_x_low]
        rop_rd = weighted_depth[..., pts_y_high, pts_x_high]
        
        rop_t = (1 - pts_x + pts_x_low) * rop_lt + (1 - pts_x_high + pts_x) * rop_rt
        rop_d = (1 - pts_x + pts_x_low) * rop_ld + (1 - pts_x_high + pts_x) * rop_rd
        rop = (1 - pts_y + pts_y_low) * rop_t + (1 - pts_y_high + pts_y) * rop_d
        return rop



def decode_detections(dets, info, calibs, cls_mean_size, threshold,trans_inv,depthmaps,pre_denorm,weighted_depth):
    '''
    NOTE: THIS IS A NUMPY FUNCTION
    input: dets, numpy array, shape in [batch x max_dets x dim]
    input: img_info, dict, necessary information of input images
    input: calibs, corresponding calibs for the input batch
    output:
    '''
    results = {}
    for i in range(dets.shape[0]):  # batch
        preds = []
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0])
            score = dets[i, j, 1]
            if score < threshold:
                continue

            # 2d bboxs decoding
            x = dets[i, j, 2] * 928 
            y = dets[i, j, 3] *  512
            w = dets[i, j, 4] * 928 
            h = dets[i, j, 5] * 512
            #x = dets[i, j, 2] * info['img_size'][i][0] 
            #y = dets[i, j, 3] * info['img_size'][i][1]
            #w = dets[i, j, 4] * info['img_size'][i][0]
            #h = dets[i, j, 5] * info['img_size'][i][1]
            bbox = np.array([x-w/2, y-h/2, x+w/2, y+h/2])
            bbox[:2] = bbox[:2]-padsize 
            bbox[2:] =bbox[2:] -padsize 
            
         
            

            bbox[:2] = bbox[:2]*2.2#affine_transform(bbox[:2], trans_inv[i])
            bbox[2:] = bbox[2:]*2.2#affine_transform(bbox[2:], trans_inv[i])
     
            # 3d bboxs decoding
            # depth decoding
            depth_p = dets[i, j, 6]

            # dimensions decoding
            dimensions = dets[i, j, 31:34]
            dimensions += cls_mean_size[int(cls_id)]

            # positions decoding
            #x3d = dets[i, j, 34]
            #y3d = dets[i, j, 35]
            size = np.array([928,512])
            #size = torch.tensor(size).to(device)
            pad = np.array([28,11])
            #pad =torch.tensor(pad).to( device)
            pad2 = np.array([2,1])
            #pad2 =torch.tensor(pad2).to(device)
            #coord =(dets[i, j, 34:36]*size-pad)/16+[2,1] 
            coord =(dets[i, j, 34:36]*size-pad)/16 # 本来图像的除以35.2 x,y
            pts = np.array(coord)
            w = 56
            h = 31
            pts_x = pts[0]
            pts_x  = np.clip(pts_x, a_min=0,a_max =w-1) 
            pts_y = pts[1]
            pts_y  = np.clip(pts_y , a_min=0,a_max = h-1) 
            
            denorm = pre_denorm[i] # 32,58,4
            P =  calibs[i].P2
            #coord = np.array([i+2,j+1])
            '''
            d  = denorm[int(pts_y)+1,int(pts_x)+2] #[16.69273   5.326345]
            W =torch.tensor([[P[0,0]/35.2,0,P[0,2]/35.2-coord[0]],[0,P[1,1]/35.2,P[1,2]/35.2-coord[1]],[d[0],d[1],d[2]]])
            result = torch.tensor([0,0,-d[3]]).reshape(-1,1)
            W_inv = torch.inverse(W) 
            vvxyz = torch.mm(W_inv,result)
            depth=vvxyz[2,0] 
            '''
            '''
            #print(coord.shape)
            #weighteddepth = depthmaps[i] 
            coord = np.array([pts_x+2,pts_y+1])
            weighteddepth = weighted_depth[i] 
            #weighteddepth = weighteddepth.transpose(1,0) # 
            weighteddepth = weighted_depth.cpu().numpy() # 32,58
            depth = softget(weighteddepth,coord)
            '''
            
            x3d = dets[i, j, 34] * 928 
            y3d = dets[i, j, 35] * 512
            
            
            x3d = x3d - padsize[0] # -28
            y3d = y3d - padsize[1] #-11
            xy = np.array([x3d, y3d])
            
            xy= xy *2.2 #affine_transform(xy , trans_inv[i])
            #xy= affine_transform(xy , trans_inv[i])
            x3d = xy[0]
            y3d =  xy[1]
            #x3d = dets[i, j, 34] * info['img_size'][i][0]
            #y3d = dets[i, j, 35] * info['img_size'][i][1]
            #locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)  
            locations = calibs[i].img_to_rect(x3d, y3d, depth_p).reshape(-1)   
            #locations[1] += dimensions[0] / 2
            
            # heading angle decoding
            alpha = get_heading_angle(dets[i, j, 7:31])
            ry = calibs[i].alpha2ry(alpha, x)


             
            score = score * dets[i, j, -1]
            preds.append([cls_id, alpha] + bbox.tolist() + dimensions.tolist() + locations.tolist() + [ry, score])
        results[info['img_id'][i]] = preds
    return results


def extract_dets_from_outputs(outputs, K=50, topk=50):
    # get src outputs

    # b, q, c
    out_logits = outputs['pred_logits']
    out_bbox = outputs['pred_boxes']

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), topk, dim=1)

    # final scores
    scores = topk_values
    # final indexes
    topk_boxes = (topk_indexes // out_logits.shape[2]).unsqueeze(-1)
    # final labels
    labels = topk_indexes % out_logits.shape[2]
    
    heading = outputs['pred_angle']
    size_3d = outputs['pred_3d_dim']
    depth = outputs['pred_depth'][:, :, 0: 1]
    sigma = outputs['pred_depth'][:, :, 1: 2]
    sigma = torch.exp(-sigma)


    # decode
    boxes = torch.gather(out_bbox, 1, topk_boxes.repeat(1, 1, 6))  # b, q', 4

    xs3d = boxes[:, :, 0: 1] 
    ys3d = boxes[:, :, 1: 2] 

    heading = torch.gather(heading, 1, topk_boxes.repeat(1, 1, 24))
    depth = torch.gather(depth, 1, topk_boxes)
    sigma = torch.gather(sigma, 1, topk_boxes) 
    size_3d = torch.gather(size_3d, 1, topk_boxes.repeat(1, 1, 3))

    corner_2d = box_ops.box_cxcylrtb_to_xyxy(boxes)

    xywh_2d = box_ops.box_xyxy_to_cxcywh(corner_2d)
    size_2d = xywh_2d[:, :, 2: 4]
    
    xs2d = xywh_2d[:, :, 0: 1]
    ys2d = xywh_2d[:, :, 1: 2]

    batch = out_logits.shape[0]
    labels = labels.view(batch, -1, 1)
    scores = scores.view(batch, -1, 1)
    xs2d = xs2d.view(batch, -1, 1)
    ys2d = ys2d.view(batch, -1, 1)
    xs3d = xs3d.view(batch, -1, 1)
    ys3d = ys3d.view(batch, -1, 1)

    detections = torch.cat([labels.float(), scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, sigma], dim=2)
    #detections = torch.cat([labels.float(), scores, xs2d, ys2d, size_2d,  heading, size_3d, xs3d, ys3d ], dim=2)

    return detections


############### auxiliary function ############


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)
