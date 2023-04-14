from cgi import print_directory
import os
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True
import math
import math
import cv2
import numba
from sympy import symbols, Eq, solve

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.kitti.kitti_utils import get_objects_from_label
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.kitti_utils import affine_transform
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
from lib.datasets.kitti.kitti_eval_python.eval import get_distance_eval_result
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
import copy
from .pd import PhotometricDistort

 
class KITTI_Dataset(data.Dataset):
    def __init__(self, split, cfg):

        # basic configuration
        self.root_dir = cfg.get('root_dir') # 'data/KITTIDataset'
        self.split = split#   'train' #  
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        #self.resolution = np.array([1280, 384])  # W * H
        self.resolution = np.array([928, 512],dtype = int)
        self.inputsize = np.array([1920//2.2,1080//2.2],dtype = int) #872,490
         
        

        #self.resolution = np.array([1920//2.2,1080//2.2])
        self.res_height = 512
        self.res_width = 928
         
        self.use_3d_center = cfg.get('use_3d_center', True)
        self.writelist = cfg.get('writelist', ['Car']) # 把这个改成Pe  Cyc
        #self.writelist = ["Car","Pedestrian",'Cyclist']
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg.get('bbox2d_type', 'anno')
        assert self.bbox2d_type in ['anno', 'proj']
        self.meanshape = cfg.get('meanshape', False)
        self.class_merging = cfg.get('class_merging', False)
        self.use_dontcare = cfg.get('use_dontcare', False)

        if self.class_merging:
            self.writelist.extend(['Van', 'Truck'])
        if self.use_dontcare:
            self.writelist.extend(['DontCare'])

        # data split loading
        assert self.split in ['train', 'val', 'trainval', 'test','hom_train','hom_val'] # 'train', 'val'
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt') # 有
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]
        #if self.split == 'train':
            #self.idx_list =self.idx_list [:5000]
        #else:
            #self.idx_list =self.idx_list [:2000]
        #self.idx_list =self.idx_list [:5000]
    

        # path configuration2022
        
        self.data_dir = os.path.join(self.root_dir, 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2') # 'training/image_2
        self.calib_dir = os.path.join(self.data_dir, 'calib') # 'training/calib
        self.label_dir = os.path.join(self.data_dir, 'label_2')# 'training/label_2
        self.denorm_dir = os.path.join(self.data_dir, 'denorm')# 'training/denorm

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False

        self.aug_pd = cfg.get('aug_pd', False)
        self.aug_crop = False #cfg.get('aug_crop', False)
        self.aug_calib = cfg.get('aug_calib', False)

        self.random_flip = cfg.get('random_flip', 0.5)
        self.random_crop = cfg.get('random_crop', 0.5)
        self.scale = cfg.get('scale', 0.4)
        self.shift = cfg.get('shift', 0.1)

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)   #有待商榷
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)    #有待商榷
        self.cls_mean_size = np.array([[1.76255119    ,0.66068622   , 0.84422524   ],
                                       [1.52563191462 ,1.62856739989, 3.88311640418],
                                       [1.73698127    ,0.59706367   , 1.76282397   ]])
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.downsample = 32
        self.pd = PhotometricDistort()
        self.clip_2d = cfg.get('clip_2d', False)

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.jpg' % idx) #'%06d.jpg' png
        #print(img_file)
        assert os.path.exists(img_file)
        return Image.open(img_file)    # (H, W, 3) RGB mode
    def get_denorm(self, idx):
        denorm_file = os.path.join(self.denorm_dir,'%06d.txt' % idx)
        assert os.path.exists(denorm_file)
        with open(denorm_file, 'r') as f:
            lines = f.readlines()
             
        denorm = np.array([float(item) for item in lines[0].split(' ')], dtype= 'float32')
        return denorm    # (H, W, 3) RGB mode
    
    
    
    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def eval(self, results_dir, logger):
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        img_ids = sorted(img_ids)
        dt_annos = kitti.get_label_annos(results_dir,g=False)
        gt_annos = kitti.get_label_annos(self.label_dir, img_ids,g = True)
         

        test_id = {'Car': 0, 'Pedestrian':1, 'Cyclist': 2}

        logger.info('==> Evaluating (official) ...')
        car_moderate = 0
        for category in self.writelist:
            results_str, results_dict, mAP3d_R40 = get_official_eval_result(gt_annos, dt_annos, test_id[category])
            if category == 'Car':
                car_moderate = mAP3d_R40
            logger.info(results_str)
            #if category == 'Pedestrian':
                #Pedestrian_moderate = mAP3d_R40
            #logger.info(results_str)
            #if category == 'Cyclist':
                #Cyclist_moderate = mAP3d_R40
            #logger.info(results_str)
        return car_moderate

    def __len__(self):
        return self.idx_list.__len__()
    def pad_image(self, image):
        img = np.array(image)
        h, w, c = img.shape
        ret_img = np.zeros((self.res_height, self.res_width, c)) # 512，928，256
        pad_y = (self.res_height - h) // 2 # 11
        pad_x = (self.res_width - w) // 2  # 28
        #print([h,w])
        #print([self.input_height, self.input_width])
        #print([pad_y, pad_x])
        ret_img[pad_y: pad_y + h, pad_x: pad_x + w] = img
        pad_size = np.array([pad_x, pad_y])

        return Image.fromarray(ret_img.astype(np.uint8)), pad_size
    
    #@numba.jit(nopython=True)
    def get_GPdepthmap(self, P, denorms,GPdepthmap,flip,imagesize):
        for i in range(int(denorms.shape[0])):  
            for j in range(int(denorms.shape[1])):  
                #denorm = denorms[i,j]
                w1 = P[0,0]/35.2
                w11 = P[0,2]/35.2
                w2 = P[1,1]/35.2
                w22 = P[1,2]/35.2
                d = denorms[i,j]
                if flip==True:
                    ii = imagesize[0]/35.2-i
                else:
                    ii =i
                W = np.matrix([[w1,0,w11-ii],[0,w2,w22-j],[d[0],d[1],d[2]]])
                result = np.array([0,0,-d[3]])
                W_inv = W.I 
                vvxyz = np.dot(W_inv,result)
                #print(vvxyz[0,2])
                GPdepthmap[i,j]=vvxyz[0,2]
              
        return GPdepthmap
 
    def GPdepthmapFunc(self, P,denorms,GPdepthmap, flip, imagesize):
     
        w1 = P[0, 0] / 35.2
        w11 = P[0, 2] / 35.2
        w2 = P[1, 1] / 35.2
        w22 = P[1, 2] / 35.2

        ii = np.arange(denorms.shape[0]) #[0,54]

        j = np.arange(denorms.shape[1])
        if flip == True:
            ii = imagesize[0] / 35.2 - ii

        W = np.ones((55, 31, 3, 3))
        W[:,:,:2,:2] = np.array([[w1, 0], [0, w2]])
        #print((w11 - ii).shape)(55,)
        #print((w11 - ii)[:, np.newaxis].shape)(55, 1)
        #print((w22 - j)[np.newaxis, :].shape)(1,31)
        W[:, :, 0, 2] = (w11 - ii)[:, np.newaxis]
        W[:, :, 1, 2] = (w22 - j)[np.newaxis, :]
        W[:, :, 2, :] = denorms[:, :, :3]
        W_inv = np.linalg.inv(W)
        result = np.zeros((55, 31, 3, 1))
        result[:, :, 2, 0] = -1 * denorms[:, :, 3]

        vvxyz = np.matmul(W_inv, result)
        GPdepthmap[:,:,0] = vvxyz[:,:,2,0]
        
        return GPdepthmap
                
    def softget(self,weighted_depth,outputs_coord):
        h,w = weighted_depth.shape[-2], weighted_depth.shape[-1]
        pts_x = outputs_coord[0]  
        pts_y = outputs_coord[1]
        
        pts_x_low = math.floor(pts_x) #pts_x.floor().long()
        pts_x_high = math.ceil(pts_x)  #pts_x.ceil().long() 
        pts_y_low = math.floor(pts_y) #pts_y.floor().long() 
        pts_y_high = math.ceil(pts_y) #pts_y.ceil().long()
        pts_x_low = np.clip(pts_x_low, a_min=0,a_max = w-1) 
        pts_x_high = np.clip(pts_x_high, a_min=0,a_max = w-1) 
        pts_y_low = np.clip(pts_y_low, a_min=0,a_max = h-1) 
        pts_y_high = np.clip(pts_y_high, a_min=0,a_max = h-1) 
 
        rop_lt = weighted_depth[..., pts_y_low, pts_x_low]
        rop_rt = weighted_depth[..., pts_y_low, pts_x_low]
        rop_ld = weighted_depth[..., pts_y_high, pts_x_low]
        rop_rd = weighted_depth[..., pts_y_high, pts_x_high]
        
        rop_t = (1 - pts_x + pts_x_low) * rop_lt + (1 - pts_x_high + pts_x) * rop_rt
        rop_d = (1 - pts_x + pts_x_low) * rop_ld + (1 - pts_x_high + pts_x) * rop_rd
        rop = (1 - pts_y + pts_y_low) * rop_t + (1 - pts_y_high + pts_y) * rop_d
        return rop
    def Find_Recent(self, x, y, x_, y_, k):
        list_stack_temp = []  # 建立一个空的栈
        for i in range(len(x)):
            list_temp = []
            if x[i] != x_:
                length = math.sqrt(pow(x[i] - x_, 2) + pow(y[i] - y_, 2))
                if len(list_stack_temp) < k:
                    list_stack_temp.append([(x[i], y[i]), length])
                    #print("临时栈中多了一组数据，目前有" + str(len(list_stack_temp)) + "组数据")
    
                else:
                    for m in list_stack_temp:
                        list_temp.append(m[1])
                        #print("临时列表中有" + str(len(list_temp)) + "组数据")
                    list_temp.append(length)
                    list_temp.sort()
                    if length != list_temp[-1]:
                        last_ = list_temp[-1]
                        for n in list_stack_temp:
                            if n[1] == last_:
                                list_stack_temp.remove(n)
                            else:
                                continue
                        list_stack_temp.append([(x[i], y[i]), length])
                    else:
                        continue
            else:
                continue
        return list_stack_temp
    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  
          # index mapping, get real data id
        # image loading
        img = self.get_image(index)
        denorm = self.get_denorm(index)
        #print(index)
        
        img_size = np.array(img.size)
        #print(img_size)
        depthmap = np.array([img_size[0]//35.2, img_size[1]//35.2]) # 54,30
        depthmaps = np.zeros((int(self.resolution[0]/16), int(self.resolution[1]/16), 1), dtype=denorm .dtype)#58,32 1
        
        input_denorms = np.zeros((int(self.resolution[0]/16), int(self.resolution[1]/16), 4), dtype=denorm .dtype)#58,32,4
        input_ori_denorms= np.zeros((int(self.resolution[0]/16), int(self.resolution[1]/16), 4), dtype=denorm .dtype)#58,32,4
      
        center = np.array(img_size) / 2
        crop_size, crop_scale = img_size, 1
        
        denorms = np.zeros((int(depthmap[0])+1, int(depthmap[1])+1, 4), dtype=denorm.dtype) #55,31
        ori_denorms = np.zeros((int(depthmap[0])+1, int(depthmap[1])+1, 4), dtype=denorm.dtype) #55,31,4
        denorms[:,:]=denorm
        ori_denorms[:,:]=denorm
        input_ori_denorms[:,:] = denorm
        
        input_denorms[:,:]=denorm
        GPdepthmap = np.zeros((int(depthmap[0])+1, int(depthmap[1])+1, 1), dtype=denorm .dtype) #55,31
        ori_GPdepthmap = np.zeros((int(depthmap[0])+1, int(depthmap[1])+1, 1), dtype=denorm .dtype) #55,31
        pad_x = (self.resolution[0]/16 - img_size[0]//35.2) // 2 #2 
        pad_y = (self.resolution[1]/16 - img_size[1]//35.2) // 2 #1 
        
        
        
        
        #img, pad_size = self.pad_image(img)
        features_size = self.resolution // self.downsample    # W * H 29,16

        # data augmentation for image
      
        random_flip_flag, random_crop_flag = False, False

        if self.data_augmentation:

            if self.aug_pd:
                img = np.array(img).astype(np.float32)
                img = self.pd(img).astype(np.uint8)
                img = Image.fromarray(img)

            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            if self.aug_crop:
                if np.random.random() < self.random_crop:
                    random_crop_flag = True
                    crop_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                    crop_size = img_size * crop_scale
                    center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                    center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.inputsize, inv=1) # 图片缩小2.2倍
 
        
        img = img.transform(tuple(self.inputsize.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        img, pad_size = self.pad_image(img) # 图片填充到928，512大小
        #print(img.size)
        #print(random_flip_flag)
        
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size}

        if self.split == 'test':
            calib = self.get_calib(index)
            return img, calib.P2, img, info

        #  ============================   get labels   ==============================
        objects = self.get_label(index)
        calib = self.get_calib(index)
        P = calib.P2
        # data augmentation for labels
        if random_flip_flag:
            if self.aug_calib:
                calib.flip(img_size)
            for object in objects:
                [x1, _, x2, _] = object.box2d
             
                object.box2d[0],  object.box2d[2] = img_size[0] - x2-1, img_size[0] - x1 -1 #已经处理过了
                object.alpha = np.pi - object.alpha
                object.ry = np.pi - object.ry
                if self.aug_calib:
                    object.pos[0] *= -1
                if object.alpha > np.pi:  object.alpha -= 2 * np.pi  # check range
                if object.alpha < -np.pi: object.alpha += 2 * np.pi
                if object.ry > np.pi:  object.ry -= 2 * np.pi
                if object.ry < -np.pi: object.ry += 2 * np.pi

        # labels encoding
        calibs = np.zeros((self.max_objs, 3, 4), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=np.bool)
        labels = np.zeros((self.max_objs), dtype=np.int8)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32) 
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        boxes = np.zeros((self.max_objs, 4), dtype=np.float32)
        boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32)
        bottoms_3d = np.zeros((self.max_objs, 2), dtype=np.float32)


        poses = []
        imgposes =[]
        #print(index)
        #print(random_flip_flag)
        for i in range(len(objects)):
            pos =  objects[i].pos
            pos = pos.reshape(-1, 3) 
            imgpos, _=  calib.rect_to_img(pos)
            imgpos = imgpos[0] 
          
            if random_flip_flag and not self.aug_calib: 
                imgpos[0] = img_size[0] - imgpos[0] 
             
            imgpos = imgpos/35.2
            if imgpos[0]<0 or imgpos[0]>depthmap[0]:
                continue
            if imgpos[1]<0 or imgpos[1]>depthmap[1]:
                continue
            pos = objects[i].pos
            poses.append(pos)
            imgposes.append(imgpos)
            
        L = len(poses)//3
        poses = np.array(poses) 
        imgposes = np.array(imgposes) 
        #print(index)
                                                                   
        if len(imgposes)>0:
        
            x = imgposes[:,0]
            sorted_id_x = sorted(range(len(x)), key=lambda k: x[k], reverse=False)
            Imgposes = imgposes[sorted_id_x]
            Poses = poses[sorted_id_x]
            pair = {}
            x = Imgposes[:,0]
            y = Imgposes[:,1]
            x_discrete = []
            y_discrete = []
            
            for i in range(len(x)):
                pair[x[i]]=Poses[i]
                x_discrete.append(x[i])
                y_discrete.append(y[i])
            while len(x_discrete)>2:
                list1 = self.Find_Recent(x_discrete, y_discrete, x_discrete[0], y_discrete[0], 2)
                x_ = [x_discrete[0]]
                y_ = [y_discrete[0]]
                x_discrete.remove(x_[0])
                y_discrete.remove(y_[0])
                for i in list1:
                    x_.append(i[0][0])
                    y_.append(i[0][1])
                #if len(pair)<
                if len(x_)<3:
                    continue
                pos1=pair[x_[0]]
                pos2=pair[x_[1]]
                pos3=pair[x_[2]]
                #pos1=pair[x_discrete[0]]
                #pos2=pair[list1[0][0][0]]
                #pos3=pair[list1[1][0][0]]
                a = (pos2[1]-pos1[1])*(pos3[2]-pos1[2])-(pos3[1]-pos1[1])*(pos2[2]-pos1[2])
                b = (pos2[2]-pos1[2])*(pos3[0]-pos1[0])-(pos3[2]-pos1[2])*(pos2[0]-pos1[0])
                c = (pos2[0]-pos1[0])*(pos3[1]-pos1[1])-(pos3[0]-pos1[0])*(pos2[1]-pos1[1])
                d = -(a*pos1[0]+b*pos1[1]+c*pos1[2])
                xmin = min(x_)
                ymin = min(y_)
                xmax= max(x_)
                ymax = max(y_)
                xmin =  math.floor(xmin)
                ymin = math.floor(ymin)
                xmax = math.ceil(xmax)
                ymax = math.ceil(ymax)
                size = d/denorm[3]
                a /= size
                b /= size
                c /=size
                d /= size
                denorms[xmin:xmax+1,ymin:ymax+1] = np.array([a,b,c,d])
                #for x1 in x_:
                #x_discrete.remove(x[0])
                #for y1 in y_:
                #y_discrete.remove(y[0])
        
            '''       
            p = 0
            while p < len(Imgposes)-2:
                pos1=Poses[p]
                pos2=Poses[p+1]
                pos3=Poses[p+2]
                a = (pos2[1]-pos1[1])*(pos3[2]-pos1[2])-(pos3[1]-pos1[1])*(pos2[2]-pos1[2])
                b = (pos2[2]-pos1[2])*(pos3[0]-pos1[0])-(pos3[2]-pos1[2])*(pos2[0]-pos1[0])
                c = (pos2[0]-pos1[0])*(pos3[1]-pos1[1])-(pos3[0]-pos1[0])*(pos2[1]-pos1[1])
                d = -(a*pos1[0]+b*pos1[1]+c*pos1[2])
                xmin = min(Imgposes[p:p+3][:,0])
                ymin = min(Imgposes[p:p+3][:,1])
                xmax= max(Imgposes[p:p+3][:,0])
                ymax = max(Imgposes[p:p+3][:,1])
                xmin =  math.floor(xmin)
                ymin = math.floor(ymin)
                xmax = math.ceil(xmax)
                ymax = math.ceil(ymax)
                size = d/denorm[3]
                a /= size
                b /= size
                c /=size
                d /= size
                denorms[xmin:xmax+1,ymin:ymax+1] = np.array([a,b,c,d])
                p = p+2
            '''
            #GPdepthmap = self.get_GPdepthmap(P,denorms,GPdepthmap,random_flip_flag,img_size)  
            GPdepthmap = self.GPdepthmapFunc(P,denorms,GPdepthmap,random_flip_flag,img_size)  
            #GPdepthmap = self.get_GPdepthmap(P,ori_denorms,GPdepthmap,random_flip_flag,img_size) 
            depthmaps[int(pad_x): int(pad_x) + int(depthmap[0])+1, int(pad_y): int(pad_y) +int(depthmap[1])+1] = GPdepthmap
        else:
            GPdepthmap = GPdepthmap
            depthmaps[int(pad_x): int(pad_x) + int(depthmap[0])+1, int(pad_y): int(pad_y) +int(depthmap[1])+1] = GPdepthmap
    
 
                 
        input_ori_denorms[int(pad_x): int(pad_x) + int(depthmap[0])+1, int(pad_y): int(pad_y) +int(depthmap[1])+1] = ori_denorms
         
        
        input_denorms[int(pad_x): int(pad_x) + int(depthmap[0])+1, int(pad_y): int(pad_y) +int(depthmap[1])+1] = denorms
         

        '''
        #如果翻转了，生成深度时，需要进行下面这步骤
        for p in imgposes:
            if random_flip_flag==True:
                i = img_size[0]/35.2-p[0]
            else:
                i =p[0]
            #i = img_size[0]/35.2-p[0] #这一步很关键
            j = p[1]
            coord = np.array([i+2,j+1])
            weighteddepth =depthmaps.squeeze(-1)
            weighteddepth = weighteddepth.transpose(1,0)
            depth_map = self.softget(weighteddepth,coord)
            print(depth_map)
            #i = img_size[0] - i*35.2
            denorm = denorms[int(p[0]),int(j)] #[16.69273   5.326345]
          
            W = np.matrix([[P[0,0]/35.2,0,P[0,2]/35.2-i],[0,P[1,1]/35.2,P[1,2]/35.2-j],[denorm[0],denorm[1],denorm[2]]])
            result = np.array([0,0,-denorm[3]])
            W_inv = W.I 
            vvxyz = np.dot(W_inv,result)
            print(vvxyz)
        '''
        
        
        
        
        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
        poses = []
        imgposes =[]

        for i in range(object_num):
            # filter objects by writelist
            
            if objects[i].cls_type not in self.writelist:
                continue

            # filter inappropriate samples
            if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                continue

            # ignore the samples beyond the threshold [hard encoding]
            threshold = 205
            if objects[i].pos[-1] > threshold:
                continue

            # process 2d bbox & get 2d center
            bbox_2d = objects[i].box2d.copy()
            
            # add affine transformation for 2d boxes.
            
            bbox_2d[:2] = bbox_2d[:2]/2.2# affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = bbox_2d[2:]/2.2#affine_transform(bbox_2d[2:], trans)
          
   
            bbox_2d[:2] = bbox_2d[:2]+pad_size
            bbox_2d[2:] = bbox_2d[2:]+pad_size
            #print(pad_size)
            
       
            if  bbox_2d[0]<0 or bbox_2d[2]>self.resolution[0]:
                continue
            if bbox_2d[1]<0 or bbox_2d[3]>self.resolution[1]:
                continue

            # process 3d center
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
            corner_2d = bbox_2d.copy()
            
            
            pos =  objects[i].pos 
             
            pos = pos.reshape(-1, 3) 
            imgpos, _=  calib.rect_to_img(pos)
            pos =  objects[i].pos 
            imgpos = imgpos[0] 
            if random_flip_flag and not self.aug_calib: 
                imgpos[0] = img_size[0] - imgpos[0] -1
            imgpos = imgpos/35.2
            if imgpos[0]<0 or imgpos[0]>depthmap[0]:
                continue
            if imgpos[1]<0 or imgpos[1]>depthmap[1]:
                continue
            imgposes.append(imgpos)
            poses.append(pos)
            
            l, h, w =  objects[i].l,objects[i].h,objects[i].w  
            ry = objects[i].ry # 处理过了，翻转
            x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2,0,0]
            y_corners = [0, 0, 0, 0, -h, -h, -h, -h,0,-h/2]
            z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2,0,0]
            corners3d = np.vstack([x_corners, y_corners, z_corners])
            R = np.array([[np.cos( ry), 0, np.sin( ry)],
                        [0, 1, 0],
                            [-np.sin( ry), 0, np.cos( ry)]])
            corners3d = np.dot(R, corners3d)
            denorm_norm = denorm[:3] / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
            ori_denorm = np.array([0.0, -1.0, 0.0]) 
            theta = -1 * math.acos(np.dot(denorm_norm, ori_denorm))
            n_vector = np.cross(denorm[:3], ori_denorm)
            n_vector_norm = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
            rotation_matrix, j = cv2.Rodrigues(theta * n_vector_norm)
            corners3d = np.dot(rotation_matrix, corners3d)
            pos =  objects[i].pos 
            corners3d = corners3d.T + pos
            
            
            
            #center_3d = corners3d[9]
            center_3d = corners3d[8]
            #center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment
            if random_flip_flag and not self.aug_calib:  # random flip for center3d
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d =center_3d/2.2 #affine_transform(center_3d.reshape(-1), trans)
            center_3d = (center_3d + pad_size)
            
            bottom_3d = corners3d[8]
            #center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
            bottom_3d = bottom_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            bottom_3d, _ = calib.rect_to_img(bottom_3d)  # project 3D center to image plane
            bottom_3d = bottom_3d[0]  # shape adjustment
            if random_flip_flag and not self.aug_calib:  # random flip for center3d
                bottom_3d[0] = img_size[0] - bottom_3d[0]
            bottom_3d = affine_transform(bottom_3d.reshape(-1), trans)
            bottom_3d = (bottom_3d + pad_size)
         
       
            

            # filter 3d center out of img
            proj_inside_img = True

            if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]: 
                proj_inside_img = False
            if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]: 
                proj_inside_img = False

            if proj_inside_img == False:
                    continue

            # class
            cls_id = self.cls2id[objects[i].cls_type]
            labels[i] = cls_id

            # encoding 2d/3d boxes
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[i] = 1. * w, 1. * h

            center_2d_norm = center_2d / self.resolution
            size_2d_norm = size_2d[i] / self.resolution

            corner_2d_norm = corner_2d
            corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution
            corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution
            bottom_3d_norm = bottom_3d / self.resolution
            center_3d_norm = center_3d / self.resolution
        

            l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l = np.clip(l, 0, 1)
                    r = np.clip(r, 0, 1)
                    t = np.clip(t, 0, 1)
                    b = np.clip(b, 0, 1)
                else:
                    continue		

            boxes[i] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            boxes_3d[i] = center_3d_norm[0], center_3d_norm[1], l, r, t, b
            bottoms_3d[i] = bottom_3d_norm[0], bottom_3d_norm[1]

            # encoding depth
            #depth[i] = corners3d[9][-1] * crop_scale
            depth[i] = objects[i].pos[-1] * crop_scale

            # encoding heading angle
            heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
            if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
            if heading_angle < -np.pi: heading_angle += 2 * np.pi
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding size_3d
            src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size

            if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
                mask_2d[i] = 1

            calibs[i] = calib.P2
        #L = len(poses)//3
        #poses = np.array(poses)  #29,3
        #print(poses)
        #imgposes = np.array(imgposes)  #29,2
        #print(imgposes)
        #print(random_flip_flag)
        #P = calib.P2
        #for i in range (L):
            #pos1=poses[i*3:i*3+1][0]
            #pos2=poses[i*3+1:i*3+2][0]
            #pos3=poses[i*3+2:i*3+3][0]
            #a = (pos2[1]-pos1[1])*(pos3[2]-pos1[2])-(pos3[1]-pos1[1])*(pos2[2]-pos1[2])
            #b = (pos2[2]-pos1[2])*(pos3[0]-pos1[0])-(pos3[2]-pos1[2])*(pos2[0]-pos1[0])
            #c = (pos2[0]-pos1[0])*(pos3[1]-pos1[1])-(pos3[0]-pos1[0])*(pos2[1]-pos1[1])
            #d = -(a*pos1[0]+b*pos1[1]+c*pos1[2])
            #size = d/denorm[3]
            #a /= size
            #b /= size
            #c /=size
            #d /= size
            
            #xmin = min(imgposes[i*3:i*3+3][:,0])
            #ymin = min(imgposes[i*3:i*3+3][:,1])
            #xmax= max(imgposes[i*3:i*3+3][:,0])
            #ymax = max(imgposes[i*3:i*3+3][:,1])
            #xmin =  math.floor(xmin)
            #ymin = math.floor(ymin)
            #xmax = math.ceil(xmax)
            #ymax = math.ceil(ymax)
            #denorms[xmin:xmax+1,ymin:ymax+1] = np.array([a,b,c,d])
            
            #w1 = P[0,0]/35.2
            #w11 = P[0,2]/35.2
            #w2 = P[1,1]/35.2
            #w22 = P[1,2]/35.2
            #d = denorms[23,13]
            #23.14293 img_size[0]/35.2-
            #d = denorm
            #W = np.matrix([[w1,0,w11-31.4025245],[0,w2,w22-13.549998],[d[0],d[1],d[2]]])
            #result = np.array([0,0,-d[3]])
            #W_inv = W.I 
            #vvxyz = np.dot(W_inv,result)
            #print(vvxyz)
            #GPdepthmap
            
         
        #GPdepthmap = self.get_GPdepthmap(P,denorms,GPdepthmap,random_flip_flag,img_size)
        #print(GPdepthmap)
        #print(pad_x.dtype)
         
        #print(imgposes)
        #print(poses)
        #print(P)
        #print(GPdepthmap[33,16])
        #print(GPdepthmap[8,9])
        #print((bottoms_3d[0][:2] *[928,512]-[28,11])/16)
        
        #depthmaps[int(pad_x): int(pad_x) + int(depthmap[0])+1, int(pad_y): int(pad_y) +int(depthmap[1])+1] = GPdepthmap
        #depthmaps[:int(pad_x)]
 
        #print(depthmaps[22,18])
        #print(depthmaps[23,18])
        #print(depthmaps[22,19])
        #print(depthmaps[23,19])
        #print(denorms[20,17])
        #print(depthmaps[14,17])
        # collect return data
        inputs = img
        targets = {
                   'calibs': calibs,
                   'indices': indices,
                   'img_size': img_size,
                   'labels': labels,
                   'boxes': boxes,
                   'boxes_3d': boxes_3d,
                   'depth': depth,
                   'size_2d': size_2d,
                   'size_3d': size_3d,
                   'src_size_3d': src_size_3d,
                   'heading_bin': heading_bin,
                   'heading_res': heading_res,
                   'mask_2d': mask_2d,
                   'depthmaps':depthmaps,
                   'denorms':input_denorms,
                   'ori_denorms':input_ori_denorms,
                   'bottoms_3d':bottoms_3d}

        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size}
        targets['trans_inv'] = trans_inv
        return inputs, calib.P2, targets, info,random_flip_flag


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    cfg = {'root_dir': '../../../data/KITTI',
           'random_flip': 0.0, 'random_crop': 1.0, 'scale': 0.8, 'shift': 0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist':['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center':False}
    dataset = KITTI_Dataset('train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.show()
        # print(targets['size_3d'][0][0])

        # test heatmap
        heatmap = targets['heatmap'][0]  # image id
        heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        heatmap.show()

        break

    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
