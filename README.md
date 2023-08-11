# MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection
Official implementation of the paper ['MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection'](https://arxiv.org/pdf/2203.13310.pdf).

## Introduction
MonoDETR is the **first DETR-based model** for monocular 3D detection **without additional depth supervision, anchors or NMS**, which achieves leading performance on KITTI *val* and *test* set. We enable the vanilla transformer in DETR to be depth-aware and enforce the whole detection process guided by depth. In this way, each object estimates its 3D attributes adaptively from the depth-informative regions on the image, not limited by center-around features.
<div align="center">
  <img src="pipeline.jpg"/>
</div>

## Main Results
The randomness of training for monocular detection would cause the variance of ±1 AP<sub>3D</sub>. For reproducibility, we provide four training logs of MonoDETR on KITTI *val* set for the car category: (the stable version is still under tuned)

<table>
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Val, AP<sub>3D|R40</sub></td>   
        <td rowspan="2",div align="center">Logs</td>
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod.</td> 
        <td div align="center">Hard</td> 
    </tr>
    <tr>
        <td rowspan="4",div align="center">MonoDETR</td>
        <td div align="center">28.84%</td> 
        <td div align="center">20.61%</td> 
        <td div align="center">16.38%</td> 
        <td div align="center"><a href="https://drive.google.com/file/d/124u2WW_DqDyKrpUe3lQ8TR6xth8rn9YH/view?usp=sharing">log</a></td>
    </tr>  
    <tr>
        <td div align="center">26.66%</td> 
        <td div align="center">20.14%</td> 
        <td div align="center">16.88%</td> 
        <td div align="center"><a href="https://drive.google.com/file/d/1gSof60oOnno_qAHRViXKQ6CyqRI7O0tr/view?usp=sharing">log</a></td>
    </tr> 
    <tr>
        <td div align="center">29.53%</td> 
        <td div align="center">20.13%</td> 
        <td div align="center">16.57%</td> 
        <td div align="center"><a href="https://drive.google.com/file/d/1rrayzzwHGpddE1f_mfvq0RQb5xpWcPAL/view?usp=sharing">log</a></td>
    </tr> 
    <tr>
        <td div align="center">27.11%</td> 
        <td div align="center">20.08%</td> 
        <td div align="center">16.18%</td> 
        <td div align="center"><a href="https://drive.google.com/file/d/1D6IOkscfypGSEbsXcHZ60-q492zvMLp7/view?usp=sharing">log</a></td>
    </tr> 
</table>

MonoDETR on *test* set from official [KITTI benckmark](http://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=22a0e176d4f7794e7c142c93f4f8891749aa738f) for the car category:
<table>
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Test, AP<sub>3D|R40</sub></td>   
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod.</td> 
        <td div align="center">Hard</td> 
    </tr>
    <tr>
        <td rowspan="2",div align="center">MonoDETR</td>
        <td div align="center">24.52%</td> 
        <td div align="center">16.26%</td> 
        <td div align="center">13.93%</td> 
    </tr>  
    <tr>
        <td div align="center">25.00%</td> 
        <td div align="center">16.47%</td> 
        <td div align="center">13.58%</td> 
    </tr>  
    
</table>

## Docker Image
docker pull yanglei2024/yjx_cuda10-1-1:base

## 根据镜象建立容器：同时容器路径映射为本地目录
示例：docker run -it --gpus all --shm-size=32g -v /home/yujiaxin:/root --name cuda10-1  ea049d4cd81c bash
## 激活环境
conda activate monodetr
## Installation
1. Clone this project
    ```
    git clone https://github.com/HIYYJX/MonoGAE.git
    cd MonoGAE
    ```
2. compile the deformable attention:
    ```
    cd lib/models/monodetr/ops/
    bash make.sh
    
    cd ../../../..
    ```
    
3. Make dictionary for saving training losses:
    ```
    mkdir logs
    ```
4. Download [DAIR-V2X-I/single-infrastructure-side](https://thudair.baai.ac.cn/roadtest) 然后根据https://github.com/AIR-THU/DAIR-V2X/blob/main/tools/dataset_converter/dair2kitti.py 将其转换为kitti格式 and prepare the directory structure as:
    mkdir data

    ```
    │MonoGAE/
    ├──...
    ├──data/KITTIDataset/
    │   ├──ImageSets/
    |   |   |──test.txt
    |   |   |──train.txt
    |   |   |──trainval.txt
    |   |   |──val.txt
    │   ├──training/
    |   |   |──calib/
    |   |   |   |──000001.txt
    |   |   |   |──......
    |   |   |──denorm/
    |   |   |──image_2/
    |   |   |──label_2/
    |   |   |──velodyne/
    │   ├──testing/
    ├──...
    ```
    其中的training/denorm 是根据 get_denorm.py 获得
    You can also change the data path at "dataset/root_dir" in `configs/monodetr.yaml`.
    
## Get Started

### Train
/GroundDETR/lib/models/monodetr/monodetr.py 中第665行 losses中，'denorms' 意味着地平面方程矩阵编码，'depth_map'  意味着地平面深度图编码,可通过删减'denorms'或者'depth_map' 决定地平面的编码方式
 
You can modify the settings of models and training in `configs/monodetr.yaml` and appoint the GPU in `train.sh`:

    bash train.sh configs/monodetr.yaml > logs/monodetr.log
   
### Test
The best checkpoint will be evaluated as default. You can change it at "tester/checkpoint" in `configs/monodetr.yaml`:

    bash test.sh configs/monodetr.yaml


## Acknowlegment
This repo benefits from the excellent [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) and [MonoDLE](https://github.com/xinzhuma/monodle).

## Citation
```bash
@article{zhang2022monodetr,
  title={MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection},
  author={Zhang, Renrui and Qiu, Han and Wang, Tai and Xu, Xuanzhuo and Guo, Ziyu and Qiao, Yu and Gao, Peng and Li, Hongsheng},
  journal={arXiv preprint arXiv:2203.13310},
  year={2022}
}
```

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn.
