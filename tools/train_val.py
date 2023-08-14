import warnings
warnings.filterwarnings("ignore")
import torch
import os
import sys
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
 
import random
import yaml
import argparse
import datetime

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader, my_worker_init_fn
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from torch.utils.data import DataLoader, DistributedSampler

def main_parser():
    parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
    parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
    parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--seed', default=None, type=int,  help='seed for initializing training. ')
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    return args


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def main_worker(local_rank, cfg, nprocs):
    dist.init_process_group(backend='nccl')
    model_name = cfg['model_name']
    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    # build dataset
    print(cfg['dataset']['train_split'])
    train_dataset = KITTI_Dataset(split=cfg['dataset']['train_split'], cfg=cfg['dataset'])
    test_dataset = KITTI_Dataset(split=cfg['dataset']['test_split'], cfg=cfg['dataset'])
    
    # build model
    model, loss = build_model(cfg['model'])
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    cudnn.benchmark = True

    distributed = nprocs > 1
    train_sampler, val_sampler = None, None
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,  device_ids=[local_rank], find_unused_parameters=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        model = model

    # build dataloader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg['dataset']['batch_size'],
                              num_workers=4,
                              worker_init_fn=my_worker_init_fn,
                              # shuffle=True,
                              pin_memory=False,
                              drop_last=False,
                              sampler=train_sampler)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=cfg['dataset']['batch_size'],
                             num_workers=4,
                             worker_init_fn=my_worker_init_fn,
                             # shuffle=False,
                             pin_memory=False,
                             drop_last=False,
                             sampler=test_sampler)
 
    if args.evaluate_only:
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        train_cfg=cfg['trainer'],
                        model_name=model_name)
        tester.test()
        return
 
    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      loss=loss,
                      model_name=model_name)

    tester = Tester(cfg=cfg['tester'],
                    model=trainer.model,
                    dataloader=test_loader,
                    logger=logger,
                    train_cfg=cfg['trainer'],
                    model_name=model_name)
    if cfg['dataset']['test_split'] != 'test':
        trainer.tester = tester

    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))

    trainer.train(train_loader, test_loader, train_sampler, test_sampler, distributed, local_rank)

    if cfg['dataset']['test_split'] == 'test':
        return
    '''
    logger.info('###################  Testing  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Split: %s' % (cfg['dataset']['test_split']))
    tester.test()
    '''

def evaluate_worker(cfg):
    model_name = cfg['model_name']
    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    # build dataset
    test_dataset = KITTI_Dataset(split=cfg['dataset']['test_split'], cfg=cfg['dataset'])
    
    # build model
    model, loss = build_model(cfg['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=cfg['dataset']['batch_size'],
                             num_workers=4,
                             worker_init_fn=my_worker_init_fn,
                             # shuffle=False,
                             pin_memory=False,
                             drop_last=False)
 
    logger.info('###################  Evaluation Only  ##################')
    tester = Tester(cfg=cfg['tester'],
                    model=model,
                    dataloader=test_loader,
                    logger=logger,
                    train_cfg=cfg['trainer'],
                    model_name=model_name)
    tester.test()
    return

if __name__ == '__main__':
    args = main_parser()
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    output_path = os.path.join('./' + cfg["trainer"]['save_path'], cfg['model_name'])
    os.makedirs(output_path, exist_ok=True)

    random.seed(cfg['random_seed'])
    torch.manual_seed(cfg['random_seed'])
    cudnn.deterministic = True
    nprocs = torch.cuda.device_count()

    if not args.evaluate_only:
        main_worker(args.local_rank, cfg, nprocs)
    else:
        evaluate_worker(cfg)
 
