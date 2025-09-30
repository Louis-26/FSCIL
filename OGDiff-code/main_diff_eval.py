import argparse
import torch
import pickle
import os
import datetime
import time
from dataset.dataloader import get_dataloader, get_domain_specific_dataloader
from model.condition import generate_condition_vector_from_current_split

from model.model_diff import ConvNet, Linear_fw_Diffusion, MutiClassifier, MutiClassifier_, resnet18_fast, resnet50_fast, MutiProtoClassifier
from model.model_diff import FeatureDiffusionNet
from optimizer.optimizer import get_optimizer, get_scheduler
from loss.OVALoss import OVALoss
from train.test import *
from util.log import log
from util.util import *
import types
import os
import csv
import numpy as np
from torch.amp import GradScaler, autocast

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()  # 释放所有空闲的缓存块

print("可用 GPU 数量:", torch.cuda.device_count())
print("当前使用的 GPU:", torch.cuda.current_device())
import os
print(f"My Process ID is: {os.getpid()}")

def compute_hscore(y_true, y_pred, known_class_set):
    known_mask = np.isin(y_true, list(known_class_set))
    unknown_mask = ~known_mask
    acc_known = (y_pred[known_mask] == y_true[known_mask]).mean() if known_mask.sum() > 0 else 0
    acc_unknown = (y_pred[unknown_mask] == y_true[unknown_mask]).mean() if unknown_mask.sum() > 0 else 0
    if acc_known + acc_unknown == 0:
        return 0.0
    return 2 * acc_known * acc_unknown / (acc_known + acc_unknown)

def compute_oscr(y_true, y_pred, known_class_set):
    known_mask = np.isin(y_true, list(known_class_set))
    unknown_mask = ~known_mask
    tpr = (y_pred[known_mask] == y_true[known_mask]).mean() if known_mask.sum() > 0 else 0
    fpr = (y_pred[unknown_mask] != y_true[unknown_mask]).mean() if unknown_mask.sum() > 0 else 0
    return tpr - fpr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='PACS')
    parser.add_argument('--source-domain', nargs='+', default=['photo', 'cartoon', 'art_painting'])
    parser.add_argument('--target-domain', nargs='+', default=['sketch'])
    parser.add_argument('--known-classes', nargs='+', default=['dog', 'elephant', 'giraffe', 'horse', 'guitar', 'house', 'person'])
    parser.add_argument('--unknown-classes', nargs='+', default=[])
    
    # parser.add_argument('--dataset', default='OfficeHome')
    # parser.add_argument('--source-domain', nargs='+', default=['Art', 'Clipart', 'Product'])
    # parser.add_argument('--target-domain', nargs='+', default=['RealWorld'])
    # parser.add_argument('--known-classes', nargs='+', default=['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 
    #     'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 
    #     'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 
    #     'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 
    #     'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 
    #     'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard',  
    #     'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 
    #     'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven',
    #     'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 
    #     'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler',
    #     'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 
    #     'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 
    #     'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam'
    #     ])
    # parser.add_argument('--unknown-classes', nargs='+', default=[ 
            

    #     ])

    # parser.add_argument('--dataset', default='DigitsDG')
    # parser.add_argument('--source-domain', nargs='+', default=['mnist', 'mnist_m', 'svhn'])
    # parser.add_argument('--target-domain', nargs='+', default=['syn'])
    # parser.add_argument('--known-classes', nargs='+', default=['0', '1', '2', '3', '4', '5'])
    # parser.add_argument('--unknown-classes', nargs='+', default=['6', '7', '8', '9'])

    # parser.add_argument('--dataset', default='VLCS')
    # parser.add_argument('--source-domain', nargs='+', default=['CALTECH', 'PASCAL', 'SUN'])
    # parser.add_argument('--target-domain', nargs='+', default=['LABELME',])
    # parser.add_argument('--known-classes', nargs='+', default=['0', '1', '2', '3', '4'])
    # parser.add_argument('--unknown-classes', nargs='+', default=[])

    # parser.add_argument('--dataset', default='TerraIncognita')
    # parser.add_argument('--source-domain', nargs='+', default=['location_38', 'location_43', 'location_46'])
    # parser.add_argument('--target-domain', nargs='+', default=['location_100'])
    # parser.add_argument('--known-classes', nargs='+', default=['bobcat', 'coyote', 'dog', 'opossum', 'rabbit', 'raccoon', 'squirrel', 'bird', 'cat', 'empty',])
    # parser.add_argument('--unknown-classes', nargs='+', default=[])

    # parser.add_argument('--dataset', default='DomainNet')
    # parser.add_argument('--source-domain', nargs='+', default=['clipart', 'infograph', 'painting', 'quickdraw', 'real'])
    # parser.add_argument('--target-domain', nargs='+', default=['sketch'])
    # parser.add_argument('--known-classes', nargs='+', default=['aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 
    #     'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 
    #     'barn', 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 
    #     'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 
    #     'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 
    #     'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 
    #     'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 
    #     'ceiling_fan', 'cello', 'cell_phone', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 
    #     'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 
    #     'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 
    #     'dragon', 'dresser', 'drill', 'drums', 'duck', 
    #     'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 
    #     'feather', 'fence', 'finger', 'fireplace', 'firetruck', 'fire_hydrant', 'fish', 'flamingo', 'flashlight', 'flip_flops', 
    #     'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 'giraffe',
    #     'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 
    #     'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 
    #     'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 
    #     'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'lighter', 
    #     'lighthouse', 'lightning', 'light_bulb', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 
    #     'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 
    #     'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 
    #     'octagon', 'octopus', 'onion', 'oven', 'owl', 'paintbrush', 'paint_can', 'palm_tree', 'panda', 'pants', 
    #     'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 
    #     'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool', 
    #     'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 
    #     'rake', 'remote_control', 'rhinoceros', 'rifle', 'river', 'rollerskates', 'roller_coaster', 'sailboat', 'sandwich', 'saw', 
    #     'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe', 
    #     'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag', 'smiley_face', 'snail', 'snake', 
    #     'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 
    #     'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop_sign', 'stove', 
    #     'strawberry', 'streetlight', 'string_bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 
    #     'syringe', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent', 'The_Eiffel_Tower', 
    #     'The_Great_Wall_of_China', 'The_Mona_Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 
    #     'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 
    #     'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 
    #     'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag'
    #     ])
    # parser.add_argument('--unknown-classes', nargs='+', default=[])

    parser.add_argument('--random-split', action='store_true')
    parser.add_argument('--gpu', default='0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--task-d', type=int, default=3)
    parser.add_argument('--task-c', type=int, default=3)
    parser.add_argument('--task-per-step', nargs='+', type=int, default=[3, 3, 3])

    parser.add_argument('--net-name', default='resnet18')
    parser.add_argument('--optimize-method', default="SGD")
    parser.add_argument('--schedule-method', default='StepLR')
    parser.add_argument('--num-epoch', type=int, default=6000)
    parser.add_argument('--eval-step', type=int, default=3)
    parser.add_argument('--lr', type=float, default=8e-4) 
    parser.add_argument('--meta-lr', type=float, default=1e-2)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--without-cls', action='store_true')
    parser.add_argument('--without-bcls', action='store_true')
    parser.add_argument('--share-param', action='store_true')

    parser.add_argument('--save-dir', default='./experiment')
    parser.add_argument('--save-name', default='demo_eval')
    parser.add_argument('--save-best-test', action='store_true')
    parser.add_argument('--save-later', action='store_true')

    parser.add_argument('--num-epoch-before', type=int, default=0)

    parser.add_argument('--load-pretrained-path', type=str, default='./experiment/model/val/pacs_resnet18.tar', 
                        help='Path to the pre-trained .tar model to use for diffusion training')
    parser.add_argument('--diffusion-lr', type=float, default=1e-4, help='扩散模型优化器的学习率')
    parser.add_argument('--diffusion-timesteps', type=int, default=50, help='扩散过程的时间步数')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 释放所有空闲的缓存块
        
        num_gpus_available = torch.cuda.device_count() # This count is AFTER CUDA_VISIBLE_DEVICES is applied
        print(f"可用 (可见的) GPU 数量: {num_gpus_available}")

        if num_gpus_available == 0:
            if args.gpu: # User specified GPUs but none are seen
                print(f"警告: 指定的 GPUs '{args.gpu}' 导致没有可见的 GPU. 请检查 CUDA 安装和 GPU ID.")
            else: # No GPUs specified and none found
                print("警告: CUDA 可用但未检测到 GPU 或没有可见的 GPU.")
            device = torch.device("cpu")
            print("将在 CPU 上运行。")
            num_gpus_to_use = 0
        else:
            # Check if the number of specified GPUs matches available ones
            specified_gpu_ids = [id.strip() for id in args.gpu.split(',') if id.strip()]
            if args.gpu and len(specified_gpu_ids) > num_gpus_available:
                print(f"警告: 请求了 {len(specified_gpu_ids)} 个 GPUs ({args.gpu}), 但只有 {num_gpus_available} 个变为可见。将使用 {num_gpus_available} 个。")
            
            num_gpus_to_use = num_gpus_available # Use all visible GPUs
            # The primary device for placing the model initially and for single-GPU operations.
            # DataParallel handles data transfer to other GPUs.
            # PyTorch re-indexes visible GPUs starting from 0. So 'cuda:0' is the first visible GPU.
            device = torch.device("cuda:2") 
            print(f"将使用 {num_gpus_to_use} 个 GPU。主设备: {device}")
            print(f"当前 PyTorch 使用的 GPU (逻辑索引): {torch.cuda.current_device()} (这应该是0对于主设备)")
    else:
        print("CUDA 不可用，将在 CPU 上运行。")
        device = torch.device("cpu")
        num_gpus_to_use = 0

    # It can be used to replace the following code, but the editor may take it as an error.
    # locals().update(vars(args))

    # It can be replaced by the preceding code.
    dataset = args.dataset
    source_domain = sorted(args.source_domain)
    target_domain = sorted(args.target_domain)
    known_classes = sorted(args.known_classes)
    unknown_classes = sorted(args.unknown_classes)   
    random_split = args.random_split
    gpu = args.gpu
    batch_size = args.batch_size
    task_d = args.task_d
    task_c = args.task_c
    task_per_step = args.task_per_step
    net_name = args.net_name
    optimize_method = args.optimize_method
    schedule_method = args.schedule_method
    num_epoch = args.num_epoch
    eval_step = args.eval_step
    lr = args.lr
    meta_lr = args.meta_lr
    nesterov = args.nesterov
    without_cls = args.without_cls
    without_bcls = args.without_bcls
    share_param = args.share_param
    save_dir = args.save_dir
    save_name = args.save_name   
    save_later = args.save_later
    save_best_test = args.save_best_test
    num_epoch_before = args.num_epoch_before

    assert task_d * task_c == sum(task_per_step)

    torch.set_num_threads(4)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    crossval = True

    if dataset == 'PACS':
        train_dir = 'data/PACS_train'
        val_dir = 'data/PACS_crossval'
        test_dir = 'data/PACS'
        sub_batch_size = batch_size // 2    
        small_img = False
    # elif dataset == 'OfficeHome':
    #     train_dir = ''
    #     val_dir = ''
    #     test_dir = ''
    #     sub_batch_size = batch_size // 2
    #     small_img = False
    # elif dataset == "DigitsDG":
    #     train_dir = ''
    #     val_dir = ''
    #     test_dir = ''
    #     sub_batch_size = batch_size // 2
    #     small_img = True
    # elif dataset == 'VLCS':
    #     train_dir = ''
    #     val_dir = ''
    #     test_dir = ''
    #     sub_batch_size = batch_size 
    #     small_img = False
    # elif dataset == 'TerraIncognita':
    #     train_dir = ''
    #     val_dir = ''
    #     test_dir = ''
    #     sub_batch_size = batch_size
    #     small_img = False
    # elif dataset == "DomainNet":
    #     train_dir = ''
    #     val_dir = ''
    #     test_dir = ''
    #     sub_batch_size = batch_size // 2
    #     small_img = False
    
    
    # ========== 优化日志文件名，加入时间戳 ===========
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(save_dir, 'log', f'{save_name}_{now_str}_train.txt')
    param_path = os.path.join(save_dir, 'param', save_name + '.pkl')
    model_val_path = os.path.join(save_dir, 'model', 'val', save_name + '.tar')
    model_test_path = os.path.join(save_dir, 'model', 'test', save_name + '.tar')
    renovate_step = int(num_epoch*0.85) if save_later else 0
    model_save_path = os.path.join(save_dir, 'model', 'diffusion', save_name + '.tar')

    log('GPU: {}'.format(gpu), log_path)

    log('Loading path...', log_path)

    log('Save name: {}'.format(save_name), log_path)
    log('Save best test: {}'.format(save_best_test), log_path)
    log('Save later: {}'.format(save_later), log_path)

    with open(param_path, 'wb') as f: 
        pickle.dump(vars(args), f, protocol=pickle.HIGHEST_PROTOCOL)

    log('Loading dataset...', log_path)

    num_domain = len(source_domain)
    num_classes = len(known_classes)

    class_index = [i for i in range(num_classes)]
    group_length = (num_classes-1) // 10 + 1

    if dataset == "OfficeHome" and len(unknown_classes) == 0:
        group_length = 6
    elif dataset == 'TerraIncognita' and len(unknown_classes) == 0:
        group_length = 2
    elif dataset == 'DomainNet' and len(unknown_classes) == 0:
        group_length = 35

    log('Group length: {}'.format(group_length), log_path)
    
    group_index_list = [i for i in range((num_classes-1)//group_length + 1)]
    num_group = len(group_index_list)
 
    domain_specific_loader, val_k = get_domain_specific_dataloader(root_dir = train_dir, domain=source_domain, classes=known_classes, group_length=group_length, batch_size=sub_batch_size, small_img=small_img, crossval=crossval and random_split)
    if crossval and val_k == None:
        val_k, *_ = get_dataloader(root_dir=val_dir, domain=source_domain, classes=known_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="val", small_img=small_img, shuffle=False, drop_last=False, num_workers=16)

 
    test_k, *_ = get_dataloader(root_dir=test_dir, domain=target_domain, classes=known_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=16)
    if len(unknown_classes) > 0:
        test_u, *_ = get_dataloader(root_dir=test_dir, domain=target_domain, classes=unknown_classes, batch_size=batch_size, get_domain_label=False, get_class_label=False, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=16)   
    else:
        test_u = None

    log('DataSet: {}'.format(dataset), log_path)
    log('Source domain: {}'.format(source_domain), log_path)
    log('Target domain: {}'.format(target_domain), log_path)
    log('Known classes: {}'.format(known_classes), log_path)
    log('Unknown classes: {}'.format(unknown_classes), log_path)
    log('Batch size: {}'.format(batch_size), log_path)
    log('Number of task(domain): {}'.format(task_d), log_path)
    log('Number of task(class): {}'.format(task_c), log_path)
    log('Tasks per step: {}'.format(task_per_step), log_path)
    log('CrossVal: {}'.format(crossval), log_path)
    log('Loading models...', log_path)

    # ========== 新增：定义train_loader ===========
    train_loader, *_ = get_dataloader(
        root_dir=train_dir,
        domain=source_domain,
        classes=known_classes,
        batch_size=batch_size,
        get_domain_label=False,
        get_class_label=True,
        instr="train",
        small_img=small_img,
        shuffle=True,
        drop_last=True,
        num_workers=16
    )

    # 定义扩散模型配置 (可以从args传入或在此处硬编码)
    diffusion_hyperparams = {
        'num_timesteps': args.diffusion_timesteps if hasattr(args, 'diffusion_timesteps') else 100, # 例如，添加命令行参数
        'schedule_type': args.diffusion_schedule if hasattr(args, 'diffusion_schedule') else 'linear',
        'loss_type': 'l2',
        'ema_decay': 0.999, # EMA衰减可以小一些，因为权重会频繁生成
        'ema_start': 50,   # 较早开始EMA
        'time_embed_dim_for_dmfunc': 128,
        'hidden_dim_scale_factor_for_dmfunc': 0.5
    }

    # ========== 新增：特征扩散主网络初始化 ===========
    if net_name == 'resnet18':
        backbone = resnet18_fast(pretrained=False)
        feature_dim = 512
    elif net_name == 'resnet50':
        backbone = resnet50_fast(pretrained=False)
        feature_dim = 2048
    elif net_name == 'convnet':
        backbone = ConvNet()
        feature_dim = 256
    else:
        raise ValueError(f"未知的网络名称: {net_name}")

    # 分类器可选：原型分类器或线性层
    classifier = nn.Linear(feature_dim, num_classes)  # 可替换为ProtoClassifier(feature_dim, num_classes)
    net = FeatureDiffusionNet(backbone, classifier, feature_dim, diffusion_steps=100, hidden_dim=512)
    net = net.to(device)
    if num_gpus_to_use > 1:
        net = nn.DataParallel(net) 

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_rec = torch.nn.MSELoss()
    scaler = GradScaler('cuda')

    # ========== 新训练循环 ===========
    patience = 200  # 你可以根据需要调整
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_epoch = 0
    try:
        start_time = time.time()
        all_train_acc = []
        all_val_acc = []
        all_val_hscore = []
        all_val_oscr = []
        metrics_csv_path = os.path.join(save_dir, 'metrics', f'{save_name}_{now_str}_metrics.csv')
        os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
        with open(metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Acc', 'Val Acc', 'Val H-Score', 'Val OSCR'])

        for epoch in range(num_epoch):
            net.train()
            total_loss, total_cls, total_rec = 0, 0, 0
            for batch in train_loader:  # 需保证train_loader输出(x, y)
                x, y = batch[0].to(device), batch[1].to(device)
                with autocast('cuda'):
                    logits, feat, noisy_feat, denoised_feat, t, noise = net(x, return_feature=True)
                    loss_cls = criterion_cls(logits, y)
                    loss_rec = criterion_rec(denoised_feat, feat)
                    loss = loss_cls + loss_rec
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                total_cls += loss_cls.item()
                total_rec += loss_rec.item()
            scheduler.step()
            log(f"Epoch {epoch+1}: loss={total_loss:.4f}, cls={total_cls:.4f}, rec={total_rec:.4f}", log_path)

            # ========== 新增：每轮统计训练集准确率 ==========
            net.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in train_loader:
                    x, y = batch[0].to(device), batch[1].to(device)
                    logits = net(x)
                    pred = logits.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            acc = correct / total if total > 0 else 0
            log(f"Train Acc: {acc:.4f}", log_path)

            # ========== 新增：每轮统计验证集准确率、H-Score、OSCR ==========
            val_acc, val_hscore, val_oscr = None, None, None
            if 'val_k' in locals() and val_k is not None:
                y_true_all, y_pred_all = [], []
                with torch.no_grad():
                    for batch in val_k:
                        x, y = batch[0].to(device), batch[1].to(device)
                        logits = net(x)
                        pred = logits.argmax(dim=1)
                        y_true_all.append(y.cpu().numpy())
                        y_pred_all.append(pred.cpu().numpy())
                y_true_all = np.concatenate(y_true_all)
                y_pred_all = np.concatenate(y_pred_all)
                val_acc = (y_pred_all == y_true_all).mean()
                known_class_set = set(range(num_classes))
                val_hscore = compute_hscore(y_true_all, y_pred_all, known_class_set)
                val_oscr = compute_oscr(y_true_all, y_pred_all, known_class_set)
                log(f"Val Acc: {val_acc:.4f}", log_path)
            # 记录到列表
            all_train_acc.append(acc)
            all_val_acc.append(val_acc if val_acc is not None else 0)
            all_val_hscore.append(val_hscore if val_hscore is not None else 0)
            all_val_oscr.append(val_oscr if val_oscr is not None else 0)
            # 写入csv
            with open(metrics_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, acc, val_acc, val_hscore, val_oscr])

            # ========== 每10个epoch记录一次耗时 ==========
            if (epoch + 1) % 10 == 0:
                elapsed = time.time() - start_time
                msg = f"[Time] Epoch {epoch+1}: 累计耗时 {elapsed/60:.2f} 分钟 ({elapsed:.1f} 秒)"
                print(msg)
                log(msg, log_path)

            # ========== Early Stopping 机制 ==========
            if val_acc is not None:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                    log(f"[EarlyStopping] 新最佳Val Acc: {best_val_acc:.4f} (epoch {best_epoch})", log_path)
                else:
                    epochs_no_improve += 1
                    log(f"[EarlyStopping] Val Acc未提升，已连续{epochs_no_improve}轮", log_path)
                if epochs_no_improve >= patience:
                    log(f"[EarlyStopping] 连续{patience}轮Val Acc未提升，提前终止训练。最佳Val Acc: {best_val_acc:.4f} (epoch {best_epoch})", log_path)
                    print(f"[EarlyStopping] 连续{patience}轮Val Acc未提升，提前终止训练。最佳Val Acc: {best_val_acc:.4f} (epoch {best_epoch})")
                    break
    except KeyboardInterrupt:
        print("训练被中断，已记录到日志。")
        log("训练被用户中断（KeyboardInterrupt）！", log_path)
        # ========== 新增：中断时保存最佳模型及参数 ==========
        if best_val_acc > 0:
            # 保存当前最佳模型权重
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save({
                'model_state_dict': net.state_dict(),
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
                'args': vars(args) if 'args' in locals() else None
            }, model_save_path)
            msg = f"[Interrupt] 最佳模型已保存到: {model_save_path}\n最佳Val Acc: {best_val_acc:.4f} (epoch {best_epoch})"
            print(msg)
            log(msg, log_path)
        else:
            msg = "[Interrupt] 尚无最佳模型，无模型保存。"
            print(msg)
            log(msg, log_path)
