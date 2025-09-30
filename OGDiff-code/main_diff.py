# import argparse
# import torch
# import pickle
# import os
# from dataset.dataloader import get_dataloader, get_domain_specific_dataloader
# from model.condition import generate_condition_vector_from_current_split

# from model.model_diff import ConvNet, Linear_fw_Diffusion, MutiClassifier, MutiClassifier_, resnet18_fast, resnet50_fast
# from optimizer.optimizer import get_optimizer, get_scheduler
# from loss.OVALoss import OVALoss
# from train.test import *
# from util.log import log
# from util.util import *
# import types
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# torch.cuda.empty_cache()  # 释放所有空闲的缓存块

# print("可用 GPU 数量:", torch.cuda.device_count())
# print("当前使用的 GPU:", torch.cuda.current_device())
# import os
# print(f"My Process ID is: {os.getpid()}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('--dataset', default='PACS')
#     parser.add_argument('--source-domain', nargs='+', default=['photo', 'cartoon', 'art_painting'])
#     parser.add_argument('--target-domain', nargs='+', default=['sketch'])
#     parser.add_argument('--known-classes', nargs='+', default=['dog', 'elephant', 'giraffe', 'horse', 'guitar', 'house', 'person'])
#     parser.add_argument('--unknown-classes', nargs='+', default=[])
    
#     # parser.add_argument('--dataset', default='OfficeHome')
#     # parser.add_argument('--source-domain', nargs='+', default=['Art', 'Clipart', 'Product'])
#     # parser.add_argument('--target-domain', nargs='+', default=['RealWorld'])
#     # parser.add_argument('--known-classes', nargs='+', default=['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 
#     #     'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 
#     #     'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 
#     #     'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 
#     #     'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 
#     #     'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard',  
#     #     'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 
#     #     'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven',
#     #     'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 
#     #     'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler',
#     #     'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 
#     #     'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 
#     #     'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam'
#     #     ])
#     # parser.add_argument('--unknown-classes', nargs='+', default=[ 
            

#     #     ])

#     # parser.add_argument('--dataset', default='DigitsDG')
#     # parser.add_argument('--source-domain', nargs='+', default=['mnist', 'mnist_m', 'svhn'])
#     # parser.add_argument('--target-domain', nargs='+', default=['syn'])
#     # parser.add_argument('--known-classes', nargs='+', default=['0', '1', '2', '3', '4', '5'])
#     # parser.add_argument('--unknown-classes', nargs='+', default=['6', '7', '8', '9'])

#     # parser.add_argument('--dataset', default='VLCS')
#     # parser.add_argument('--source-domain', nargs='+', default=['CALTECH', 'PASCAL', 'SUN'])
#     # parser.add_argument('--target-domain', nargs='+', default=['LABELME',])
#     # parser.add_argument('--known-classes', nargs='+', default=['0', '1', '2', '3', '4'])
#     # parser.add_argument('--unknown-classes', nargs='+', default=[])

#     # parser.add_argument('--dataset', default='TerraIncognita')
#     # parser.add_argument('--source-domain', nargs='+', default=['location_38', 'location_43', 'location_46'])
#     # parser.add_argument('--target-domain', nargs='+', default=['location_100'])
#     # parser.add_argument('--known-classes', nargs='+', default=['bobcat', 'coyote', 'dog', 'opossum', 'rabbit', 'raccoon', 'squirrel', 'bird', 'cat', 'empty',])
#     # parser.add_argument('--unknown-classes', nargs='+', default=[])

#     # parser.add_argument('--dataset', default='DomainNet')
#     # parser.add_argument('--source-domain', nargs='+', default=['clipart', 'infograph', 'painting', 'quickdraw', 'real'])
#     # parser.add_argument('--target-domain', nargs='+', default=['sketch'])
#     # parser.add_argument('--known-classes', nargs='+', default=['aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 
#     #     'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 
#     #     'barn', 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 
#     #     'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 
#     #     'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 
#     #     'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 
#     #     'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 
#     #     'ceiling_fan', 'cello', 'cell_phone', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 
#     #     'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 
#     #     'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 
#     #     'dragon', 'dresser', 'drill', 'drums', 'duck', 
#     #     'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 
#     #     'feather', 'fence', 'finger', 'fireplace', 'firetruck', 'fire_hydrant', 'fish', 'flamingo', 'flashlight', 'flip_flops', 
#     #     'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 'giraffe',
#     #     'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 
#     #     'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 
#     #     'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 
#     #     'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'lighter', 
#     #     'lighthouse', 'lightning', 'light_bulb', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 
#     #     'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 
#     #     'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 
#     #     'octagon', 'octopus', 'onion', 'oven', 'owl', 'paintbrush', 'paint_can', 'palm_tree', 'panda', 'pants', 
#     #     'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 
#     #     'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool', 
#     #     'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 
#     #     'rake', 'remote_control', 'rhinoceros', 'rifle', 'river', 'rollerskates', 'roller_coaster', 'sailboat', 'sandwich', 'saw', 
#     #     'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe', 
#     #     'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag', 'smiley_face', 'snail', 'snake', 
#     #     'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 
#     #     'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop_sign', 'stove', 
#     #     'strawberry', 'streetlight', 'string_bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 
#     #     'syringe', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent', 'The_Eiffel_Tower', 
#     #     'The_Great_Wall_of_China', 'The_Mona_Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 
#     #     'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 
#     #     'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 
#     #     'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag'
#     #     ])
#     # parser.add_argument('--unknown-classes', nargs='+', default=[])

#     parser.add_argument('--random-split', action='store_true')
#     parser.add_argument('--gpu', default='0,1,2,3')
#     parser.add_argument('--batch-size', type=int, default=4)
#     parser.add_argument('--task-d', type=int, default=3)
#     parser.add_argument('--task-c', type=int, default=3)
#     parser.add_argument('--task-per-step', nargs='+', type=int, default=[3, 3, 3])

#     parser.add_argument('--net-name', default='resnet50')
#     parser.add_argument('--optimize-method', default="SGD")
#     parser.add_argument('--schedule-method', default='StepLR')
#     parser.add_argument('--num-epoch', type=int, default=6000)
#     parser.add_argument('--eval-step', type=int, default=300)
#     parser.add_argument('--lr', type=float, default=2e-4) 
#     parser.add_argument('--meta-lr', type=float, default=1e-2)
#     parser.add_argument('--nesterov', action='store_true')
#     parser.add_argument('--without-cls', action='store_true')
#     parser.add_argument('--without-bcls', action='store_true')
#     parser.add_argument('--share-param', action='store_true')

#     parser.add_argument('--save-dir', default='./experiment')
#     parser.add_argument('--save-name', default='demo')
#     parser.add_argument('--save-best-test', action='store_true')
#     parser.add_argument('--save-later', action='store_true')

#     parser.add_argument('--num-epoch-before', type=int, default=0)
    
#     args = parser.parse_args()

#     # ---- MODIFIED SECTION: GPU Setup and Environment Configuration ----
#     # Set PYTORCH_CUDA_ALLOC_CONF early
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#     # Configure GPUs based on arguments
#     # The 'gpu' argument from parser is now 'args.gpu'
#     if args.gpu:
#         os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#         print(f"CUDA_VISIBLE_DEVICES set to: {args.gpu}")
#     else:
#         # If args.gpu is empty, PyTorch will see all available GPUs by default.
#         # You might want to explicitly set it to "" or handle it based on your needs.
#         print("No specific GPUs assigned via --gpu arg, PyTorch will use its default visibility.")

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()  # 释放所有空闲的缓存块
        
#         num_gpus_available = torch.cuda.device_count() # This count is AFTER CUDA_VISIBLE_DEVICES is applied
#         print(f"可用 (可见的) GPU 数量: {num_gpus_available}")

#         if num_gpus_available == 0:
#             if args.gpu: # User specified GPUs but none are seen
#                 print(f"警告: 指定的 GPUs '{args.gpu}' 导致没有可见的 GPU. 请检查 CUDA 安装和 GPU ID.")
#             else: # No GPUs specified and none found
#                 print("警告: CUDA 可用但未检测到 GPU 或没有可见的 GPU.")
#             device = torch.device("cpu")
#             print("将在 CPU 上运行。")
#             num_gpus_to_use = 0
#         else:
#             # Check if the number of specified GPUs matches available ones
#             specified_gpu_ids = [id.strip() for id in args.gpu.split(',') if id.strip()]
#             if args.gpu and len(specified_gpu_ids) > num_gpus_available:
#                 print(f"警告: 请求了 {len(specified_gpu_ids)} 个 GPUs ({args.gpu}), 但只有 {num_gpus_available} 个变为可见。将使用 {num_gpus_available} 个。")
            
#             num_gpus_to_use = num_gpus_available # Use all visible GPUs
#             # The primary device for placing the model initially and for single-GPU operations.
#             # DataParallel handles data transfer to other GPUs.
#             # PyTorch re-indexes visible GPUs starting from 0. So 'cuda:0' is the first visible GPU.
#             device = torch.device("cuda:0") 
#             print(f"将使用 {num_gpus_to_use} 个 GPU。主设备: {device}")
#             print(f"当前 PyTorch 使用的 GPU (逻辑索引): {torch.cuda.current_device()} (这应该是0对于主设备)")
#     else:
#         print("CUDA 不可用，将在 CPU 上运行。")
#         device = torch.device("cpu")
#         num_gpus_to_use = 0

#     # It can be used to replace the following code, but the editor may take it as an error.
#     # locals().update(vars(args))

#     # It can be replaced by the preceding code.
#     dataset = args.dataset
#     source_domain = sorted(args.source_domain)
#     target_domain = sorted(args.target_domain)
#     known_classes = sorted(args.known_classes)
#     unknown_classes = sorted(args.unknown_classes)   
#     random_split = args.random_split
#     gpu = args.gpu
#     batch_size = args.batch_size
#     task_d = args.task_d
#     task_c = args.task_c
#     task_per_step = args.task_per_step
#     net_name = args.net_name
#     optimize_method = args.optimize_method
#     schedule_method = args.schedule_method
#     num_epoch = args.num_epoch
#     eval_step = args.eval_step
#     lr = args.lr
#     meta_lr = args.meta_lr
#     nesterov = args.nesterov
#     without_cls = args.without_cls
#     without_bcls = args.without_bcls
#     share_param = args.share_param
#     save_dir = args.save_dir
#     save_name = args.save_name   
#     save_later = args.save_later
#     save_best_test = args.save_best_test
#     num_epoch_before = args.num_epoch_before

#     assert task_d * task_c == sum(task_per_step)

#     torch.set_num_threads(4)
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
#     # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     crossval = True

#     if dataset == 'PACS':
#         train_dir = 'data/PACS_train'
#         val_dir = 'data/PACS_crossval'
#         test_dir = 'data/PACS'
#         sub_batch_size = batch_size // 2    
#         small_img = False
#     # elif dataset == 'OfficeHome':
#     #     train_dir = ''
#     #     val_dir = ''
#     #     test_dir = ''
#     #     sub_batch_size = batch_size // 2
#     #     small_img = False
#     # elif dataset == "DigitsDG":
#     #     train_dir = ''
#     #     val_dir = ''
#     #     test_dir = ''
#     #     sub_batch_size = batch_size // 2
#     #     small_img = True
#     # elif dataset == 'VLCS':
#     #     train_dir = ''
#     #     val_dir = ''
#     #     test_dir = ''
#     #     sub_batch_size = batch_size 
#     #     small_img = False
#     # elif dataset == 'TerraIncognita':
#     #     train_dir = ''
#     #     val_dir = ''
#     #     test_dir = ''
#     #     sub_batch_size = batch_size
#     #     small_img = False
#     # elif dataset == "DomainNet":
#     #     train_dir = ''
#     #     val_dir = ''
#     #     test_dir = ''
#     #     sub_batch_size = batch_size // 2
#     #     small_img = False
    
    
#     log_path = os.path.join(save_dir, 'log', save_name + '_train.txt')
#     param_path = os.path.join(save_dir, 'param', save_name + '.pkl')
#     model_val_path = os.path.join(save_dir, 'model', 'val', save_name + '.tar')
#     model_test_path = os.path.join(save_dir, 'model', 'test', save_name + '.tar')
#     renovate_step = int(num_epoch*0.85) if save_later else 0

#     log('GPU: {}'.format(gpu), log_path)

#     log('Loading path...', log_path)

#     log('Save name: {}'.format(save_name), log_path)
#     log('Save best test: {}'.format(save_best_test), log_path)
#     log('Save later: {}'.format(save_later), log_path)

#     with open(param_path, 'wb') as f: 
#         pickle.dump(vars(args), f, protocol=pickle.HIGHEST_PROTOCOL)

#     log('Loading dataset...', log_path)

#     num_domain = len(source_domain)
#     num_classes = len(known_classes)

#     class_index = [i for i in range(num_classes)]
#     group_length = (num_classes-1) // 10 + 1

#     if dataset == "OfficeHome" and len(unknown_classes) == 0:
#         group_length = 6
#     elif dataset == 'TerraIncognita' and len(unknown_classes) == 0:
#         group_length = 2
#     elif dataset == 'DomainNet' and len(unknown_classes) == 0:
#         group_length = 35

#     log('Group length: {}'.format(group_length), log_path)
    
#     group_index_list = [i for i in range((num_classes-1)//group_length + 1)]
#     num_group = len(group_index_list)
 
#     domain_specific_loader, val_k = get_domain_specific_dataloader(root_dir = train_dir, domain=source_domain, classes=known_classes, group_length=group_length, batch_size=sub_batch_size, small_img=small_img, crossval=crossval and random_split)
#     if crossval and val_k == None:
#         val_k, *_ = get_dataloader(root_dir=val_dir, domain=source_domain, classes=known_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="val", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)

 
#     test_k, *_ = get_dataloader(root_dir=test_dir, domain=target_domain, classes=known_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)
#     if len(unknown_classes) > 0:
#         test_u, *_ = get_dataloader(root_dir=test_dir, domain=target_domain, classes=unknown_classes, batch_size=batch_size, get_domain_label=False, get_class_label=False, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)   
#     else:
#         test_u = None

#     log('DataSet: {}'.format(dataset), log_path)
#     log('Source domain: {}'.format(source_domain), log_path)
#     log('Target domain: {}'.format(target_domain), log_path)
#     log('Known classes: {}'.format(known_classes), log_path)
#     log('Unknown classes: {}'.format(unknown_classes), log_path)
#     log('Batch size: {}'.format(batch_size), log_path)
#     log('Number of task(domain): {}'.format(task_d), log_path)
#     log('Number of task(class): {}'.format(task_c), log_path)
#     log('Tasks per step: {}'.format(task_per_step), log_path)
#     log('CrossVal: {}'.format(crossval), log_path)
#     log('Loading models...', log_path)

#     # 定义扩散模型配置 (可以从args传入或在此处硬编码)
#     diffusion_hyperparams = {
#         'num_timesteps': args.diffusion_timesteps if hasattr(args, 'diffusion_timesteps') else 100, # 例如，添加命令行参数
#         'schedule_type': args.diffusion_schedule if hasattr(args, 'diffusion_schedule') else 'linear',
#         'loss_type': 'l2',
#         'ema_decay': 0.999, # EMA衰减可以小一些，因为权重会频繁生成
#         'ema_start': 50,   # 较早开始EMA
#         'time_embed_dim_for_dmfunc': 128,
#         'hidden_dim_scale_factor_for_dmfunc': 0.5
#     }

#     # 获取特征提取器的输出维度 (这里需要根据 net_name 确定)
#     # 这是一个示例，您需要根据实际的骨干网络确定 feature_dim
#     if net_name == 'resnet18':
#         temp_backbone = resnet18_fast(pretrained=False) # 仅用于获取维度
#         # ResNet通常在fc层之前有一个AdaptiveAvgPool2d，输出通道数是block.expansion * 512
#         # 对于 ResNet18, block.expansion = 1, 所以是 512
#         actual_feature_dim = 512 
#     elif net_name == 'resnet50':
#         temp_backbone = resnet50_fast(pretrained=False)
#         actual_feature_dim = 2048 # ResNet50 block.expansion = 4
#     elif net_name == "convnet":
#         # ConvNet 类中定义了 self._out_features = 256
#         actual_feature_dim = 256
#     else:
#         raise ValueError(f"未知的网络名称: {net_name} 用于确定 feature_dim")

#     # 定义 MEDIC 条件向量的维度，这里我们假设它等于特征提取器的输出维度
#     medic_condition_dim = actual_feature_dim

#     if share_param:
#         muticlassifier_init = MutiClassifier_
#     else:
#         muticlassifier_init = MutiClassifier

#     # 实例化特征提取器 (骨干网络)
#     backbone_net = None
#     if net_name == 'resnet18':
#         backbone_net = resnet18_fast() # pretrained 参数根据您的需求设置
#     elif net_name == 'resnet50':
#         backbone_net = resnet50_fast()
#     elif net_name == "convnet":
#         backbone_net = ConvNet()
    
#     # 实例化包含扩散线性层的 MutiClassifier
#     net = muticlassifier_init(
#         net=backbone_net, 
#         num_classes=num_classes, 
#         feature_dim=actual_feature_dim,
#         condition_dim=medic_condition_dim, # 传递 condition_dim
#         diffusion_config=diffusion_hyperparams # 传递扩散模型配置
#     )
#     # if torch.cuda.is_available():
#     #     print(f"Torch 版本: {torch.__version__}")
#     #     print(f"CUDA 可用: {torch.cuda.is_available()}")
#     #     for i in range(torch.cuda.device_count()):
#     #         try:
#     #             total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**2)
#     #             # torch.cuda.mem_get_info(i) 返回 (free, total) 以字节为单位
#     #             free_mem, _ = torch.cuda.mem_get_info(i)
#     #             free_mem_mb = free_mem / (1024**2)
#     #             allocated_mem_mb = torch.cuda.memory_allocated(i) / (1024**2)
#     #             reserved_mem_mb = torch.cuda.memory_reserved(i) / (1024**2) # PyTorch缓存的显存
#     #             print(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
#     #             print(f"  总显存: {total_mem:.2f} MB")
#     #             print(f"  通过 mem_get_info 获取的空闲显存: {free_mem_mb:.2f} MB")
#     #             print(f"  PyTorch 已分配显存: {allocated_mem_mb:.2f} MB")
#     #             print(f"  PyTorch 预留/缓存显存: {reserved_mem_mb:.2f} MB")
#     #         except Exception as e:
#     #             print(f"获取GPU {i} 信息时出错: {e}")
#     # else:
#     #     print("CUDA 不可用。")
#     #     net = net.to(device)

#     if num_gpus_to_use > 1:
#         net = nn.DataParallel(net) 
#         log(f'Model wrapped with nn.DataParallel for {num_gpus_to_use} GPUs.', log_path)
#     elif num_gpus_to_use == 1 : # Single GPU
#         log(f'Model running on single GPU: {device}', log_path)
#     else: # CPU
#         log('Model running on CPU.', log_path)
#     # ---- END OF MODIFIED SECTION ----


#     diffusion_model_params = []
#     # Access module correctly if DataParallel is used
#     net = net.module if num_gpus_to_use > 1 else net
#     net.to(device)
#     print(f"After net.to(device): model_accessor.net.conv1.weight device: {net.net.conv1.weight.device}")

#     # --- 为扩散模型参数创建单独的优化器 ---
#    # 使用 model_accessor 确保在 DataParallel 或普通模式下都能正确访问
#     model_accessor = net.module if isinstance(net, torch.nn.DataParallel) else net

#     if isinstance(model_accessor.classifier, Linear_fw_Diffusion): # MutiClassifier 的情况
#         diffusion_model_params.extend(list(model_accessor.classifier.diffusion_model.denoise_model.parameters()))
#     if isinstance(model_accessor.b_classifier, Linear_fw_Diffusion): # MutiClassifier 和 MutiClassifier_ 都有
#         diffusion_model_params.extend(list(model_accessor.b_classifier.diffusion_model.denoise_model.parameters()))
    
#     # 如果 MutiClassifier_ 只有一个 b_classifier, 上面的 if 会处理
#     # 如果 MutiClassifier 中还有其他 Linear_fw_Diffusion 实例，也需要加入

#     optimizer_diffusion = None
#     if diffusion_model_params:
#         # --- 修改开始 ---
#         # 原来的 AdamW 优化器:
#         # optimizer_diffusion = torch.optim.AdamW(filter(lambda p: p.requires_grad, diffusion_model_params), 
#         #                                         lr=args.diffusion_lr if hasattr(args, 'diffusion_lr') else 1e-4, 
#         #                                         weight_decay=1e-6)
#         # log('为扩散模型创建了 AdamW 优化器。', log_path)

#         # 替换为 SGD 优化器:
#         diffusion_lr = getattr(args, 'diffusion_lr', 1e-3) # SGD通常需要比AdamW稍大的学习率
#         optimizer_diffusion = torch.optim.SGD(filter(lambda p: p.requires_grad, diffusion_model_params),
#                                               lr=diffusion_lr,
#                                               momentum=0.9) # 可以选择添加 momentum
#         log(f'为扩散模型创建了 SGD 优化器，学习率: {diffusion_lr}。', log_path)
#         # --- 修改结束 ---
#     else:
#         log('警告：未找到扩散模型的参数，不创建 optimizer_diffusion。', log_path)


#     # --- MEDIC 主优化器 (主要优化特征提取器) ---
#     # 从 net.parameters() 中排除扩散模型的参数 (或者更精确地只选择骨干网络的参数)
#     # main_model_params = [p for name, p in net.named_parameters() if 'diffusion_model.denoise_model' not in name and p.requires_grad]
#     # 如果主优化器只优化骨干网络 net.net:
#     main_model_params = list(model_accessor.net.parameters()) # 确保只获取骨干网络的参数
    
#     # optimize_method = args.optimize_method # 从args获取
#     # lr = args.lr # 从args获取
#     # nesterov = args.nesterov # 从args获取
#     # schedule_method = args.schedule_method # 从args获取
#     # num_epoch = args.num_epoch # 从args获取

#     if args.optimize_method == 'SGD':
#         optimizer = get_optimizer(params=main_model_params, instr=args.optimize_method, lr=args.lr, nesterov=args.nesterov) 
#         scheduler = get_scheduler(optimizer=optimizer, instr=args.schedule_method, step_size=int(args.num_epoch*0.8), gamma=0.1)
#     elif args.optimize_method == 'Adam': # 你主优化器用的是Adam，这里保持不变，只改扩散模型的
#         optimizer = get_optimizer(params=main_model_params, instr=args.optimize_method, lr=args.lr)
#         scheduler = types.SimpleNamespace(step=lambda: 0) # 假设Adam不需要复杂的scheduler


#     log('Network: {}'.format(net_name), log_path)
#     log('Number of epoch: {}'.format(num_epoch), log_path)
#     log('Optimize method: {}'.format(optimize_method), log_path)
#     log('Learning rate: {}'.format(lr), log_path)
#     log('Meta learning rate: {}'.format(meta_lr), log_path)

#     if num_epoch_before != 0:
#         log('Loading state dict...', log_path)  
#         if save_best_test == False:
#             net.load_state_dict(torch.load(model_val_path))
#         else:
#             net.load_state_dict(torch.load(model_test_path))
#         for epoch in range(num_epoch_before):
#             scheduler.step()
#         log('Number of epoch-before: {}'.format(num_epoch_before), log_path)

#     log('Without close set classifier: {}'.format(without_cls), log_path)
#     log('Without binary classifier: {}'.format(without_bcls), log_path)
#     log('Share Parameter: {}'.format(share_param), log_path)

#     log('Start training...', log_path)  

#     recall = {
#         'va': 0,
#         'ta': 0,
#         'oscrc': 0, 
#         'oscrb': 0,
#         'bva': 0, 
#         'bvta': 0, 
#         'bvt': [],
#         'bta': 0, 
#         'btt': []
#     }

#     criterion = torch.nn.CrossEntropyLoss()
#     if without_cls:
#         criterion = lambda *args: 0
#     ovaloss = OVALoss()
#     if without_bcls:
#         ovaloss = lambda *args: 0


#     domain_index_list = [i for i in range(num_domain)]
#     domain_split = divide_list(shuffle_list(domain_index_list), task_d)
#     group_split = divide_list(shuffle_list(group_index_list), task_c)
#     task_pool = shuffle_list([(id, ig) for id in range(task_d) for ig in range(task_c)])
   
#    # --- 修改训练循环 ---
#     # fast_parameters 现在主要指 net.net (特征提取器) 的参数的快速版本
#     # net.parameters() 将包含 net.net 和 Linear_fw_Diffusion 的原始 (非 fast) 权重
#     # 我们需要确保梯度计算和更新正确地应用于 net.net
    
#     # 初始化 fast_parameters (现在主要针对特征提取器 net.net)
#     # 注意: Linear_fw_Diffusion 的权重是由扩散模型生成的，不直接参与这种 MAML 风格的梯度更新
#     # 但其父类 nn.Linear 的 self.weight 和 self.bias 仍然是 parameters() 的一部分
#     # 我们需要确保只为特征提取器创建 fast 副本，或者调整 MAML 更新逻辑

#     # 初始化快速参数列表，用于元学习
#     fast_parameters = list(net.parameters())
#     # 将所有参数的fast属性置为None
#     for weight in net.parameters():
#         weight.fast = None
#     # 清空梯度
#     net.zero_grad()
    
#     # 简化：让 fast_parameters 包含 net.net 的参数
#     fast_parameters = list(net.net.parameters()) 
#     for weight_fp in fast_parameters: # weight_fp 指 feature extractor parameter
#         weight_fp.fast = None
    
#     # MutiClassifier 中的 Linear_fw_Diffusion 层也继承自 nn.Linear，所以它们的 self.weight, self.bias
#     # 也是 net.parameters() 的一部分。我们需要处理它们。
#     # 方案1: 在MAML内循环中，这些原始权重不更新，只更新 fast 特征提取器权重。
#     # 方案2: 允许它们也像特征提取器一样有 .fast 版本，但它们的 .fast 值实际上由扩散模型设置。
#     # MEDIC的原始代码将所有 net.parameters() 都赋予了 .fast 属性。

#     # 我们让所有参数都有 .fast 属性，但 Linear_fw_Diffusion 的 .fast 由扩散模型填充
#     all_learnable_params = list(net.parameters()) # 这包含了 net.net 和 Linear_fw_Diffusion 的原始weight/bias
#     for weight in all_learnable_params:
#         weight.fast = None
    
#     net.zero_grad()
#     print(f"main前显存占用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
#     print(f"最大显存占用: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
#     print(torch.cuda.memory_summary())  # 详细内存分配情况

#     for epoch in range(num_epoch_before, num_epoch):
#         net.train()
#         task_count = 0
#         step_index = 0
#         input_sum_for_split = [] # 重命名以示清晰
#         label_sum_for_split = []

#         for id_val, ig_val in task_pool: # 重命名循环变量
#             domain_index = domain_split[id_val]
#             group_index = group_split[ig_val]
        
#             for i in domain_index:
#                 domain_specific_loader[i].keep(group_index)
#                 input_data, label_data = domain_specific_loader[i].next(batch_size=batch_size//len(domain_index))
#                 domain_specific_loader[i].reset()

#                 input_data = input_data.to(device)
#                 label_data = label_data.to(device)
#                 input_sum_for_split.append(input_data)
#                 label_sum_for_split.append(label_data)
            
#             task_count = (task_count + 1) % task_per_step[step_index]
#             if task_count == 0:
#                 current_split_input = torch.cat(input_sum_for_split, dim=0)
#                 current_split_label = torch.cat(label_sum_for_split, dim=0)

#                 # --- 1. 训练扩散模型 (每个 Linear_fw_Diffusion 层) ---
#                 if optimizer_diffusion is not None:
#                     # a. 生成条件向量
#                     condition_vec = generate_condition_vector_from_current_split(
#                         feature_extractor=net.net, # 使用特征提取器骨干
#                         current_split_data=current_split_input,
#                         current_split_labels=current_split_label,
#                         feature_dim=actual_feature_dim,
#                         device=device
#                     )

#                     # b. 确定目标权重 W0 (为每个扩散线性层)
#                     #    这里是一个简化的概念：实际中需要临时训练标准 nn.Linear
#                     #    我们假设已经有了一个函数 get_target_W0_for_linear(features, labels, linear_layer_config, loss_fn)
                    
#                     # 以 net.classifier 为例 (如果是 Linear_fw_Diffusion 类型)
#                     if isinstance(net.classifier, Linear_fw_Diffusion):
#                         # --- 获取 W0 (概念性) ---
#                         # 实际操作：用 net.net(current_split_input) 提取特征
#                         # 然后用这些特征和 current_split_label 训练一个临时的 nn.Linear 得到 W0_w, W0_b
#                         # 为简化，我们假设 W0_w, W0_b 是已知的或通过某种方式获取的
#                         # 例如，可以使用当前 net.classifier 的原始 .weight.data 和 .bias.data 作为近似目标
#                         # (这只是一个简化的占位符，实际应通过临时训练得到更理想的W0)
#                         with torch.no_grad():
#                             temp_features_for_w0 = net.net(current_split_input)
#                         # 此处应有临时训练逻辑... 假设我们得到了 target_w, target_b
#                         # 为了代码能运行，我们先用现有权重（这在逻辑上不是完美的 W0，但作为演示）
#                         target_w_cls = net.classifier.weight.data.clone().detach()
#                         target_b_cls = net.classifier.bias.data.clone().detach() if net.classifier.bias is not None else None
                        
#                         net.classifier.train_diffusion_step(
#                             target_clean_weight=target_w_cls,
#                             target_clean_bias=target_b_cls,
#                             condition_vector=condition_vec.squeeze(0), # 确保是 (condition_dim,)
#                             optimizer_diffusion=optimizer_diffusion
#                         )
                    
#                     # 对 net.b_classifier (以及 OVA 头，如果它们也是 Linear_fw_Diffusion) 重复此操作
#                     if isinstance(net.b_classifier, Linear_fw_Diffusion):
#                         with torch.no_grad():
#                              temp_features_for_w0_b = net.net(current_split_input)
#                         # 同样，这里是简化的 W0 获取
#                         target_w_bcls = net.b_classifier.weight.data.clone().detach()
#                         target_b_bcls = net.b_classifier.bias.data.clone().detach() if net.b_classifier.bias is not None else None

#                         net.b_classifier.train_diffusion_step(
#                             target_clean_weight=target_w_bcls,
#                             target_clean_bias=target_b_bcls,
#                             condition_vector=condition_vec.squeeze(0),
#                             optimizer_diffusion=optimizer_diffusion
#                         )
                
#                 # --- 2. MEDIC 元学习更新 (主要更新特征提取器 net.net) ---
#                 # a. 为当前分裂生成分类器权重
#                 condition_vec_for_medic_loss = generate_condition_vector_from_current_split(
#                     feature_extractor=net.net,
#                     current_split_data=current_split_input,
#                     current_split_labels=current_split_label,
#                     feature_dim=actual_feature_dim,
#                     device=device
#                 )
#                 if isinstance(net.classifier, Linear_fw_Diffusion):
#                     net.classifier.generate_and_set_weights(condition_vec_for_medic_loss.squeeze(0), device)
#                 if isinstance(net.b_classifier, Linear_fw_Diffusion):
#                     net.b_classifier.generate_and_set_weights(condition_vec_for_medic_loss.squeeze(0), device)
#                 # 对 OVA 头也进行类似操作

#                 # b. 执行 MEDIC 的前向传播和损失计算 (使用 .fast 权重)
#                 out_c, out_b = net.c_forward(x=current_split_input) # MutiClassifier 内部的 Linear_fw_Diffusion 会使用 .fast 权重
#                 out_b = out_b.view(out_b.size(0), net.num_classes, 2) # 确保与 OVALoss 匹配
#                                                                    # 或 out_b.view(out_b.size(0), 2, -1)
                
#                 loss = criterion(out_c, current_split_label) + ovaloss(out_b, current_split_label)
#                 loss *= task_per_step[step_index]
                
#                 # c. 计算梯度 (针对参与元学习的参数，主要是 net.net)
#                 #   原始代码的 fast_parameters 指向 net.parameters()
#                 #   我们需要确保这里的 grad 只影响我们希望通过元学习优化的参数
#                 #   对于 Linear_fw_Diffusion，其 .weight 和 .bias 是 nn.Parameter，也会在 net.parameters() 中
#                 #   但它们的 .fast 属性是由扩散模型设置的，它们不应该通过 MEDIC 的 meta_lr * grad[k] 更新
                
#                 # 获取参与 MAML 更新的参数 (特征提取器 + Linear_fw_Diffusion 的原始weight/bias，但不包括扩散模型内部参数)
#                 # 注意：这里的 fast_parameters 应该与之前定义 all_learnable_params 的逻辑一致
#                 maml_update_params = []
#                 for p_group in optimizer.param_groups: # optimizer 只包含 net.net 参数
#                     maml_update_params.extend(p_group['params'])
#                 # 如果 Linear_fw_Diffusion 的原始 weight/bias 也想通过 MAML 框架更新（不推荐，因为它们被扩散模型管理 .fast），则需调整

#                 current_grads = torch.autograd.grad(loss, maml_update_params, create_graph=False, allow_unused=True)
#                 current_grads = [g.detach() if g is not None else None for g in current_grads]

#                 # d. 更新特征提取器参数的 .fast 版本
#                 # 这里的 fast_parameters_for_maml 实际上是指 net.net.parameters()
#                 # 我们需要一种方式将 net.net 的参数和 current_grads 对应起来
                
#                 # 遍历 net.net 的参数进行更新
#                 idx = 0
#                 for net_param in net.net.parameters(): # 只更新特征提取器的 fast 权重
#                     if net_param.requires_grad: # 确保参数是可训练的
#                         if current_grads[idx] is not None:
#                             if net_param.fast is None:
#                                 net_param.fast = net_param.data - meta_lr * current_grads[idx]
#                             else:
#                                 net_param.fast = net_param.fast.data - meta_lr * current_grads[idx]
#                         idx += 1
#                 # 对于 Linear_fw_Diffusion 的 self.weight 和 self.bias，它们的 .fast 已经由扩散模型设置
#                 # 所以它们不参与这里的减去梯度的步骤
                                
#                 input_sum_for_split = []
#                 label_sum_for_split = []
#                 step_index += 1

#         # --- MEDIC 外层优化器步骤 (更新 net.net 的原始权重) ---
#         optimizer.zero_grad()
#         # 只有 net.net 的参数需要计算 .grad 用于 optimizer.step()
#         for net_param in net.net.parameters():
#             if net_param.requires_grad:
#                 if net_param.fast is not None:
#                     net_param.grad = (net_param.data - net_param.fast) / meta_lr
#                     net_param.fast = None # 清除 net.net 的 fast 权重
#                 else:
#                     net_param.grad = None # 如果没有 fast 权重（例如，由于 grad 为 None），则梯度也为 None

#         # Linear_fw_Diffusion 层的 .fast 权重也需要清除，以便下次迭代重新生成
#         if isinstance(net.classifier, Linear_fw_Diffusion):
#             net.classifier.weight.fast = None
#             if net.classifier.bias is not None: net.classifier.bias.fast = None
#         if isinstance(net.b_classifier, Linear_fw_Diffusion):
#             net.b_classifier.weight.fast = None
#             if net.b_classifier.bias is not None: net.b_classifier.bias.fast = None
#         # ... 对 OVA 头也进行类似操作 ...

#         optimizer.step() # 更新 net.net 的参数

#         # 重新打乱任务池等
#         domain_split = divide_list(shuffle_list(domain_index_list), task_d)
#         group_split = divide_list(shuffle_list(group_index_list), task_c) 
#         task_pool = shuffle_list([(id_val, ig_val) for id_val in range(task_d) for ig_val in range(task_c)]) 

#         # fast_parameters = list(net.parameters()) # 这一行可能不再需要，或者需要重新定义其含义
#         # for weight in net.parameters(): # 这个循环也需要调整
#         #     weight.fast = None          # .fast 清除已在上面完成
#         # net.zero_grad() # 在 optimizer.step() 后通常会自动清零，或在下一次迭代前清零

#         if (epoch+1) % eval_step == 0:      
       
#             net.eval()  

#             recall['va'], recall['ta'], recall['oscrc'], recall['oscrb'] = eval_all(net, val_k, test_k, test_u, log_path, epoch, device)
#             update_recall(net, recall, log_path, model_val_path)

            
#         if epoch+1 == renovate_step:
#                 log("Reset accuracy history...", log_path)
#                 recall['bva'] = 0
#                 recall['bvta'] = 0
#                 recall['bvt'] = []
#                 recall['bta'] = 0
#                 recall['btt'] = []

#         scheduler.step()