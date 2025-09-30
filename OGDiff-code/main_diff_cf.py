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
#     parser.add_argument('--batch-size', type=int, default=16)
#     parser.add_argument('--task-d', type=int, default=3)
#     parser.add_argument('--task-c', type=int, default=3)
#     parser.add_argument('--task-per-step', nargs='+', type=int, default=[3, 3, 3])

#     parser.add_argument('--net-name', default='resnet18')
#     parser.add_argument('--optimize-method', default="SGD")
#     parser.add_argument('--schedule-method', default='StepLR')
#     parser.add_argument('--num-epoch', type=int, default=6000)
#     parser.add_argument('--eval-step', type=int, default=50)
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

#     parser.add_argument('--load-pretrained-path', type=str, default='./experiment/model/val/pacs_resnet18.tar', 
#                         help='Path to the pre-trained .tar model to use for diffusion training')
#     parser.add_argument('--diffusion-lr', type=float, default=1e-4, help='扩散模型优化器的学习率')
#     parser.add_argument('--diffusion-timesteps', type=int, default=100, help='扩散过程的时间步数')
#     args = parser.parse_args()

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
#             device = torch.device("cuda:3") 
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
#     model_save_path = os.path.join(save_dir, 'model', 'diffusion', save_name + '.tar')

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
#         # 替换为 SGD 优化器:
#         diffusion_lr = getattr(args, 'diffusion_lr') # SGD通常需要比AdamW稍大的学习率
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
#     # main_model_params = list(model_accessor.net.parameters()) # 确保只获取骨干网络的参数
    
#     # optimize_method = args.optimize_method # 从args获取
#     # lr = args.lr # 从args获取
#     # nesterov = args.nesterov # 从args获取
#     # schedule_method = args.schedule_method # 从args获取
#     # num_epoch = args.num_epoch # 从args获取

#     # if args.optimize_method == 'SGD':
#     #     optimizer = get_optimizer(params=main_model_params, instr=args.optimize_method, lr=args.lr, nesterov=args.nesterov) 
#     #     scheduler = get_scheduler(optimizer=optimizer, instr=args.schedule_method, step_size=int(args.num_epoch*0.8), gamma=0.1)
#     # elif args.optimize_method == 'Adam': # 你主优化器用的是Adam，这里保持不变，只改扩散模型的
#     #     optimizer = get_optimizer(params=main_model_params, instr=args.optimize_method, lr=args.lr)
#     #     scheduler = types.SimpleNamespace(step=lambda: 0) # 假设Adam不需要复杂的scheduler


#     log('Network: {}'.format(net_name), log_path)
#     log('Number of epoch: {}'.format(num_epoch), log_path)
#     log('Optimize method: {}'.format(optimize_method), log_path)
#     log('Learning rate: {}'.format(lr), log_path)
#     log('Meta learning rate: {}'.format(meta_lr), log_path)

#     # if num_epoch_before != 0:
#     #     log('Loading state dict...', log_path)  
#     #     if save_best_test == False:
#     #         net.load_state_dict(torch.load(model_val_path))
#     #     else:
#     #         net.load_state_dict(torch.load(model_test_path))
#     #     for epoch in range(num_epoch_before):
#     #         scheduler.step()
#     #     log('Number of epoch-before: {}'.format(num_epoch_before), log_path)

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
   
#     # --- 加载预训练权重 ---
#     log(f"Loading pre-trained weights from: {args.load_pretrained_path}", log_path)
#     state_dict = torch.load(args.load_pretrained_path, map_location=device, weights_only=True)
#     # 使用 strict=False 来加载，这将加载所有匹配的键，并忽略所有缺失的键（即扩散模型参数）
#     missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
#     log(f"Loaded state_dict with strict=False. Found {len(missing_keys)} missing keys (expected for diffusion model).", log_path)
#     if unexpected_keys: log(f"Warning: Found unexpected keys in state_dict: {unexpected_keys}", log_path)

#     # --- 冻结特征提取器 ---
#     log("Freezing feature extractor (net.net)...", log_path)
#     net.net.eval() # 设置为评估模式
#     for param in net.net.parameters():
#         param.requires_grad = False # 不计算梯度

#     # --- 提取并保存“理想”的分类器权重作为训练目标 ---
#     log("Extracting target weights for diffusion model training...", log_path)
#     target_weights_store = {}
#     if isinstance(net.classifier, Linear_fw_Diffusion):
#         target_weights_store['classifier'] = {
#             'weight': net.classifier.weight.data.clone().detach(),
#             'bias': net.classifier.bias.data.clone().detach() if net.classifier.bias is not None else None
#         }
#     if isinstance(net.b_classifier, Linear_fw_Diffusion):
#         target_weights_store['b_classifier'] = {
#             'weight': net.b_classifier.weight.data.clone().detach(),
#             'bias': net.b_classifier.bias.data.clone().detach() if net.b_classifier.bias is not None else None
#         }
    
#     # --- 【调试代码 1：检查目标权重】 ---
#     log("--- Debugging: Checking Target Weights 调试代码 1:检查目标权重 ---", log_path)
#     if 'classifier' in target_weights_store:
#         w_norm = torch.linalg.norm(target_weights_store['classifier']['weight']).item()
#         b_norm = torch.linalg.norm(target_weights_store['classifier']['bias']).item() if target_weights_store['classifier']['bias'] is not None else 0
#         log(f"[Target] classifier weight norm: {w_norm:.4f}, bias norm: {b_norm:.4f}", log_path)
#     if 'b_classifier' in target_weights_store:
#         w_norm = torch.linalg.norm(target_weights_store['b_classifier']['weight']).item()
#         b_norm = torch.linalg.norm(target_weights_store['b_classifier']['bias']).item() if target_weights_store['b_classifier']['bias'] is not None else 0
#         log(f"[Target] b_classifier weight norm: {w_norm:.4f}, bias norm: {b_norm:.4f}", log_path)
#     log("--- End Debugging ---", log_path)
#     # --- 【结束调试代码 1】 ---

#     # --- 训练循环 ---
#     log('Start training diffusion models...', log_path)
    
#     best_val_oscr = 0.0 # 用于保存最佳模型

#     # # --- 【调试代码 3：单批次过拟合测试】 ---
#     # log("--- Debugging: Starting single-batch overfitting test... ---", log_path)
#     # # 从数据加载器中取出一个固定的批次
#     # fixed_input_batch, fixed_label_batch, *_ = next(iter(train_loader))
#     # fixed_input_batch = fixed_input_batch.to(device)
#     # fixed_label_batch = fixed_label_batch.to(device)

#     # # 为这个固定批次生成一个固定的条件向量
#     # fixed_condition_vec = generate_condition_vector_from_current_split(
#     #     feature_extractor=net.net,
#     #     current_split_data=fixed_input_batch,
#     #     current_split_labels=fixed_label_batch,
#     #     feature_dim=actual_feature_dim,
#     #     device=device
#     # )
#     # log("Fixed batch and condition vector created for overfitting test.", log_path)
#     # # --- 【结束调试代码 3】 ---


#     # for epoch in range(args.num_epoch):
#     #     net.train()
#     #     net.net.eval()

#     #     total_loss_epoch = 0
#     #     # --- 【调试代码 3 修改】: 不再从数据加载器循环，而是直接使用固定批次 ---
#     #     # for i, (input_data, label_data) in enumerate(train_loader):
#     #         # a. 直接使用固定的条件向量
#     #     condition_vec = fixed_condition_vec

#     #     # b. 训练每个扩散线性层
#     #     current_loss = 0
#     #     if 'classifier' in target_weights_store:
#     #         loss_cls = net.classifier.train_diffusion_step(
#     #             target_clean_weight=target_weights_store['classifier']['weight'],
#     #             target_clean_bias=target_weights_store['classifier']['bias'],
#     #             condition_vector=condition_vec.squeeze(0),
#     #             optimizer_diffusion=optimizer_diffusion
#     #         )
#     #         current_loss += loss_cls

#     #     if 'b_classifier' in target_weights_store:
#     #         loss_bcls = net.b_classifier.train_diffusion_step(
#     #             target_clean_weight=target_weights_store['b_classifier']['weight'],
#     #             target_clean_bias=target_weights_store['b_classifier']['bias'],
#     #             condition_vector=condition_vec.squeeze(0),
#     #             optimizer_diffusion=optimizer_diffusion
#     #         )
#     #         current_loss += loss_bcls
        
#     #     total_loss_epoch = current_loss # 因为只有一个步骤
        
#     #     log(f"Epoch [{epoch+1}/{args.num_epoch}], Overfitting Denoising Loss: {total_loss_epoch:.6f}", log_path)
#     #    # --- 【结束调试代码 3 修改】 ---

#     criterion = torch.nn.CrossEntropyLoss()
#     ovaloss = OVALoss()

#     # ... (优化器设置等代码保持不变) ...

#     def get_target_W0_for_split(features, labels, num_classes, linear_layer_config, device, criterion, ovaloss):
#         """
#         为一个特定的数据分裂，通过短暂训练一个临时线性层来获取“理想”的目标权重。
#         """
#         # 创建一个临时的、标准的线性分类器
#         temp_classifier = torch.nn.Linear(linear_layer_config['in_features'], linear_layer_config['out_features']).to(device)
        
#         # 为这个临时分类器创建一个简单的优化器
#         optimizer_temp = torch.optim.Adam(temp_classifier.parameters(), lr=1e-3)

#         # 进行少量迭代来拟合当前批次的数据
#         for _ in range(5): # 迭代次数是一个可以调整的超参数
#             # 如果是为 b_classifier 获取权重 (One-vs-All)
#             if linear_layer_config['out_features'] == num_classes * 2:
#                 out_b_temp = temp_classifier(features)
#                 # 修正 reshape 以匹配 OVALoss 的期望输入格式
#                 out_b_temp_reshaped = out_b_temp.view(out_b_temp.size(0), 2, -1)
#                 loss = ovaloss(out_b_temp_reshaped, labels)
#             # 如果是为 classifier 获取权重 (标准分类)
#             else:
#                 out_c_temp = temp_classifier(features)
#                 loss = criterion(out_c_temp, labels)

#             optimizer_temp.zero_grad()
#             loss.backward()
#             optimizer_temp.step()
        
#         # 返回训练好的临时分类器的权重和偏置，作为本次迭代的“标准答案”
#         return temp_classifier.weight.data.clone().detach(), temp_classifier.bias.data.clone().detach()
#     # --- 修改后的训练循环 ---
#     for epoch in range(args.num_epoch):
#         net.train()
#         net.net.eval()

#         total_loss_epoch = 0
#         step_index = 0
#         task_count = 0
#         input_sum_for_split = [] # 重命名以示清晰
#         label_sum_for_split = []
#         # ... (数据加载和任务池循环保持不变)
#         for id, ig in task_pool:
#             domain_index = domain_split[id]
#             group_index = group_split[ig]
        
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

#                 # a. 生成条件向量 (与之前相同)
#                 condition_vec = generate_condition_vector_from_current_split(
#                     feature_extractor=net.net,
#                     current_split_data=current_split_input,
#                     current_split_labels=current_split_label,
#                     feature_dim=actual_feature_dim,
#                     device=device
#                 )

#                 # --- 【核心修改点】为当前数据分裂动态生成目标权重 W0 ---
#                 with torch.no_grad(): # 特征提取不需要梯度
#                     current_features = net.net(current_split_input)
                
#                 # 为 classifier (闭集分类器) 动态生成目标 W0
#                 target_w_cls, target_b_cls = get_target_W0_for_split(
#                     features=current_features.detach(), # 传入分离后的特征
#                     labels=current_split_label,
#                     num_classes=num_classes,
#                     linear_layer_config={'in_features': actual_feature_dim, 'out_features': num_classes},
#                     device=device,
#                     criterion=criterion,
#                     ovaloss=lambda *args, **kwargs: 0 # 此时不使用 ovaloss
#                 )
                
#                 # 为 b_classifier (开集分类器) 动态生成目标 W0
#                 target_w_bcls, target_b_bcls = get_target_W0_for_split(
#                     features=current_features.detach(),
#                     labels=current_split_label,
#                     num_classes=num_classes,
#                     linear_layer_config={'in_features': actual_feature_dim, 'out_features': num_classes * 2},
#                     device=device,
#                     criterion=lambda *args, **kwargs: 0, # 此时不使用 criterion
#                     ovaloss=ovaloss
#                 )
                
#                 # c. 使用动态生成的目标权重来训练扩散模型
#                 current_loss = 0
#                 if isinstance(net.classifier, Linear_fw_Diffusion):
#                     loss_cls = net.classifier.train_diffusion_step(
#                         target_clean_weight=target_w_cls,
#                         target_clean_bias=target_b_cls,
#                         condition_vector=condition_vec.squeeze(0),
#                         optimizer_diffusion=optimizer_diffusion
#                     )
#                     current_loss += loss_cls

#                 if isinstance(net.b_classifier, Linear_fw_Diffusion):
#                     loss_bcls = net.b_classifier.train_diffusion_step(
#                         target_clean_weight=target_w_bcls,
#                         target_clean_bias=target_b_bcls,
#                         condition_vector=condition_vec.squeeze(0),
#                         optimizer_diffusion=optimizer_diffusion
#                     )
#                     current_loss += loss_bcls

#                 total_loss_epoch += current_loss
                
#                 # 清空数据累加器
#                 input_sum_for_split = []
#                 label_sum_for_split = []
#                 step_index += 1 # 别忘了更新 step_index
        
#         avg_loss = total_loss_epoch / step_index if step_index > 0 else 0
#         if (epoch + 1) % 10 == 0:
#             log(f"Epoch [{epoch+1}/{args.num_epoch}], Average Denoising Loss: {avg_loss:.4f}", log_path)
#         # log(f"Epoch [{epoch+1}/{args.num_epoch}], Average Denoising Loss: {avg_loss:.4f}", log_path)

#         # --- 评估和保存模型 ---
#         if (epoch + 1) % args.eval_step == 0:
#             net.eval() # 评估时整个模型都设为 eval 模式

#             # eval_all 函数需要修改，以接收一个标志位来触发权重生成
#             # 这里简化地假设 eval_all 内部会处理
#             # 并且会为验证集/测试集生成一个合适的条件向量
#             _, _, val_oscrc, val_oscrb, _, _, _, _, _, _ = eval_all_diff(
#                 net=net,
#                 val_k=val_k,
#                 test_k=test_k, # 传递给 eval_all 以便其内部评估
#                 test_u=test_u,
#                 log_path=log_path,
#                 epoch=epoch,
#                 device=device,
#                 condition_generator_fn=generate_condition_vector_from_current_split,
#                 feature_dim_for_cond=actual_feature_dim,
#                 needs_diffusion_weights=True
#             )
            
#             # 保存性能最好的模型（这里只保存扩散模型的参数）
#             current_val_oscr = (val_oscrc + val_oscrb) / 2
#             if current_val_oscr > best_val_oscr:
#                 best_val_oscr = current_val_oscr
#                 log(f"New best validation OSCR: {best_val_oscr:.4f}. Saving model to {model_save_path}", log_path)
                
#                 # 我们只关心扩散模型的参数，因为特征提取器是固定的
#                 diffusion_state_dict = {}
#                 if isinstance(net.classifier, Linear_fw_Diffusion):
#                     diffusion_state_dict['classifier_diffusion'] = net.classifier.diffusion_model.state_dict()
#                 if isinstance(net.b_classifier, Linear_fw_Diffusion):
#                     diffusion_state_dict['b_classifier_diffusion'] = net.b_classifier.diffusion_model.state_dict()
                
#                 torch.save(diffusion_state_dict, model_save_path)

#     log("Training finished.", log_path)

