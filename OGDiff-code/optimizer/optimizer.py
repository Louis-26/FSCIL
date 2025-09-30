from torch import optim

def get_optimizer(net, instr="SGD", **options):    
    optimizer_map = {
        "SGD": (optim.SGD, {
            "params": net.parameters(),
            "lr": None, # <require value>
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "nesterov": False
        }), 
        "Adam": (optim.Adam, {
            "params": net.parameters(),
            "lr": None, # <require value>
            "weight_decay": 0,
        })
    }
    
    optimizer, args = optimizer_map[instr]
    args.update(options)

    return optimizer(**args)

def get_scheduler(optimizer, instr="StepLR", **options):
    scheduler_map = {
        "StepLR": (optim.lr_scheduler.StepLR, {
            "optimizer": optimizer,
            "step_size": None, # <require value>
            "gamma": 0.1
        })
    }

    scheduler, args = scheduler_map[instr]
    args.update(options)

    return scheduler(**args)

# from torch import optim

# def get_optimizer(params, instr="SGD", **options): # 将 net 修改为 params
#     """
#     获取优化器实例。

#     参数:
#     - params (iterable): 模型参数的可迭代对象 (例如 model.parameters() 或特定的参数列表)。
#     - instr (str): 优化器类型的字符串 ("SGD", "Adam")。
#     - **options: 传递给优化器构造函数的其他选项 (例如 lr, nesterov)。
#     """
#     optimizer_map = {
#         "SGD": (optim.SGD, {
#             "params": params, # 修改此处
#             "lr": None, # <require value>
#             "momentum": 0.9,
#             "weight_decay": 0.0005,
#             "nesterov": False
#         }),
#         "Adam": (optim.Adam, {
#             "params": params, # 修改此处
#             "lr": None, # <require value>
#             "weight_decay": 0, # Adam 默认的 weight_decay 通常是0，但可以按需调整
#         }),
#         # 您可以根据需要添加更多优化器，例如 AdamW
#         "AdamW": (optim.AdamW, {
#             "params": params,
#             "lr": None, # <require value>
#             "weight_decay": 0.01 # AdamW 通常使用一个小的 weight_decay
#         })
#     }

#     if instr not in optimizer_map:
#         raise ValueError(f"未知的优化器类型: {instr}. 可选的有: {list(optimizer_map.keys())}")

#     optimizer_class, args = optimizer_map[instr]
    
#     # 确保必要的参数 (如 lr) 被提供
#     if args.get("lr") is None and "lr" not in options:
#         raise ValueError(f"优化器 {instr} 需要提供学习率 'lr'")
        
#     args.update(options)

#     return optimizer_class(**args)

# def get_scheduler(optimizer, instr="StepLR", **options):
#     """
#     获取学习率调度器实例。

#     参数:
#     - optimizer (torch.optim.Optimizer): 关联的优化器。
#     - instr (str): 调度器类型的字符串 ("StepLR")。
#     - **options: 传递给调度器构造函数的其他选项 (例如 step_size, gamma)。
#     """
#     scheduler_map = {
#         "StepLR": (optim.lr_scheduler.StepLR, {
#             "optimizer": optimizer,
#             "step_size": None, # <require value>
#             "gamma": 0.1
#         }),
#         "CosineAnnealingLR": (optim.lr_scheduler.CosineAnnealingLR, {
#             "optimizer": optimizer,
#             "T_max": None, # <require value> (例如，总的训练 epochs 或 steps)
#             "eta_min": 0   # 最小学习率
#         })
#         # 您可以根据需要添加更多调度器
#     }

#     if instr not in scheduler_map:
#         raise ValueError(f"未知的调度器类型: {instr}. 可选的有: {list(scheduler_map.keys())}")

#     scheduler_class, args = scheduler_map[instr]

#     # 确保必要的参数被提供
#     if instr == "StepLR" and args.get("step_size") is None and "step_size" not in options:
#         raise ValueError(f"调度器 StepLR 需要提供 'step_size'")
#     if instr == "CosineAnnealingLR" and args.get("T_max") is None and "T_max" not in options:
#         raise ValueError(f"调度器 CosineAnnealingLR 需要提供 'T_max'")

#     args.update(options)

#     return scheduler_class(**args)