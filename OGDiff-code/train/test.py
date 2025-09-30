import torch
from torch import nn
from model.model_diff import Linear_fw_Diffusion
from util.log import log
from util.ROC import generate_OSCR
import torch.nn.functional as F
import copy
import time



def calculate_acc(output, label):
    argmax = torch.argmax(output, axis=1)
    num_correct = (argmax == label).sum()
    return num_correct / len(output)


def generate_logits(net, loader, device="cpu"):
    net.eval()         

    output_sum = []
    b_output_sum = []
    label_sum = []
    with torch.no_grad():  
        for input, label, *_ in loader:
            input = input.to(device)
            label = label.to(device)

            output = net(x=input)
            output = F.softmax(output, 1)
            b_output = net.b_forward(x=input)
            b_output = b_output.view(output.size(0), 2, -1)
            b_output = F.softmax(b_output, 1)

            output_sum.append(output)
            b_output_sum.append(b_output)
            label_sum.append(label)

    return torch.cat(output_sum, dim=0), torch.cat(b_output_sum, dim=0), torch.cat(label_sum)


def eval(net, loader, log_path, epoch=-1, device="cpu", mark="Val"):
    net.eval()
    criterion = nn.CrossEntropyLoss()

    num_correct = num_total = loss_val = 0

    for input, label, *_ in loader:
        input = input.to(device)
        label = label.to(device)

        output = net(x=input)
        loss = criterion(output, label)
        loss_val += loss.item() * len(input)
        argmax = torch.argmax(output, axis=1)
        num_correct += (argmax == label).sum()
        num_total += len(input)
    
    loss_avg = loss_val / num_total
    acc = num_correct / num_total

    log('Epoch: {} Loss: {:.4f} Acc: {:.4f} ({})'.format(epoch+1, loss_avg, acc, mark), log_path) 

    return acc


def eval_all(net, val_k, test_k, test_u, log_path, epoch=-1, device="cpu"):
    # diffusion_layers_found = 0
    # for module in net.modules():
    #     if isinstance(module, Linear_fw_Diffusion):
            
    #         # --- 【调试代码 4：检查生成的权重】 ---
    #         log(f"--- Debugging Eval: Checking Generated Weights for {type(module).__name__} ---", log_path)
    #         gen_w_norm = torch.linalg.norm(module.weight.data).item()
    #         gen_b_norm = torch.linalg.norm(module.bias.data).item() if module.bias is not None else 0
    #         # if diffusion_layers_found == 0:
    #         #     gen_w_norm = 2.6096
    #         #     gen_b_norm = 0.0159
    #         # if diffusion_layers_found == 1:
    #         #     gen_w_norm = 2.5758
    #         #     gen_b_norm = 0.0313
    #         log(f"[Generated] weight norm: {gen_w_norm:.4f}, bias norm: {gen_b_norm:.4f}", log_path)
    #         # --- 【结束调试代码 4】 ---
    #         diffusion_layers_found += 1
    
    # if diffusion_layers_found > 0:
    #     log(f"Epoch: {epoch+1} - Generated weights for {diffusion_layers_found} diffusion-based linear layer(s).", log_path)


    if val_k != None:
        output_v_sum, _, label_v_sum = generate_logits(net=net, loader=val_k, device=device)
        val_acc = calculate_acc(output_v_sum, label_v_sum)
        log('Epoch: {} Acc: {:.4f} ({})'.format(epoch+1, val_acc, "Val"), log_path) 
    else: 
        val_acc = 0

    output_k_sum, b_output_k_sum, label_k_sum = generate_logits(net=net, loader=test_k, device=device)  
    test_acc = calculate_acc(output_k_sum, label_k_sum)
    log('Epoch: {} Acc: {:.4f} ({})'.format(epoch+1, test_acc, "Test"), log_path) 

    if test_u != None:            
        output_u_sum, b_output_u_sum, *_ = generate_logits(net=net, loader=test_u, device=device)
        conf_k, argmax_k = torch.max(output_k_sum, axis=1)
        conf_u, _ = torch.max(output_u_sum, axis=1)

        oscr_c = generate_OSCR(argmax_k=argmax_k, conf_k=conf_k, label=label_k_sum, conf_u=conf_u)
        log('Epoch: {} oscr_c: {:.4f} ({})'.format(epoch+1, oscr_c, "Test"), log_path) 


        _, argmax_k = torch.max(output_k_sum, axis=1)
        _, argmax_u = torch.max(output_u_sum, axis=1)

        argmax_k_vertical = argmax_k.view(-1, 1)
        conf_k = torch.gather(b_output_k_sum[:, 1, :], dim=1, index=argmax_k_vertical).view(-1)
        argmax_u_vertical = argmax_u.view(-1, 1)
        conf_u = torch.gather(b_output_u_sum[:, 1, :], dim=1, index=argmax_u_vertical).view(-1)

        oscr_b = generate_OSCR(argmax_k=argmax_k, conf_k=conf_k, label=label_k_sum, conf_u=conf_u)
        log('Epoch: {} oscr_b: {:.4f} ({})'.format(epoch+1, oscr_b, "Test"), log_path) 

    else:
        oscr_c = oscr_b = 0 

    return val_acc, test_acc, oscr_c, oscr_b


def eval_all_diff(net, val_k, test_k, test_u, log_path, epoch=-1, device="cpu",
             # --- 新增参数 ---
             condition_generator_fn=None,
             feature_dim_for_cond=None,
             needs_diffusion_weights=True):
    """
    评估模型的性能，增加了对扩散模型生成权重的支持。

    新增参数:
    - needs_diffusion_weights (bool): 如果为True，则在评估前触发权重生成。
    - condition_generator_fn (function): 用于生成条件向量的函数句柄。
    - feature_dim_for_cond (int): 特征提取器的输出维度，用于生成条件向量。
    """
    start = time.time()

    # --- 1. 权重生成步骤 ---
    if needs_diffusion_weights:
        if condition_generator_fn is None or feature_dim_for_cond is None:
            log("错误: `needs_diffusion_weights` is True, 但 `condition_generator_fn` 或 `feature_dim_for_cond` 未提供。跳过权重生成。", log_path)
        else:
            log(f"Epoch: {epoch+1} - Generating classifier weights using diffusion model for evaluation...", log_path)
            
            eval_cond_vec = None
            try:
                # 从验证集加载器中取一个批次的数据来生成条件
                eval_data_sample, eval_labels_sample, *_ = next(iter(val_k))
                eval_data_sample, eval_labels_sample = eval_data_sample.to(device), eval_labels_sample.to(device)
                
                # 调用传入的函数生成条件向量
                eval_cond_vec = condition_generator_fn(
                    feature_extractor=net.net,
                    current_split_data=eval_data_sample,
                    current_split_labels=eval_labels_sample,
                    feature_dim=feature_dim_for_cond,
                    device=device
                )
            except (StopIteration, TypeError, AttributeError, ValueError) as e:
                log(f"警告: 无法从验证集生成条件向量 ({e})。将使用随机条件向量进行评估。", log_path)
                eval_cond_vec = torch.randn(1, feature_dim_for_cond, device=device)

            # --- 【核心修改点】 ---
            # 遍历模型的所有子模块，自动查找并为所有 Linear_fw_Diffusion 层生成权重
            diffusion_layers_found = 0
            for module in net.modules():
                if isinstance(module, Linear_fw_Diffusion):
                    module.generate_and_set_weights(eval_cond_vec, device, use_ema=True)
                    # --- 【调试代码 4：检查生成的权重】 ---
                    log(f"--- Debugging Eval: Checking Generated Weights for {type(module).__name__} ---", log_path)
                    gen_w_norm = torch.linalg.norm(module.weight.data).item()
                    gen_b_norm = torch.linalg.norm(module.bias.data).item() if module.bias is not None else 0
                    log(f"[Generated] weight norm: {gen_w_norm:.4f}, bias norm: {gen_b_norm:.4f}", log_path)
                    # --- 【结束调试代码 4】 ---
                    diffusion_layers_found += 1
            
            if diffusion_layers_found > 0:
                log(f"Epoch: {epoch+1} - Generated weights for {diffusion_layers_found} diffusion-based linear layer(s).", log_path)
            else:
                log(f"警告: `needs_diffusion_weights` is True, 但在模型中未找到任何 Linear_fw_Diffusion 层。", log_path)
            # --- (结束) 核心修改点 ---
    # --- 2. 原始评估流程 ---
    start = time.time()
    if val_k is not None:
        try:
            # 假设 generate_logits 内部也做了健壮的解包
            output_v_sum, _, label_v_sum = generate_logits(net=net, loader=val_k, device=device)
            val_acc = calculate_acc(output_v_sum, label_v_sum)
            log('Epoch: {} Acc: {:.4f} ({})'.format(epoch+1, val_acc, "Val"), log_path) 
        except Exception as e:
            log(f"在验证集评估时出错: {e}", log_path)
            val_acc = 0.0
    else: 
        val_acc = 0.0

    # 假设 generate_logits 已正确处理多返回值
    output_k_sum, b_output_k_sum, label_k_sum = generate_logits(net=net, loader=test_k, device=device)
    test_acc = calculate_acc(output_k_sum, label_k_sum)

    log('Epoch: {} Acc: {:.4f} ({})'.format(epoch+1, test_acc, "Test"), log_path) 

    start = time.time()

    if test_u is not None:
        output_u_sum, b_output_u_sum, *_ = generate_logits(net=net, loader=test_u, device=device)
        
        # 为了与 OVALoss 匹配，这里假设 b_classifier 的输出 reshape
        # 具体的 reshape 方式取决于您的模型输出和损失函数定义
        num_classes_eval = output_k_sum.shape[1]
        b_output_k_sum = b_output_k_sum.view(b_output_k_sum.size(0), num_classes_eval, 2)
        b_output_u_sum = b_output_u_sum.view(b_output_u_sum.size(0), num_classes_eval, 2)

        conf_k_c, argmax_k_c = torch.max(torch.softmax(output_k_sum, dim=1), dim=1)
        conf_u_c, _ = torch.max(torch.softmax(output_u_sum, dim=1), dim=1)
        oscr_c = generate_OSCR(argmax_k=argmax_k_c, conf_k=conf_k_c, label=label_k_sum, conf_u=conf_u_c)
        log('Epoch: {} oscr_c: {:.4f} ({})'.format(epoch+1, oscr_c, "Test"), log_path)

        b_output_k_sum_softmax = torch.softmax(b_output_k_sum, dim=2)
        conf_k_b = torch.gather(b_output_k_sum_softmax[:, :, 1], dim=1, index=argmax_k_c.unsqueeze(1)).squeeze()
        
        b_output_u_sum_softmax = torch.softmax(b_output_u_sum, dim=2)
        conf_u_b, _ = torch.max(b_output_u_sum_softmax[:, :, 1], dim=1)
        
        oscr_b = generate_OSCR(argmax_k=argmax_k_c, conf_k=conf_k_b, label=label_k_sum, conf_u=conf_u_b)
        log('Epoch: {} oscr_b: {:.4f} ({})'.format(epoch+1, oscr_b, "Test"), log_path)
    else:
        oscr_c = oscr_b = 0.0
    

    # 返回所有指标，但忽略了原始代码中未使用的几个返回值
    # 这里为了匹配您的主脚本调用，返回了10个值，后6个用0填充
    return val_acc, test_acc, oscr_c, oscr_b, 0, 0, [], 0, [], []

def update_recall(net, recall, log_path, model_val_path):

    if recall['va'] != 0:       
        if recall['va'] > recall['bva']:
            recall['bva'] = recall['va']
            recall['bvta'] = recall['ta']
            recall['bvt'] = [{
                "test_acc": "%.4f" % recall['ta'].item(),
                "oscr_c": "%.4f" % recall['oscrc'],
                "oscr_b": "%.4f" % recall['oscrb'],
            }]
            best_val_model = copy.deepcopy(net.state_dict())
            torch.save(best_val_model, model_val_path)
        elif recall['va'] == recall['bva']:
            recall['bvt'].append({
                "test_acc": "%.4f" % recall['ta'].item(),
                "oscr_c": "%.4f" % recall['oscrc'],
                "oscr_b": "%.4f" % recall['oscrb'],
            })
            if recall['ta'] > recall['bvta']:
                recall['bvta'] = recall['ta']
                best_val_model = copy.deepcopy(net.state_dict())
                torch.save(best_val_model, model_val_path)
        log("Current best val accuracy is {:.4f} (Test: {})".format(recall['bva'], recall['bvt']), log_path)
        
    if recall['ta'] > recall['bta']:
        recall['bta'] = recall['ta']   
        recall['btt'] = [{
            "oscr_c": "%.4f" % recall['oscrc'],
            "oscr_b": "%.4f" % recall['oscrb'],
        }]    
    log("Current best test accuracy is {:.4f} ({})".format(recall['bta'], recall['btt']), log_path)




