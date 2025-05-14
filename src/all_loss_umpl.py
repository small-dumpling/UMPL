import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_rmse_lid2d(predict, target, node_mask):
    """
    Args:
        predict: [batch, time_steps, nodes, channels]
        target: [batch, time_steps, nodes, channels]
        node_mask: [batch, 1, nodes, 1]
    """
    masked_pred = predict * node_mask
    masked_target = target * node_mask
    
    squared_error = (masked_pred - masked_target) ** 2
    
    mask_sum = node_mask.sum()
    
    # For velocity (u,v)
    rmse_u = torch.sqrt((squared_error[...,:2].sum()) / (mask_sum * 2 * predict.shape[1]))
    
    # For pressure (p)
    # rmse_p = torch.sqrt((squared_error[...,2:].sum()) / (mask_sum * predict.shape[1]))
    
    return rmse_u.item()

def get_l2_loss(output, target):
    
    # output.dim = (batch, seq, N, c) or (batch, seq, N)
    # target.dim = (batch, seq, N, c) or (batch, seq, N)
    
    if output.dim() == 4:
        
        if output.shape[-1] == 1:
            output = output.squeeze(-1) 
            target = target.squeeze(-1) 
            
            error = (output - target)
            norm_error = torch.norm(error, dim=-1) / (torch.norm(target, dim=-1) + 1e-8)
            norm_error_time = torch.mean(norm_error, dim=-1)
            norm_error_batch = torch.mean(norm_error_time, dim=0)
        else:
            error = (output - target)
            norm_error = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + 1e-8)
            norm_error_channel = torch.mean(norm_error, dim=-1)
            norm_error_time = torch.mean(norm_error_channel, dim=-1)
            norm_error_batch = torch.mean(norm_error_time, dim=0)
            
    elif output.dim() == 3:
        error = (output - target)
        norm_error = torch.norm(error, dim=-1) / (torch.norm(target, dim=-1) + 1e-8)
        norm_error_time = torch.mean(norm_error, dim=-1)
        norm_error_batch = torch.mean(norm_error_time, dim=0)
    
    return norm_error_batch

def rescale_data_lid2d(data, info, if_rescale):
    
    if if_rescale:
        data[...,0] = data[...,0] * info['u_std'] + info['u_mean']
        data[...,1] = data[...,1] * info['v_std'] + info['v_mean'] 
        # data[...,2] = data[...,2] * info['p_std'] + info['p_mean']
        
    return data

def get_val_loss_lid2d(predict_hat, predict_hat_unc, state, if_rescale, info, node_mask):
    
    device = predict_hat.device
    #################################
    state = state.to(device)
    node_mask = node_mask.unsqueeze(1).unsqueeze(-1).to(device)
    
    losses = {}
    ################
    state_ = rescale_data_lid2d(state, info, if_rescale)
    predict_hat_ = rescale_data_lid2d(predict_hat, info, if_rescale)
    
    losses['L2_u'] = get_l2_loss(predict_hat_[...,0] * node_mask[...,0], state_[...,0] * node_mask[...,0]).item()
    losses['L2_v'] = get_l2_loss(predict_hat_[...,1] * node_mask[...,0], state_[...,1] * node_mask[...,0]).item()
    # losses['L2_p'] = get_l2_loss(predict_hat_[...,2] * node_mask[...,0], state_[...,2] * node_mask[...,0]).item()

    losses['mean_l2'] = get_l2_loss(predict_hat_ * node_mask, state_ * node_mask).item()
    
    losses['RMSE_u'] = compute_rmse_lid2d(predict_hat_, state_, node_mask)    
    
    losses["each_l2"] = get_each_l2(predict_hat_ * node_mask, state_ * node_mask)
    
    return losses

import torch
import torch.nn as nn

class NLL_Loss(nn.Module):
    def __init__(self, a=5.0, b=-2.5, eps=1e-5):
        super().__init__()
        self.a = a
        self.b = b
        self.eps = eps

    def forward(self, mu, sigma_norm, target):
        # sigma_sq = F.softplus(self.a * sigma_norm + self.b) + self.eps
        sigma_sq = sigma_norm + self.eps
        loss = 0.5 * torch.log(2 * torch.pi * sigma_sq) + (target - mu) ** 2 / (2 * sigma_sq) + 1/sigma_sq*1e-3
        return loss.mean()

    

def get_train_loss_lid2d(predict_hat, predict_hat_unc, label_gt, 
                   loss_flag, if_rescale, info, node_mask):
    
    # predict_hat.dim = [b,t,n,c]
    # label_gt.dim = [b,t,n,c]
    
    MSE = nn.MSELoss()

    losses = {}
    device = predict_hat.device
    label_gt = label_gt.to(device)
    node_mask = node_mask.unsqueeze(1).unsqueeze(-1).to(device)
    
    # data_1 = get_l2_loss(predict_hat * node_mask, label_gt * node_mask)
    # print(f"data_1: {data_1}")

    # if loss_flag == 'L2_norm_loss':
        
    #     losses['loss'] = get_l2_loss(predict_hat * node_mask, label_gt * node_mask)
        
    predict_hat_ = rescale_data_lid2d(predict_hat, info, if_rescale)
    label_gt_ = rescale_data_lid2d(label_gt, info, if_rescale)
    
    # if loss_flag == 'L2_loss':
        
        # print(predict_hat_.shape, node_mask.shape, label_gt_.shape)
        
        # losses['loss'] = get_l2_loss(predict_hat_ * node_mask, label_gt_ * node_mask)
    
    losses['loss'] = NLL_Loss()(predict_hat_*node_mask, predict_hat_unc*node_mask, label_gt_*node_mask) + get_l2_loss(predict_hat_ * node_mask, label_gt_ * node_mask)
    
    losses['L2_u'] = get_l2_loss(predict_hat_[...,0] * node_mask[...,0], label_gt_[...,0] * node_mask[...,0]).item()
    losses['L2_v'] = get_l2_loss(predict_hat_[...,1] * node_mask[...,0], label_gt_[...,1] * node_mask[...,0]).item()
    # losses['L2_p'] = get_l2_loss(predict_hat_[...,2] * node_mask[...,0], label_gt_[...,2] * node_mask[...,0]).item()
    
    # losses['mean_l2'] = losses['loss'].item()
    losses['mean_l2'] = (losses['L2_u'] + losses['L2_v']) / 2
    losses['RMSE_u'] = compute_rmse_lid2d(predict_hat_, label_gt_, node_mask)    
    losses["each_l2"] = get_each_l2(predict_hat_ * node_mask, label_gt_ * node_mask)
    
    return losses

def get_each_l2(predict_hat, label_gt):
    
    t_step = label_gt.shape[1]
    losses_each_t = torch.zeros(t_step)
    
    for t in range(t_step):
        
        error = predict_hat[:,t] - label_gt[:,t]
        
        norm_error = torch.norm(error, dim=-2) / (torch.norm(label_gt[:,t], dim=-2) + 1e-6)
        norm_error_channel = torch.mean(norm_error, dim=-1)
        norm_error_batch = torch.mean(norm_error_channel, dim=0)
        
        losses_each_t[t] = norm_error_batch.item()
    
    return losses_each_t


def get_train_l2(output, target):
    
    # output.dim = (batch, seq, N, c)
    # target.dim = (batch, seq, N, c)
    
    b,t,n,c = output.shape
    
    error = (output - target)
    norm_error = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + 1e-8)
    norm_error_channel = torch.mean(norm_error, dim=-1)
    norm_error_time = torch.mean(norm_error_channel, dim=-1)
    norm_error_batch = torch.mean(norm_error_time, dim=0)
    
    return norm_error_batch
