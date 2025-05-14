import torch
from tqdm import tqdm
from torch.nn.functional import one_hot
import random
import os
import h5py
import math
import torch.nn as nn
from .all_loss_umpl import get_train_loss_lid2d, get_val_loss_lid2d
import torch.nn.functional as F
import numpy as np

def get_l2_loss(output, target):
    
    # output.dim = (batch, seq, N, c)
    # target.dim = (batch, seq, N, c)
    
    b,n,c = output.shape
    
    error = (output - target)
    norm_error = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + 1e-8)
    norm_error_channel = torch.mean(norm_error, dim=-1)
    norm_error_batch = torch.mean(norm_error_channel, dim=0)
    
    return norm_error_batch

class NLL_Loss(nn.Module):
    def __init__(self, a=5.0, b=-2.5, eps=1e-5):
        super().__init__()
        self.a = a
        self.b = b
        self.eps = eps

    def forward(self, mu, sigma_norm, target, mask):
        # sigma_sq = F.softplus(self.a * sigma_norm + self.b) + self.eps
        sigma_sq = sigma_norm + self.eps
        loss = mask*(0.5 * torch.log(2 * torch.pi * sigma_sq) + (target - mu) ** 2 / (2 * sigma_sq) + 1/sigma_sq*1e-3)
        return loss.mean()

def pointwise_loss(output, target):
    # output.dim = (batch, seq, N, c)
    # target.dim = (batch, seq, N, c)
    
    b,n,c = output.shape
    
    error = (output - target)
    # norm_error = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + 1e-8)
    norm_error_channel = torch.mean(error, dim=-1)
    return norm_error_channel

def save_inference(uvp, output_uvp_hat, if_rescale, info, num, save_path):
    device = output_uvp_hat.device
    #################################
    # uv,p --- output_uvp_hat
    uvp_target = uvp.to(device)
    
    if if_rescale:
        uvp_target[...,0] = (uvp_target[...,0] * info['u_std'] + info['u_mean'])
        uvp_target[...,1] = (uvp_target[...,1] * info['v_std'] + info['v_mean'])
        # uvp_target[...,2] = (uvp_target[...,2] * info['p_std'] + info['p_mean'])

        
        output_uvp_hat[...,0] = (output_uvp_hat[...,0] * info['u_std'] + info['u_mean'])
        output_uvp_hat[...,1] = (output_uvp_hat[...,1] * info['v_std'] + info['v_mean'])
        # output_uvp_hat[...,2] = (output_uvp_hat[...,2] * info['p_std'] + info['p_mean'])
        
        
    error = torch.abs(uvp_target - output_uvp_hat)
    save_path = f"{save_path}/infer_result/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = h5py.File(save_path + f"{num}.h5", 'w')
    f.create_dataset('predict', data=output_uvp_hat.detach().cpu().numpy())
    f.create_dataset('ground_truth', data=uvp_target.detach().cpu().numpy())
    f.create_dataset('error', data=error.detach().cpu().numpy())
    

def forward_unc(args, model, input, device, mode="train"):
    state = input['state'].clone()        
    node_pos = input['node_pos'].clone() 
    edges = input['edges'].clone() 
    node_type = input['node_type'].clone() 
    conditions_input = input['conditions_input'].clone() 
    
    node_mask = input['mask'].clone() 
        
    num_classes = int(torch.max(node_type+1))
    node_type = one_hot(node_type.long(), num_classes=num_classes).squeeze(-2)
    
    node_pos = node_pos.to(device)
    node_type = node_type.to(device)
    conditions_input = conditions_input.to(device)
    
    # state_in.dim = [B, N, 4]
    if len(node_type.shape)==3:
        mask = node_type[:, :, 0] != 0
    else:
        mask = node_type[:, :, :, 0] != 0
        

    next_state_pred, next_state_unc = model(
        node_pos,
        edges,
        node_type,
        conditions_input,
        )
        
    return next_state_pred, next_state_unc


def get_coe(args, epoch):
    return 0.5 + epoch/args.train["epochs"]*0.5

def smooth_random_field(node_pos, edges, r, eps=1e-6):
    """
    Batched version of spatial smoothing on random field using inverse distance weights.

    Args:
        node_pos: (B, N, D) - batched node positions
        edges: (B, E, 2) - batched edge indices (local indices)
        r: (B, N) - batched random values ~ N(0, 1)
        eps: float - small number to avoid division by zero

    Returns:
        xi: (B, N) - batched smoothed sign field
    """
    B, N, D = node_pos.shape
    _, E, _ = edges.shape

    device = node_pos.device

    # Step 1: flatten batch
    node_pos_flat = node_pos.reshape(B * N, D)
    r_flat = r.reshape(B * N)

    # Offset edges for global indexing
    batch_offset = torch.arange(B, device=device).view(B, 1, 1) * N  # (B, 1, 1)
    edges_offset = edges + batch_offset  # (B, E, 2)
    edges_flat = edges_offset.reshape(B * E, 2)  # (B*E, 2)

    # Step 2: compute distance and weights
    i = edges_flat[:, 0]  # receiving node index
    j = edges_flat[:, 1]  # contributing node index

    gi = node_pos_flat[i]
    gj = node_pos_flat[j]
    dist = torch.norm(gi - gj, dim=1)
    weight = 1.0 / (dist + eps)
    contrib = weight * r_flat[j]

    # Step 3: scatter to accumulate
    xi_accum = torch.zeros_like(r_flat)
    xi_accum.index_add_(0, i, contrib)

    # Step 4: reshape back to (B, N)
    xi = torch.sign(xi_accum).view(B, N)
    return xi

def train(args, model_t, model_s, train_dataloader, test_dataloader, optim_t, optim_s, epoch, local_rank):
    
    model_t.train()
    model_s.train()
    teacher_loss = 0
    feedback_loss = 0   
    student_loss = 0     
    num = 0
    
    alpha = args.alpha
    
    train_dataloader = iter(train_dataloader)
    test_dataloader = iter(test_dataloader)
    
    for i in range(len(train_dataloader)):
        input_train = next(train_dataloader)
        input_test = next(test_dataloader)
        
        model_t.to("cpu")
        model_s.to("cpu")

        # 清除梯度
        optim_t.zero_grad()
        optim_s.zero_grad()
        
        train_gt = input_train['state'].to(local_rank)
        # test_gt = input_test['state'].to(local_rank)
        node_pos = input_test['node_pos'].clone() 
        edges = input_test['edges'].clone() 
        
        # forward
        ema_mean = input_test['ema_mean'].to(local_rank)
        ema_std = input_test['ema_std'].to(local_rank)
        
        with torch.no_grad():
            name = input_test['name']
            model_t.to(local_rank)
            pred_state, _ = forward_unc(args, model_t, input_test, local_rank)
            
            ema_mean = alpha*pred_state+(1-alpha)*ema_mean
            ema_var = alpha*(pred_state-ema_mean)**2+(1-alpha)*ema_std**2
            ema_std = torch.sqrt(ema_var)
            
            pseudo_state = pred_state + ema_std*smooth_random_field(node_pos, edges, torch.randn_like(pred_state))
            
            N_topk = int(get_coe(args, epoch)*pred_state.shape[1])
            # N_topk = int(pred_state.shape[1])

            # Step 1: get topk
            _, indices = torch.topk(ema_std, N_topk, dim=1)  # (B, C, k)
            indices = indices.to(local_rank)

            # Step 2: get mask
            # initialize mask
            pseudo_mask = torch.ones_like(pseudo_state, dtype=torch.long).to(local_rank) 
            # pseudo_mask.scatter_(1, indices, 0)

            # get EMC files
            for l in range(len(name)):
                torch.save(ema_mean[l].unsqueeze(0), f"lid2d_mean_std/lid2d_mean_{int(name[l])}.pt")
                torch.save(ema_std[l].unsqueeze(0), f"lid2d_mean_std/lid2d_std_{int(name[l])}.pt")
            
            
        with torch.no_grad():        
            model_s.to(local_rank)
            model_t.to("cpu")
            pred_state_train_before, _ = forward_unc(args, model_s, input_train, local_rank)
            loss_before = pointwise_loss(pred_state_train_before, train_gt)
        
        pred_test, pred_test_unc = forward_unc(args, model_s, input_test, local_rank)
        
        loss_l2_test = NLL_Loss()(pred_test, pred_test_unc, pseudo_state, pseudo_mask)
        # loss_l2_test = get_l2_loss(pred_test, pseudo_state)
        loss_l2_test.backward()
        optim_s.step()
        
        with torch.no_grad(): 
            pred_state_train_after, _ = forward_unc(args, model_s, input_train, local_rank)
            loss_after = pointwise_loss(pred_state_train_after, train_gt)
            feed_back_cof = loss_before - loss_after
            del pred_state_train_before, pred_state_train_after
        
        model_s.train()
        loss_feed_back = pointwise_loss(pred_state, pseudo_state)
        loss_feedback = (feed_back_cof * loss_feed_back)
        
        loss_feedback = loss_feedback.mean()

        model_t.to(local_rank)
        pred_state_train_teacher, pred_state_train_teacher_unc = forward_unc(args, model_t, input_train, local_rank)
        loss_supervise = get_l2_loss(pred_state_train_teacher, train_gt)
        
        loss_all = loss_feedback + loss_supervise
        loss_all.backward()
        optim_t.step()
        
        optim_s.zero_grad()
        
        pred_state_train, pred_state_unc = forward_unc(args, model_s, input_train, local_rank)
        sup_mask = torch.ones_like(pred_state_train, dtype=torch.long).to(local_rank) 
        loss_s_l2 = get_l2_loss(pred_state_train, train_gt)
        loss_s_nll = NLL_Loss()(pred_state_train, pred_state_unc, train_gt, sup_mask)
        loss_s = loss_s_nll
        loss_s.backward()
        optim_s.step()
        
        teacher_loss += loss_supervise.item()
        feedback_loss += loss_feedback.item()
        student_loss += loss_s_l2.item()
        num = num + 1
        
        # break
    
    return teacher_loss / num, feedback_loss / num, student_loss / num 

def validate(args, model, val_dataloader, device):
    
    model.train()
    model.to(device)
    
    L2_u = 0
    L2_v = 0
    # L2_p = 0
    L2_mean = 0
    
    RMSE_u = 0
    # RMSE_p = 0
    
    num = 0
    
    with torch.no_grad():
        # inference
        # for i, [input, t] in enumerate(tqdm(val_dataloader, desc="Validation")):
        for i, input in enumerate(val_dataloader):    
            state = input['state'].to(device)
            node_mask = input['mask']
            
            # dim = [b, n]
            
            batch_num = state.shape[0]
            
            predict_hat, _ = forward_unc(args, model, input, device, "test")
            
            costs = get_val_loss_lid2d(
                predict_hat,
                state,
                args.train["if_rescale"], 
                args.train["info"],
                node_mask
                )
                
            #########################################
            L2_u = L2_u + costs['L2_u'] * batch_num
            L2_v = L2_v + costs['L2_v'] * batch_num
            # L2_p = L2_p + costs['L2_p'] * batch_num
            L2_mean = L2_mean + costs['mean_l2'] * batch_num
            
            RMSE_u = RMSE_u + costs['RMSE_u'] * batch_num
            # RMSE_p = RMSE_p + costs['RMSE_p'] * batch_num
            
            #########################################
            num = num + batch_num

    batch_error = {}

    batch_error['L2_u'] = L2_u / num
    batch_error['L2_v'] = L2_v / num
    # batch_error['L2_p'] = L2_p / num
    batch_error['mean_l2'] = L2_mean / num
    
    batch_error['RMSE_u'] = RMSE_u / num
    # batch_error['RMSE_p'] = RMSE_p / num
    
    return batch_error

