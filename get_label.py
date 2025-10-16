import torch
import numpy as np
import os
import time
import copy

from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR
from torch import nn
from tensorboardX import SummaryWriter
from torch.nn.functional import one_hot


# load
from src.lid2d import Lid2d_Dataset_Stationary
from src.MGN import MeshGraphNet
from src.MGN_with_uncertainty import MeshGraphNet_unc  
from src.train_umpl import forward_unc
from utils import set_seed, init_weights, parse_args, collate

# os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(device)

def setup():
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    return rank, world_size

def gather_tensor(tensor, world_size):
    """
    Gathers tensors from all processes and reduces them by summing up.
    """
    # Ensure the tensor is on the same device as specified for the operation
    tensor = tensor.to(device)
    # All-reduce: Sum the tensors from all processes
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # Only on rank 0, we scale the tensor to find the average
    if dist.get_rank() == 0:
        tensor /= world_size
    return tensor

def get_l2_loss(output, target):
    
    # output.dim = (batch, seq, N, c)
    # target.dim = (batch, seq, N, c)
    
    b,n,c = output.shape
    
    error = (output - target)
    norm_error = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + 1e-8)
    norm_error_channel = torch.mean(norm_error, dim=-1)
    norm_error_batch = torch.mean(norm_error_channel, dim=0)
    
    return norm_error_batch

def get_model(args, dropout, if_init=True):
    # model
    ######################################
    if args.model["name"] == "MGN":
        model = MeshGraphNet_unc(
                    N = args.model["N"], 
                    state_size = args.model["state_size"], 
                    space_size = args.model["space_size"],
                    cond_size = args.model["cond_size"],
                    input_size = args.model["input_size"],
                    noise_std = args.model["noise_std"],
                    state_embedding_dim = args.model["state_embedding_dim"],
                    dropout = dropout,
                    act =  args.model["act"]
                    )
    else:
        raise ValueError
    
    if args.model["if_init"]:
        model.apply(init_weights)   
    else:
        checkpoint_path = f"{args.checkpoint_path}" 
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])  
    return model


def get_label(model, dataloader, local_rank):
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            name = int(data['name'])
            node_pos = data['node_pos']
            edges = data['edges']
            node_type = data['node_type']
            conditions_input = data['conditions_input']

            num_classes = int(torch.max(node_type+1))
            node_type = one_hot(node_type.long(), num_classes=num_classes).squeeze(-2)
            pred_all = []
            for k in range(100):
                pred_state, _  = model(
                    node_pos,
                    edges,
                    node_type,
                    conditions_input,
                    )
                pred_state = pred_state.detach().cpu()
                pred_all.append(pred_state)
            pred_all = torch.stack(pred_all, dim=0)
            pred_mean = torch.mean(pred_all, dim=0)
            pred_std = torch.std(pred_all, dim=0)
            if not os.path.exists(f"lid2d_mean_std"):
                os.makedirs(f"lid2d_mean_std")
            torch.save(pred_mean, f"lid2d_mean_std/lid2d_mean_{name}.pt")
            torch.save(pred_std, f"lid2d_mean_std/lid2d_std_{name}.pt")


def main(args):
    # setting
    local_rank, world_size = setup()
    EPOCH = args.train["warm_up_epoch"]
    real_lr = float(args.train["lr"])

    # data
    ######################################
    # load data
    if args.dataset["name"] == "lid2d":        
        train_dataset = Lid2d_Dataset_Stationary(
            data_path = args.dataset["data_path"], 
            mode="train",
            data_num = 81,
            init_ema = True
            )
        save_dataset = Lid2d_Dataset_Stationary(
            data_path = args.dataset["data_path"], 
            mode="valid",
            data_num = 798,
            init_ema = True
            )
        
    # sampler
    save_sampler = DistributedSampler(save_dataset, num_replicas=world_size, shuffle= False, rank=local_rank)
    save_dataloader = DataLoader(save_dataset, 
                        batch_size=1, 
                        sampler= save_sampler,
                        num_workers=0,
                        collate_fn=collate)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, shuffle= False, rank=local_rank)
    train_dataloader = DataLoader(train_dataset, 
                        batch_size=args.dataset["train"]["batchsize"], 
                        sampler= train_sampler,
                        num_workers=args.dataset["train"]["num_workers"],
                        collate_fn=collate)
     
    model_t = get_model(args, dropout=args.model["dropout"], if_init=args.model["if_init"]).to(local_rank)
    model_t = DDP(model_t, device_ids=[local_rank], find_unused_parameters=True)
   
    model_parameters = filter(lambda p: p.requires_grad, model_t.parameters())    
    params = int(sum([np.prod(p.size()) for p in model_parameters]))
    
    # train
    ######################################
    real_lr = float(args.train["lr"])
    
    if args.model["name"] == "MGN":
        optimizer = Adam(model_t.parameters(), lr=real_lr)
        scheduler = ExponentialLR(optimizer=optimizer, gamma=args.train["decayRate"])
    else:
        raise ValueError

    # warmup
    for epoch in range(EPOCH):
        loss_i = 0
        for input_train in train_dataloader:
            optimizer.zero_grad()
            pred, _ = forward_unc(args, model=model_t, input=input_train, device=local_rank, mode="train")
            gt = input_train['state'].to(local_rank)
            loss = get_l2_loss(pred, gt)
            loss.backward()
            optimizer.step()
            loss_i += loss.item()
        print(f"Epoch {epoch+1}/{EPOCH}, Loss: {loss_i/len(train_dataloader)}")
        scheduler.step()

    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model_t.module.state_dict() if args.train["if_multi_gpu"] else model_t.state_dict(),
        'optimizer': optimizer.state_dict(),
        'learning_rate': scheduler.get_last_lr()[0], 
    }
    torch.save(checkpoint, f"warmup_100_epoch_lid2d.nn")

    get_label(model_t, save_dataloader, local_rank)

        
if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    # with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
    #     file.write(str(args) + "\n")
    #     file.write(f"time is {time.asctime(time.localtime(time.time()))}\n")
        
    if args.seed is not None:
        set_seed(args.seed)

    # main(args)
    if args.train["if_multi_gpu"]:
        world_size = torch.cuda.device_count()  # 获取GPU数量
        print(f"Let's use {world_size} GPUs!")
    
    main(args)
    
    # with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
    #     file.write(f"time is {time.asctime( time.localtime(time.time()) )}\n")