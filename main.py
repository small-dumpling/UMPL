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
    
    if if_init:
        model.apply(init_weights)   
    else:
        checkpoint_path = f"{args.checkpoint_path}" 
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])  
    return model


    
def main(args):
    # setting
    local_rank, world_size = setup()
    EPOCH = args.train["epoch"]
    real_lr = float(args.train["lr"])

    # data
    ######################################
    # load data
    if args.dataset["name"] == "lid2d":
        train_dataset = Lid2d_Dataset_Stationary(
            data_path = args.dataset["data_path"], 
            mode="train",
            data_num = args.data_num
            )
        
        valid_dataset = Lid2d_Dataset_Stationary(
            data_path = args.dataset["data_path"], 
            mode="valid",
            data_num = args.data_num
            )
        
        test_dataset = Lid2d_Dataset_Stationary(
            data_path = args.dataset["data_path"], 
            mode="test",
            data_num = 0
            )
        
        
    # sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, shuffle= True, seed=args.seed, rank=local_rank)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, shuffle= False, rank=local_rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size,  shuffle=False, rank=local_rank)
    
    train_dataloader = DataLoader(train_dataset, 
                        batch_size=args.dataset["train"]["batchsize"], 
                        sampler=train_sampler,
                        num_workers=args.dataset["train"]["num_workers"],
                        collate_fn=collate)

    valid_dataloader = DataLoader(valid_dataset, 
                        batch_size=args.dataset["train"]["batchsize"], 
                        sampler= valid_sampler,
                        num_workers=args.dataset["train"]["num_workers"],
                        collate_fn=collate)

    test_dataloader = DataLoader(test_dataset, 
                        batch_size=args.dataset["test"]["batchsize"], 
                        sampler= test_sampler,
                        num_workers=args.dataset["test"]["num_workers"],
                        collate_fn=collate)
    

     
    model_t = get_model(args, dropout=args.model["dropout"], if_init=False).to(local_rank)
    model_s = get_model(args, dropout=0.0, if_init=False).to(local_rank)
    
    model_t = DDP(model_t, device_ids=[local_rank], find_unused_parameters=True)
    model_s = DDP(model_s, device_ids=[local_rank], find_unused_parameters=True)
    
    model_parameters = filter(lambda p: p.requires_grad, model_t.parameters())    
    params = int(sum([np.prod(p.size()) for p in model_parameters]))
    
    if local_rank == 0:
        
        print("---train_dataloader---")
        for i, data in enumerate(train_dataloader):
            for key in data.keys():
                print(key, data[key].shape)
            break
        print("---test_dataloader---")
        for i, data in enumerate(test_dataloader):
            for key in data.keys():
                print(key, data[key].shape)
            break

        print("---------")      
        print(f"No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}")
        print(f"No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}")
        print("---------")
        print(f"EPOCH: {EPOCH}, #params: {params}")       
    
        if not os.path.exists(f"{args.save_path}/record/"):
            os.makedirs(f"{args.save_path}/record/")
            
        with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
            file.write(f"No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}\n")
            file.write(f"No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}\n")
            file.write(f"Let's use {torch.cuda.device_count()} GPUs!\n")
            file.write(f"{args.name}, #params: {params}\n")
            file.write(f"EPOCH: {EPOCH}\n")        

        log_dir = f"{args.save_path}/logs/{args.name}/rank_{local_rank}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        
    # train
    ######################################
    real_lr = float(args.train["lr"])
    
    optimizer_t = torch.optim.AdamW(model_t.parameters(), lr=real_lr, weight_decay=real_lr/100.0)
    if EPOCH == 1:
        scheduler_t = CosineAnnealingLR(optimizer_t, T_max= EPOCH, eta_min = real_lr)  
    else:
        scheduler_t = CosineAnnealingLR(optimizer_t, T_max= EPOCH, eta_min = real_lr/50)
        
    optimizer_s = torch.optim.AdamW(model_s.parameters(), lr=real_lr, weight_decay=real_lr/100.0)
    if EPOCH == 1:
        scheduler_s = CosineAnnealingLR(optimizer_s, T_max= EPOCH, eta_min = real_lr)  
    else:
        scheduler_s = CosineAnnealingLR(optimizer_s, T_max= EPOCH, eta_min = real_lr/50)
        
    # different SSL methods
    if args.method == "UMPL":
        from src.train_umpl import train, validate
    # elif args.method == "mean_TS":
    #     from src.train_mean_TS import train, validate
    # elif args.method == "PL":
    #     from src.train_PL import train, validate
    # elif args.method == "noisy_TS":
    #     from src.train_noisy_TS import train, validate
    else:
        raise ValueError
    

    for epoch in range(EPOCH):
        start_time = time.time()
        train_loss, feedback_loss, student_loss = train(args, model_t, model_s, train_dataloader, valid_dataloader, optimizer_t, optimizer_s, epoch, local_rank)
        end_time = time.time()

        # get lr
        if args.method == "UMPL" or args.method == "noisy_TS":
            scheduler_t.step()
            scheduler_s.step()
            current_lr_t = scheduler_t.get_last_lr()[0]  
            current_lr_s = scheduler_s.get_last_lr()[0] 
        else:
            scheduler_s.step()
            current_lr_t = scheduler_t.get_last_lr()[0] 
            current_lr_s = scheduler_s.get_last_lr()[0] 
        
        training_time = (end_time - start_time)
        training_time = torch.tensor(training_time, device=local_rank)
        
        current_lr_t = torch.tensor(current_lr_t, device=local_rank)
        current_lr_s = torch.tensor(current_lr_s, device=local_rank)
        
        current_lr_t = gather_tensor(current_lr_t, world_size)
        current_lr_s = gather_tensor(current_lr_s, world_size)
        training_time = gather_tensor(training_time, world_size)
        #######################
        if args.dataset["name"] == "lid2d":
            train_loss = gather_tensor(torch.tensor(train_loss, device=local_rank), world_size)
            feedback_loss = gather_tensor(torch.tensor(feedback_loss, device=local_rank), world_size)
            student_loss = gather_tensor(torch.tensor(student_loss, device=local_rank), world_size)
            
            if local_rank == 0:

                print(f"-----Training, Epoch: {epoch + 1}/{EPOCH}-----")
                print(f"Teacher Loss: {train_loss:.4e}")
                print(f"Feedback Loss: {feedback_loss:.4e}")
                print(f"Student Loss: {student_loss:.4e}")
                print(f"time pre train epoch/s:{training_time:.2f}")
                
                writer.add_scalar('Loss/teacher', train_loss, epoch)
                writer.add_scalar('Loss/feedback', feedback_loss, epoch)
                writer.add_scalar('Loss/student', student_loss, epoch)
                
                with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
                    file.write(f"-----Training, epoch: {epoch + 1}/{EPOCH}-----\n")
                    file.write(f"Teacher Loss: {train_loss:.4e}\n")
                    file.write(f"Feedback Loss: {feedback_loss:.4e}\n")
                    file.write(f"Student Loss: {student_loss:.4e}\n")
                    file.write(f"time pre train epoch/s:{training_time:.2f}\n")
           
        if (epoch+1) % 10 == 0 or epoch == 0 or (epoch+1) == EPOCH:
            # test
            start_time = time.time() 
            test_error = validate(args, model_s, test_dataloader, local_rank)
            end_time = time.time()
            
            training_time1 = (end_time - start_time)
            training_time1 = torch.tensor(training_time1, device=local_rank)
            training_time1 = gather_tensor(training_time1, world_size)
            #######################
            if args.dataset["name"] == "lid2d":
                test_L2_u = gather_tensor(torch.tensor(test_error['L2_u'], device=local_rank), world_size)
                test_L2_v = gather_tensor(torch.tensor(test_error['L2_v'], device=local_rank), world_size)
                # test_L2_p = gather_tensor(torch.tensor(test_error['L2_p'], device=local_rank), world_size)
                test_mean_l2 = gather_tensor(torch.tensor(test_error['mean_l2'], device=local_rank), world_size)   
                
                test_RMSE_u = gather_tensor(torch.tensor(test_error['RMSE_u'], device=local_rank), world_size)
                # test_RMSE_p = gather_tensor(torch.tensor(test_error['RMSE_p'], device=local_rank), world_size)
                
                
                if local_rank == 0:
                    print("---Inference Student---")
                    print(f"Epoch: {epoch + 1}/{EPOCH}, test_mean_l2: {test_mean_l2:.4e}")
                    print(f"L2 loss: u {test_L2_u:.4e}, v: {test_L2_v:.4e}")
                    print(f"RMSE loss: U {test_RMSE_u:.4e}")
                    print(f"time pre test epoch/s:{training_time1:.2f}")
                    print("--------------")
                    
                    writer.add_scalar('L2/test_mean_l2_student', test_mean_l2, epoch)
                    writer.add_scalar('L2/test_L2_u_student', test_L2_u, epoch)
                    writer.add_scalar('L2/test_L2_v_student', test_L2_v, epoch)
                    # writer.add_scalar('L2/test_L2_p', test_L2_p, epoch)
                    
                    writer.add_scalar('RMSE/test_RMSE_u', test_RMSE_u, epoch)
                    # writer.add_scalar('RMSE/test_RMSE_p', test_RMSE_p, epoch)
                    
                    
                    with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
                        file.write(f"Inference student, epoch: {epoch + 1}/{EPOCH}, test_mean_l2: {test_mean_l2:.4e}\n")
                        file.write(f"Test: L2_u: {test_L2_u:.4e}, L2_v: {test_L2_v:.4e}\n")
                        file.write(f"Test: RMSE_u: {test_RMSE_u:.4e}\n")
                        file.write(f"time pre test epoch/s:{training_time1:.2f}\n") 
                        
        if (epoch+1) % 10 == 0 or epoch == 0 or (epoch+1) == EPOCH:
            # test
            start_time = time.time() 
            test_error = validate(args, model_t, test_dataloader, local_rank)
            end_time = time.time()
            
            training_time1 = (end_time - start_time)
            training_time1 = torch.tensor(training_time1, device=local_rank)
            training_time1 = gather_tensor(training_time1, world_size)
            #######################
            if args.dataset["name"] == "lid2d":
                test_L2_u = gather_tensor(torch.tensor(test_error['L2_u'], device=local_rank), world_size)
                test_L2_v = gather_tensor(torch.tensor(test_error['L2_v'], device=local_rank), world_size)
                # test_L2_p = gather_tensor(torch.tensor(test_error['L2_p'], device=local_rank), world_size)
                test_mean_l2 = gather_tensor(torch.tensor(test_error['mean_l2'], device=local_rank), world_size)   
                
                test_RMSE_u = gather_tensor(torch.tensor(test_error['RMSE_u'], device=local_rank), world_size)
                # test_RMSE_p = gather_tensor(torch.tensor(test_error['RMSE_p'], device=local_rank), world_size)
                
                
                if local_rank == 0:
                    print("---Inference Teacher---")
                    print(f"Epoch: {epoch + 1}/{EPOCH}, test_mean_l2: {test_mean_l2:.4e}")
                    print(f"L2 loss: u {test_L2_u:.4e}, v: {test_L2_v:.4e}")
                    print(f"RMSE loss: U {test_RMSE_u:.4e}")
                    print(f"time pre test epoch/s:{training_time1:.2f}")
                    print("--------------")
                    
                    writer.add_scalar('L2/test_mean_l2_teacher', test_mean_l2, epoch)
                    writer.add_scalar('L2/test_L2_u_teacher', test_L2_u, epoch)
                    writer.add_scalar('L2/test_L2_v_teacher', test_L2_v, epoch)
                    # writer.add_scalar('L2/test_L2_p', test_L2_p, epoch)
                    
                    writer.add_scalar('RMSE/test_RMSE_u', test_RMSE_u, epoch)
                    # writer.add_scalar('RMSE/test_RMSE_p', test_RMSE_p, epoch)
                    
                    
                    with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
                        file.write(f"Inference Teacher, epoch: {epoch + 1}/{EPOCH}, test_mean_l2: {test_mean_l2:.4e}\n")
                        file.write(f"Test: L2_u: {test_L2_u:.4e}, L2_v: {test_L2_v:.4e}\n")
                        file.write(f"Test: RMSE_u: {test_RMSE_u:.4e}\n")
                        file.write(f"time pre test epoch/s:{training_time1:.2f}\n") 
        
        if (epoch+1) % 100 == 0 or epoch == 0 or (epoch+1) == EPOCH:
            if args.if_save:
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model_t.module.state_dict() if args.train["if_multi_gpu"] else model_t.state_dict(),
                    'optimizer': optimizer_t.state_dict(),
                    'learning_rate': scheduler_t.get_last_lr()[0],
                }
                nn_save_path = os.path.join(args.save_path, "nn")
                os.makedirs(nn_save_path, exist_ok=True)
                torch.save(checkpoint, f"{nn_save_path}/{args.name}_{epoch+1}_t.nn")
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model_s.module.state_dict() if args.train["if_multi_gpu"] else model_s.state_dict(),
                    'optimizer': optimizer_s.state_dict(),
                    'learning_rate': scheduler_s.get_last_lr()[0], 
                }
                torch.save(checkpoint, f"{nn_save_path}/{args.name}_{epoch+1}_s.nn")

    if local_rank == 0:
        writer.close()
        
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
        world_size = torch.cuda.device_count() 
        print(f"Let's use {world_size} GPUs!")
    
    main(args)
    
    # with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
    #     file.write(f"time is {time.asctime( time.localtime(time.time()) )}\n")