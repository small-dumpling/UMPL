import random
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
from functools import partial
import os

class Lid2d_Dataset_Stationary(Dataset):
    def __init__(self, 
                 data_path, 
                 mode="test",
                 edge_direction="both",
                 normalize = True,
                 data_num = 200,
                 init_ema = False,
                 ):
        
        super(Lid2d_Dataset_Stationary, self).__init__()
        assert mode in ["train", "valid", "test"]
                    
        self.dataloc = []
        
        self.mode = mode
        self.init_ema = init_ema
        self.do_normalization = normalize
        self.data_num = data_num
        self.edge_direction = edge_direction
        
        self.resample_data()

        def open_h5py_file(path, idx):

            data_ = h5py.File(path+self.dataloc[idx], 'r')
 
            # ['cells', 'node_type', 'pos', 'pressure', 'velocity']
            
            cells = data_['cells'][:]
            node_type = data_['node_type'][:]
            pos = data_['node_pos'][:]

            Re = data_['conditions']['Re'][()]
            v_wall = data_['conditions']['v_slip'][()]
            v_wall = round(v_wall, 2)

            # pressure = data_["state"]['p'][:]
            # pressure = pressure[..., np.newaxis]
            u = data_["state"]['u'][:]
            v = data_["state"]['v'][:]
            velocity = np.stack([u, v], axis=-1)
            
            Re = (Re-100)/(10000-100)
            v_wall = (v_wall-0.1)/(1-0.1)

            condition = torch.tensor([Re, v_wall]).unsqueeze(0).repeat(u.shape[0], 1).float()
            
            name = int(self.dataloc[idx][:-3])
            if self.mode == "valid" and not self.init_ema:
                ema_mean = torch.load(f"lid2d_mean_std/lid2d_mean_{name}.pt", map_location="cpu")[0]
                ema_std = torch.load(f"lid2d_mean_std/lid2d_std_{name}.pt", map_location="cpu")[0]
            else:
                ema_mean = torch.from_numpy(velocity)
                ema_std = torch.from_numpy(velocity)

            return cells, node_type, pos, velocity, condition, ema_mean, ema_std, name
        

        self.data_files = partial(open_h5py_file, data_path)
        
    def resample_data(self):
        self.dataloc = []
        if self.mode == "train":
            with open("./data_split/train_list_stationary.txt", "r") as f:
                for item in f.readlines():
                    self.dataloc.append(item.strip())
                    
                num_samples = self.data_num 

                resampled_data = np.random.choice(self.dataloc, size=(num_samples), replace=False)
                self.dataloc = list(resampled_data)
                
        elif self.mode == "valid":
            with open("./data_split/valid_list_stationary.txt", "r") as f:
                for item in f.readlines():
                    self.dataloc.append(item.strip())
                if not self.init_ema:
                    num_samples = self.data_num 
                    resampled_data = np.random.choice(self.dataloc, size=(num_samples), replace=False)
                    self.dataloc = list(resampled_data)

        elif self.mode == "test":
            with open("./data_split/test_list_stationary.txt", "r") as f:
                for item in f.readlines():
                    self.dataloc.append(item.strip())
        else:
            raise ValueError
                    

    def __len__(self):
        return len(self.dataloc)

    def make_edges(self, faces):

        edges = torch.cat([faces[:, :2], faces[:, 1:], faces[:, ::2]], dim=0)
        
        receivers, _ = torch.min(edges, dim=-1)
        senders, _ = torch.max(edges, dim=-1)
        
        packed_edges = torch.stack([senders, receivers], dim=-1).int() # dim = [edges, 2]
        unique_edges = torch.unique(packed_edges, dim=0)
        
        if self.edge_direction == "both":
            unique_edges = torch.cat([unique_edges, torch.flip(unique_edges, dims=[-1])], dim=0)
        
        return unique_edges
        
    def __getitem__(self, idx):
        if self.data_num > 0 and idx == 0:
            self.resample_data()
        
        cells, node_type, pos, velocity, condition, ema_mean, ema_std, name = self.data_files(idx)
        
        cells = torch.from_numpy(cells).long()
        node_type = torch.from_numpy(node_type).long()
        node_pos = torch.from_numpy(pos).float()
        velocity = torch.from_numpy(velocity).float()
        name = torch.tensor(name).long()

        if self.do_normalization:
            state = self.normalize(velocity, None)
        else:
            state = torch.cat([velocity], dim=-1)
            
        mesh_edges = self.make_edges(cells.clone())
            
        input = {
                'node_pos': node_pos,
                'edges': mesh_edges,
                'state': state,
                'cells': cells,
                'node_type': node_type.unsqueeze(-1),
                'conditions_input': condition,
                'ema_mean': ema_mean,
                'ema_std': ema_std,
                'name': name,
                }
        
        return input
    
    def normalize(self, velocity=None, pressure=None):
        
        if pressure is not None:
            pressure_shape = pressure.shape
            mean = torch.tensor([0.0338]).to(pressure.device)
            std = torch.tensor([0.3260]).to(pressure.device)
            pressure = pressure.reshape(-1, 1)
            pressure = (pressure - mean) / std
            pressure = pressure.reshape(pressure_shape)
        
        if velocity is not None:
            velocity_shape = velocity.shape
            mean = torch.tensor([0.5648, 0.0006]).to(velocity.device).view(-1, 2)
            std = torch.tensor([0.5705, 0.1436]).to(velocity.device).view(-1, 2)
            velocity = velocity.reshape(-1, 2)
            velocity = (velocity - mean) / std
            velocity = velocity.reshape(velocity_shape)
            
        state = torch.cat([velocity], dim=-1)
            
        return state
    

'''
u.dim = (1131714000,)
v.dim = (1131714000,)
p.dim = (1131714000,)
x.dim = (1131714000,)
y.dim = (1131714000,)
{'u': {'mean': 0.5648434, 'std': 0.5704901, 'max': 4.4910502, 'min': -1.1680238}, 
 'v': {'mean': 0.00062732404, 'std': 0.14361902, 'max': 2.2336626, 'min': -2.3224459}, 
 'p': {'mean': 0.03376516, 'std': 0.32603782, 'max': 7.699855, 'min': -6.2453523}, 
 'x': {'mean': 0.7472681, 'std': 0.4610336, 'max': 1.6, 'min': 0.0}, 
 'y': {'mean': 0.20479651, 'std': 0.15084454, 'max': 0.41, 'min': 0.0}
}
'''
if __name__ == "__main__":
    dataset = Lid2d_Dataset_Stationary(data_path = "../lid2d_stationary/", 
            mode="test",
            normalize = True,)
    for data in dataset:
        for key in data.keys():
            print(key)
            print(data[key].shape)
        break
