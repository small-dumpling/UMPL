import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torch.nn.functional import one_hot


class MLP(nn.Module):
    def __init__(self, 
                 input_size = 3, 
                 output_size=128, 
                 layer_norm=True, 
                 n_hidden=2, 
                 hidden_size=128, 
                 dropout=0.0,
                 act = 'ReLU'):
        super(MLP, self).__init__()
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'SiLU':
            self.act = nn.SiLU()
    
        if hidden_size == 0:
            f = [nn.Linear(input_size, output_size),  nn.Dropout(dropout)]
        else:
            f = [nn.Linear(input_size, hidden_size), self.act, nn.Dropout(dropout)]
            h = 1
            for i in range(h, n_hidden):
                f.append(nn.Linear(hidden_size, hidden_size))
                f.append(self.act)
                f.append(nn.Dropout(dropout))
            f.append(nn.Linear(hidden_size, output_size))
            if layer_norm:
                f.append(nn.LayerNorm(output_size))

        self.f = nn.Sequential(*f)

    def forward(self, x):
        # x = x.float()
        return self.f(x)

class GNN(nn.Module):
    def __init__(self, n_hidden=2, node_size=128, edge_size=128, output_size=None, layer_norm=False, dropout=0.0):
        super(GNN, self).__init__()
        output_size = output_size or node_size
        
        self.f_edge = MLP(input_size=edge_size + node_size * 2, n_hidden=n_hidden, layer_norm=layer_norm,
                          output_size=edge_size, dropout=dropout)
        
        self.f_node = MLP(input_size=edge_size + node_size, n_hidden=n_hidden, layer_norm=layer_norm,
                          output_size=output_size, dropout=dropout)

    def forward(self, V, E, edges):
        
        edges = edges.long()
        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, V.shape[-1]))

        edge_inpt = torch.cat([senders, receivers, E], dim=-1)
        edge_embeddings = self.f_edge(edge_inpt)

        col = edges[..., 1].unsqueeze(-1).repeat(1, 1, edge_embeddings.shape[-1])
        edge_sum = scatter_mean(edge_embeddings, col, dim=-2, dim_size=V.shape[1])

        node_inpt = torch.cat([V, edge_sum], dim=-1)
        node_embeddings = self.f_node(node_inpt)

        return node_embeddings, edge_embeddings
    
class Encoder(nn.Module):
    def __init__(self, 
                 state_size = 3, 
                 input_size = 2,
                 cond_size = 2,
                 space_size = 2,
                 state_embedding_dim = 128,
                 dropout=0.0,
                 act = 'ReLU'):
        super(Encoder, self).__init__()

        self.fv = MLP(input_size = input_size+cond_size+space_size, output_size=state_embedding_dim, act = act, dropout=dropout)
        self.fe = MLP(input_size = space_size + 1, output_size=state_embedding_dim, act = act, dropout=dropout)
        

    def forward(self, node_pos, edges, node_type, conditions_input):
        
        # Get nodes embeddings
        edges = edges.long()
        V = torch.cat([conditions_input, node_type, node_pos], dim=-1)

        # Get edges attr
        senders = torch.gather(node_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, 2))
        receivers = torch.gather(node_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, 2))

        distance = receivers - senders
        norm = torch.sqrt((distance ** 2).sum(-1, keepdims=True))
        E = torch.cat([distance, norm], dim=-1)
        V = self.fv(V)
        E = self.fe(E)

        return V, E

class Processor(nn.Module):
    def __init__(self, 
                 N=15,
                 state_embedding_dim = 128,
                 dropout=0.0,
                 act = 'ReLU'):
        super(Processor, self).__init__()
        
        self.gnn = nn.ModuleList([])
        for i in range(N):
            self.gnn.append(GNN(node_size=state_embedding_dim, edge_size=state_embedding_dim, output_size=state_embedding_dim, layer_norm=True, dropout=dropout))

    def forward(self, V, E, edges):
        
        for i, gn in enumerate(self.gnn):
            edges = edges
            v, e = gn(V, E, edges)
            V = V + v
            E = E + e

        V = V
        E = E
        return V, E

class MeshGraphNet_unc(nn.Module):
    def __init__(self, 
                 N = 12, 
                 state_size = 2, 
                 space_size = 2,
                 cond_size = 3,
                 input_size = 2,
                 noise_std = 2e-2,
                 state_embedding_dim = 128,
                 act = 'ReLU', # or 'SiLU'
                 dropout=0.0,
                 ):
        super(MeshGraphNet_unc, self).__init__()
        self.noise_std = float(noise_std)
        self.state_size = state_size
        
        self.encoder = Encoder(
            state_size = state_size,
            input_size = input_size,
            cond_size = cond_size,
            space_size = space_size,
            state_embedding_dim = state_embedding_dim, 
            act = act,
            dropout=dropout) 
        
        self.processor = Processor(
            N = N,
            state_embedding_dim = state_embedding_dim, 
            act = act,
            dropout=dropout)
        
        self.decoder_pred = MLP(
            input_size = state_embedding_dim, 
            output_size = state_size, 
            layer_norm = False,
            act = act,
            dropout=dropout)
        
        self.decoder_unc = MLP(
            input_size = state_embedding_dim, 
            output_size = state_size, 
            layer_norm = False,
            act = act,
            dropout=0)
        
    def forward(self, node_pos, edges, node_type,conditions_input):
        V, E = self.encoder(
            node_pos,
            edges, 
            node_type,
            conditions_input)
            
        V, E = self.processor(
            V,
            E, 
            edges)
        
        next_state_pred = self.decoder_pred(V)
        next_state_unc = self.decoder_unc(V)
        next_state_unc = nn.Sigmoid()(next_state_unc)
        
        return next_state_pred, next_state_unc



if __name__ == '__main__':
    node_pos = torch.rand(1,1000,2)
    conditions_input = torch.rand(1,1000,2)
    edges = torch.randint(0,10,(1,1000,2))
    node_type = torch.rand(1,1000,3)
    model = MeshGraphNet_unc()
    checkpoint_path = f"{'../result/MGN_lid2d_drop_0.01_89data_unc/nn/lid2d_MGN_base_epo_1000_1000.nn'}" 
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])  
    model.train()
    for i in range(5):
        y,_ = model(node_pos, edges, node_type, conditions_input)
        print(float(y.mean()))