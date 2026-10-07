# SPDX-License-Identifier: Apache-2.0
# Adapted from LibCity GWNET and the original Graph WaveNet implementation.
# Copyright (c) <2019> <Zonghan Wu> (original Graph WaveNet components).
# See THIRD_PARTY_NOTICES.md and third_party/licenses/ for source and license notices.
# Local modifications: multivariate sequences, scaling, handover graph, and checkpoints.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import networkx as nx
from typing import Dict, List, Any, Optional
import scipy.sparse as sp
from scipy.sparse import linalg
from sklearn.preprocessing import StandardScaler

from .pytorch_base_model import PyTorchBaseModel

class GraphWaveNetModel(PyTorchBaseModel):
    """
    Graph WaveNet implementation adapted for the pre-sequenced NumPy data pipeline.
    This model handles its own spatial setup (adjacency matrix) and data scaling
    before leveraging the parent PyTorchBaseModel for training and prediction loops.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Graph components, initialized to None
        self.adj_mx = None
        self.num_nodes = 0
        
        # Data processing components
        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # Load the graph structure from the path specified in the config
        graph_path = self.config.get("graph_path")
        if not graph_path:
            raise ValueError("Graph path not specified in model config.")
        self.graph = None if self.config.get('_loading_checkpoint') else self._load_graph(graph_path)
        
    def _load_graph(self, graph_path: str) -> nx.Graph:
        """Loads the handover graph from a pickle file."""
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        print(f"✅ Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        return graph

    def _setup_spatial_structure(self, cell_order: List[str]):
        """
        Initializes the adjacency matrix based on the graph and the ordered list of cells.
        This must be called before building the model.
        """
        print("Setting up spatial structure (adjacency matrix)...")
        self.num_nodes = len(cell_order)
        
        # Ensure all cells from the data are in the graph
        graph_nodes = set(self.graph.nodes())
        if not set(cell_order).issubset(graph_nodes):
            missing_nodes = set(cell_order) - graph_nodes
            raise ValueError(f"{len(missing_nodes)} cells from data are not in the graph: {list(missing_nodes)[:5]}...")

        # Create the adjacency matrix using the exact order of cells from the data
        adj_mx = nx.adjacency_matrix(
            self.graph, 
            nodelist=cell_order, 
            weight='weight'
        ).todense().astype(np.float32)
        
        self.adj_mx = adj_mx
        print(f"✅ Spatial structure created for {self.num_nodes} nodes.")

    def build_model(self, static_features: Optional[np.ndarray] = None) -> nn.Module:
        """Builds the underlying GWNET PyTorch module."""
        if self.adj_mx is None:
            raise RuntimeError("Adjacency matrix is not initialized. Setup must be run first.")
        
        # Static features are currently used by tree baselines, not this architecture.
        gwnet_config = {
            **self.config,
            'adj_mx': self.adj_mx,
            'device': self.device,
            'num_nodes': self.num_nodes,
            'input_feature_dim': self.num_input_features,
            'output_feature_dim': self.num_target_features,
            'input_window': self.config['input_steps'],
            'output_window': self.config['output_steps'],
        }
        
        return GWNET(
            config=gwnet_config,
            num_nodes=self.num_nodes,
            input_feature_dim=self.num_input_features,
            output_feature_dim=self.num_target_features
        )

    def train(self, 
            X_train: np.ndarray, 
            Y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            Y_val: Optional[np.ndarray] = None,
            static_features: Optional[np.ndarray] = None):
        """
        Prepares the model and data, then calls the parent training loop.
        """
        # --- 1. First-time Setup ---
        if not self.is_fitted:
            # Get dimensions and cell order from config (provided by pipeline)
            cell_order = self.config.get('cell_order')
            if not cell_order:
                raise ValueError("`cell_order` must be in the config for GraphWaveNetModel.")
            self.num_input_features = X_train.shape[-1]
            self.num_target_features = Y_train.shape[-1]
            
            # Setup adjacency matrix
            self._setup_spatial_structure(cell_order)

        # --- 2. Scale Data ---
        # Reshape data to 2D for the scaler: (samples * timesteps * nodes, features)
        X_train_flat = X_train.reshape(-1, self.num_input_features)
        Y_train_flat = Y_train.reshape(-1, self.num_target_features)

        print("Fitting scalers and transforming training data...")
        X_train_scaled_flat = self.input_scaler.fit_transform(X_train_flat)
        Y_train_scaled_flat = self.target_scaler.fit_transform(Y_train_flat)

        # Reshape back to original dimensions
        X_train_scaled = X_train_scaled_flat.reshape(X_train.shape)
        Y_train_scaled = Y_train_scaled_flat.reshape(Y_train.shape)
        print("✅ Scaling complete.")

        X_val_scaled, Y_val_scaled = None, None
        if X_val is not None and Y_val is not None:
            print("Transforming validation data...")
            X_val_flat = X_val.reshape(-1, self.num_input_features)
            Y_val_flat = Y_val.reshape(-1, self.num_target_features)
            X_val_scaled_flat = self.input_scaler.transform(X_val_flat)
            Y_val_scaled_flat = self.target_scaler.transform(Y_val_flat)
            X_val_scaled = X_val_scaled_flat.reshape(X_val.shape)
            Y_val_scaled = Y_val_scaled_flat.reshape(Y_val.shape)

        # --- 3. Call Parent Trainer ---
        # The parent `train` method will handle DataLoader creation, model building,
        # and the entire training loop.
        super().train(
            X_train=X_train_scaled,
            Y_train=Y_train_scaled,
            X_val=X_val_scaled,
            Y_val=Y_val_scaled,
            static_features=static_features # Pass it along, though GWNET doesn't use it
        )

    def predict(self, 
                X_test: np.ndarray,
                static_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Scales input, gets predictions from the parent method, and inverse-scales the output.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call 'train' before 'predict'.")

        # --- 1. Scale Input Data ---
        X_test_flat = X_test.reshape(-1, self.num_input_features)
        X_test_scaled_flat = self.input_scaler.transform(X_test_flat)
        X_test_scaled = X_test_scaled_flat.reshape(X_test.shape)

        # --- 2. Call Parent Predictor ---
        # The parent `predict` method will handle batching and inference.
        predictions_scaled = super().predict(X_test_scaled, static_features)

        # --- 3. Inverse-Scale Output Data ---
        pred_scaled_flat = predictions_scaled.reshape(-1, self.num_target_features)
        predictions_flat = self.target_scaler.inverse_transform(pred_scaled_flat)
        
        # Reshape to final output format and return
        return predictions_flat.reshape(predictions_scaled.shape)

    def _checkpoint_extra(self) -> dict:
        return {
            'adj_mx': self.adj_mx,
            'num_nodes': self.num_nodes,
            'num_input_features': self.num_input_features,
            'num_target_features': self.num_target_features,
            'input_scaler': self.input_scaler,
            'target_scaler': self.target_scaler,
        }

    def _restore_checkpoint_extra(self, extra: dict) -> None:
        if not extra:
            raise ValueError('This legacy checkpoint lacks graph/scalers; retrain and save a new checkpoint.')
        for key, value in extra.items():
            setattr(self, key, value)
    

######## Utility functions ########


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32).todense()

###### Graph WaveNet Model ######

class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, adj):
        x = torch.einsum('ncvl,vw->ncwl', (x, adj))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = NConv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GWNET(nn.Module):
    def __init__(self, config, num_nodes, input_feature_dim, output_feature_dim):
        super(GWNET, self).__init__()
        self.num_nodes = num_nodes
        self.input_feature_dim = input_feature_dim  # Now separate from output
        self.output_feature_dim = output_feature_dim  # Target KPIs dimension
        
        # Model parameters
        self.dropout = config.get('dropout', 0.3)
        self.blocks = config.get('blocks', 4)
        self.layers = config.get('layers', 2)
        self.gcn_bool = config.get('gcn_bool', True)
        self.addaptadj = config.get('addaptadj', True)
        self.adjtype = config.get('adjtype', 'doubletransition')
        self.randomadj = config.get('randomadj', True)
        self.aptonly = config.get('aptonly', False)
        self.kernel_size = config.get('kernel_size', 2)
        self.nhid = config.get('nhid', 32)
        self.residual_channels = config.get('residual_channels', self.nhid)
        self.dilation_channels = config.get('dilation_channels', self.nhid)
        self.skip_channels = config.get('skip_channels', self.nhid * 8)
        self.end_channels = config.get('end_channels', self.nhid * 16)
        self.input_window = config['input_steps']
        self.output_window = config['output_steps']
        self.device = config.get('device', torch.device('cpu'))

        # Dynamically calculate the number of layers to match receptive field to input_window
        # This logic is adapted from the LibCity implementation.
        if config.get('apt_layer', False):
            # The formula calculates the number of layers needed per block.
            # It ensures the receptive field can cover the input_window.
            self.layers = int(
                np.round(np.log2((((self.input_window - 1) / (self.blocks * (self.kernel_size - 1))) + 1)))
            )
            print(f"INFO: Dynamic layer calculation is enabled. Number of layers per block set to: {self.layers}")

        # Adjacency matrix
        self.adj_mx = config.get('adj_mx', np.eye(num_nodes))
        
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        
        self.start_conv = nn.Conv2d(
            in_channels=self.input_feature_dim,  # Use input features
            out_channels=self.residual_channels,
            kernel_size=(1, 1)
        )

        #### Calc. receptive field based on kernel size and dilation
        receptive_field = 1
        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # We calculate receptive field based on the dilation factor
                receptive_field += additional_scope * new_dilation
                new_dilation *= 2
        self.receptive_field = receptive_field

        self.cal_adj(self.adjtype)
        self.supports = [torch.tensor(i).to(self.device) for i in self.adj_mx]
        
        if self.randomadj:
            self.aptinit = None
        else:
            self.aptinit = self.supports[0] if self.supports else None
            
        if self.aptonly:
            self.supports = None

        self.supports_len = 0
        if self.supports is not None:
            self.supports_len += len(self.supports)

        if self.gcn_bool and self.addaptadj:
            if self.aptinit is None:
                if self.supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10).to(self.device),
                                             requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes).to(self.device),
                                             requires_grad=True).to(self.device)
                self.supports_len += 1
            else:
                if self.supports is None:
                    self.supports = []
                m, p, n = torch.svd(self.aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)
                self.supports_len += 1

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1, self.kernel_size), dilation=new_dilation))

                # Both spatial and temporal axes are present, so these layers use Conv2d.
                self.gate_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                 out_channels=self.dilation_channels,
                                 kernel_size=(1, self.kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, 1)))

                self.bn.append(nn.BatchNorm2d(self.residual_channels))
                new_dilation *= 2
                
                if self.gcn_bool:
                    self.gconv.append(GCN(self.dilation_channels, self.residual_channels,
                                          self.dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(
            in_channels=self.end_channels,
            out_channels=self.output_window * self.output_feature_dim,  # Use output features
            kernel_size=(1, 1),
            bias=True
        )
        

    def forward(self, inputs):
        # input shape (batch_size, input_window, num_nodes, feature_dim)
        inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window)
        # inputs: (batch_size, feature_dim, num_nodes, sequence_length)
        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            # Pad on the left to ensure causality
            x = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = inputs

        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            # (dilation, init_dilation) = self.dilations[i]
            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # (batch_size, residual_channels, num_nodes, self.receptive_field)
            # dilated convolution
            filter = self.filter_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            # parametrized skip connection
            s = x
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            s = self.skip_convs[i](s)
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            try: # First skip is initialized as 0, thus first run will give error, which we ignore
                skip = skip[:, :, :, -s.size(3):]
            except(Exception):
                skip = 0
            skip = s + skip
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            if self.gcn_bool and self.supports is not None:
                # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)
            # residual: (batch_size, residual_channels, num_nodes, self.receptive_field)
            x = x + residual[:, :, :, -x.size(3):]
            # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            x = self.bn[i](x)

        x = F.relu(skip)
         # (batch_size, skip_channels, num_nodes, self.output_dim)
        x = F.relu(self.end_conv_1(x))
        # (batch_size, end_channels, num_nodes, self.output_dim)
        x = self.end_conv_2(x)

        # Remove the last dimension of size 1
        x = x.squeeze(-1) # Shape: (batch, output_window * feature_dim, num_nodes)

        # Reshape to separate time and feature dimensions
        # New shape: (batch, output_window, feature_dim, num_nodes)
        x = x.view(x.size(0), self.output_window, self.output_feature_dim, self.num_nodes)
        x = x.permute(0, 1, 3, 2) # Permute to match the target y shape: (batch, output_window, num_nodes, feature_dim)
        return x.contiguous()   

    def cal_adj(self, adjtype):
        if adjtype == "scalap":
            self.adj_mx = [calculate_scaled_laplacian(self.adj_mx)]
        elif adjtype == "normlap":
            self.adj_mx = [calculate_normalized_laplacian(self.adj_mx).astype(np.float32).todense()]
        elif adjtype == "symnadj":
            self.adj_mx = [sym_adj(self.adj_mx)]
        elif adjtype == "transition":
            self.adj_mx = [asym_adj(self.adj_mx)]
        elif adjtype == "doubletransition":
            self.adj_mx = [asym_adj(self.adj_mx), asym_adj(np.transpose(self.adj_mx))]
        elif adjtype == "identity":
            self.adj_mx = [np.diag(np.ones(self.adj_mx.shape[0])).astype(np.float32)]
        else:
            assert 0, "adj type not defined"
