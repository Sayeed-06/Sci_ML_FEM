import torch
import random
from torch_geometric.data import Data

class RandomPlateSymmetry:
    """
    Randomly reflects the plate mesh horizontally and/or vertically.
    Exploits the geometric symmetry of the rectangular Winkler plate.
    
    The mesh is represented by node features of shape (N, C) ordered row-major
    (X varies fastest). We reshape to (ny, nx, C), flip along the appropriate axis,
    and flatten back. This preserves the exact PyG graph topology (edge_index) 
    while perfectly mirroring the physical geometry and target outputs!
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data: Data) -> Data:
        # Check if graph has structure metadata
        if not hasattr(data, 'nx') or not hasattr(data, 'ny'):
            return data
            
        nx = int(data.nx.item() if isinstance(data.nx, torch.Tensor) else data.nx)
        ny = int(data.ny.item() if isinstance(data.ny, torch.Tensor) else data.ny)
        N = nx * ny
        
        flip_h = random.random() < self.p
        flip_v = random.random() < self.p
        
        if not flip_h and not flip_v:
            return data
            
        dims_to_flip = []
        if flip_h:
            dims_to_flip.append(1)  # X axis is dim 1 in (ny, nx)
        if flip_v:
            dims_to_flip.append(0)  # Y axis is dim 0 in (ny, nx)
            
        # 1. Flip Node Features
        if hasattr(data, 'x') and data.x is not None:
            x_reshaped = data.x.view(ny, nx, -1)
            x_flipped = torch.flip(x_reshaped, dims=dims_to_flip)
            data.x = x_flipped.reshape(N, -1)
            
        # 2. Flip Node Targets
        if hasattr(data, 'y') and data.y is not None:
            y_reshaped = data.y.view(ny, nx, -1)
            y_flipped = torch.flip(y_reshaped, dims=dims_to_flip)
            data.y = y_flipped.reshape(N, -1)
            
        # 3. Flip Physical Node Positions
        if hasattr(data, 'pos') and data.pos is not None:
            pos_reshaped = data.pos.view(ny, nx, -1)
            pos_flipped = torch.flip(pos_reshaped, dims=dims_to_flip)
            data.pos = pos_flipped.reshape(N, -1)
            
        # 4. Flip Edge Attributes (dx, dy, length)
        # edge_attr: [dx_norm, dy_norm, length_norm]
        # Since we flipped the nodes, the edge that connects node i to node j
        # now represents a connection in the opposite physical direction relative to that axis.
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if flip_h:
                data.edge_attr[:, 0] *= -1.0
            if flip_v:
                data.edge_attr[:, 1] *= -1.0
                
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


class RandomLoadJitter:
    """
    Injects small Gaussian noise into the `q_local` feature to prevent exact 
    value memorization and encourage generalisation to unseen load intensities.
    The local load is stored in `data.x[:, 5]` based on our node feature schema.
    """
    def __init__(self, std: float = 0.02, p: float = 0.5):
        self.std = std
        self.p = p

    def __call__(self, data: Data) -> Data:
        if random.random() < self.p and hasattr(data, 'x') and data.x is not None:
            # Inject noise specifically to the q_local column (index 5)
            # data.x shape is (N, 11).
            noise = torch.randn_like(data.x[:, 5]) * self.std
            data.x[:, 5] += noise
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(std={self.std}, p={self.p})'
