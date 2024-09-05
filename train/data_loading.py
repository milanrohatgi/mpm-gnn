import h5py
import torch
from torch_geometric.data import Data, Batch

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            self.num_graphs = len(f.keys())

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            graph_data = f[f'graph_{idx}']
            x = torch.FloatTensor(graph_data['x'][:])
            y = torch.FloatTensor(graph_data['y'][:])
            edge_index = torch.LongTensor(graph_data['edge_index'][:]).t().contiguous()
            edge_attr = torch.FloatTensor(graph_data['edge_attr'][:]).unsqueeze(1)

        return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

def collate_fn(batch):
    return Batch.from_data_list(batch)

def create_dataloader(file_path, batch_size, num_workers=4):
    dataset = GraphDataset(file_path)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
