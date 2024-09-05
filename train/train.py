import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import random
import os

from data_loading import create_dataloader
from model import GraphAttentionNetwork

def train(model, train_loader, val_loader, num_epochs, device, nodes_per_graph=5000, sample_size=1000, save_dir='model_checkpoints'):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    wandb.init(project="particle-simulation-gat", entity="milanrohatgi")
    wandb.watch(model)

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            
            # random sampling for efficient loss
            batch_size = batch.num_graphs
            sampled_indices = [random.sample(range(i*nodes_per_graph, (i+1)*nodes_per_graph), sample_size) for i in range(batch_size)]
            sampled_indices = torch.tensor(sum(sampled_indices, []), device=device)
            
            loss = criterion(out[sampled_indices], batch.y[sampled_indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        wandb.log({"train_loss": avg_loss})

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                
                batch_size = batch.num_graphs
                sampled_indices = [random.sample(range(i*nodes_per_graph, (i+1)*nodes_per_graph), sample_size) for i in range(batch_size)]
                sampled_indices = torch.tensor(sum(sampled_indices, []), device=device)
                
                val_loss += criterion(out[sampled_indices], batch.y[sampled_indices]).item()

        avg_val_loss = val_loss / len(val_loader)
        wandb.log({"val_loss": avg_val_loss})

        if (epoch + 1) % 1 == 0:
            model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), model_path)

            full_model_path = os.path.join(save_dir, f'full_model_epoch_{epoch+1}.pth')
            torch.save(model, full_model_path)

    wandb.finish()

    final_model_path = os.path.join(save_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)

    final_full_model_path = os.path.join(save_dir, 'final_full_model.pth')
    torch.save(model, final_full_model_path)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 32
    train_loader = create_dataloader('/scratch/users/mrohatgi/MPM_processed/combined_train_flat.h5', batch_size=batch_size)
    val_loader = create_dataloader('/scratch/users/mrohatgi/MPM_processed/combined_val_flat.h5', batch_size=batch_size)

    input_dim = 17
    edge_dim = 1
    model = GraphAttentionNetwork(input_dim, edge_dim).to(device)

    num_epochs = 50
    train(model, train_loader, val_loader, num_epochs, device)
