import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dynamics_model import DynamicsModel
from dataset import DynamicsDataset, TaskCentricSampler
import numpy as np
from tqdm import tqdm

# Generate dummy data
def generate_dummy_data(num_trajectories=100, traj_length=50, state_dim=10, action_dim=5, num_tasks=5):
    trajectories = []
    for task_id in range(num_tasks):
        for _ in range(num_trajectories // num_tasks):
            states = []
            actions = []
            next_states = []
            state = np.random.randn(state_dim)
            for _ in range(traj_length):
                action = np.random.randn(action_dim)
                # Simple dynamics: next_state = state + action * 0.1 + noise
                next_state = state + action * 0.1 + np.random.randn(state_dim) * 0.01
                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                state = next_state
            trajectories.append({
                'states': states,
                'actions': actions,
                'next_states': next_states,
                'task_id': task_id
            })
    return trajectories

def train_model():
    # Hyperparams
    state_dim = 10
    action_dim = 5
    batch_size = 32
    epochs = 10
    lr = 1e-3

    # Data
    trajectories = generate_dummy_data()
    dataset = DynamicsDataset(trajectories)
    sampler = TaskCentricSampler(dataset, batch_size)
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    # Model
    model = DynamicsModel(state_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader):
            states = torch.tensor(np.array(batch['state']), dtype=torch.float32)
            actions = torch.tensor(np.array(batch['action']), dtype=torch.float32)
            next_states = torch.tensor(np.array(batch['next_state']), dtype=torch.float32)

            pred_next_states = model(states, actions)
            loss = criterion(pred_next_states, next_states)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader)}")

if __name__ == "__main__":
    train_model()