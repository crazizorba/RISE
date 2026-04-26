import torch
from torch.utils.data import Dataset, Sampler
import random
from collections import defaultdict

class DynamicsDataset(Dataset):
    def __init__(self, trajectories):
        """
        trajectories: list of dicts, each dict has 'states', 'actions', 'next_states', 'task_id'
        """
        self.data = []
        for traj in trajectories:
            states = traj['states']
            actions = traj['actions']
            next_states = traj['next_states']
            task_id = traj['task_id']
            for i in range(len(states)):
                self.data.append({
                    'state': states[i],
                    'action': actions[i],
                    'next_state': next_states[i],
                    'task_id': task_id
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class TaskCentricSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        # Group indices by task_id
        self.task_groups = defaultdict(list)
        for idx, item in enumerate(dataset):
            self.task_groups[item['task_id']].append(idx)
        self.tasks = list(self.task_groups.keys())

    def __iter__(self):
        # Shuffle tasks
        random.shuffle(self.tasks)
        for task in self.tasks:
            indices = self.task_groups[task]
            # Shuffle indices within task
            random.shuffle(indices)
            # Yield batches from this task
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]

    def __len__(self):
        total_batches = 0
        for task in self.tasks:
            total_batches += (len(self.task_groups[task]) + self.batch_size - 1) // self.batch_size
        return total_batches