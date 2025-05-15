from torch.utils.data import Dataset as torchDataset, ConcatDataset


class SequentialMultiTaskDataset(torchDataset):
    def __init__(self, datasets: List[Tuple[str, torchDataset]]):
        """
        A dataset that combines multiple task datasets sequentially.
        
        Args:
            datasets: List of tuples containing (task_name, dataset) pairs
        """
        assert len(datasets) > 0, "At least one dataset is required."
        self.tasks_names = np.array([name for name, _ in datasets])
        self.tasks_datasets = ConcatDataset([d for _, d in datasets])
        tasks_num_samples = [len(d) for _, d in datasets]
        self.task_indices = torch.concatenate([torch.zeros(1), torch.cumsum(tasks_num_samples, out=self.task_indices)])

    def __len__(self):
        return self.task_indices[-1]
    
    def __getitem__(self, index):
        batch_task_idx = torch.searchsorted(self.task_indices, index)
        batch_task_name = self.tasks_names[batch_task_idx]
        batch_task_dataset =self.tasks_datasets[index]
        mask = torch.ones_like(index, dtype=torch.bool)
        
        return mask, batch_task_name, batch_task_dataset


class ParallelMultiTaskDataset(torchDataset):
    def __init__(self, datasets: List[Tuple[str, torchDataset]]):
        assert len(datasets) > 0, "At least one dataset is required."
        assert len(set([len(dataset) for _, dataset in datasets])) == 1, "All datasets must have the same length."
        self.datasets = datasets

    def __len__(self):
        return max([len(dataset) for dataset in self.tasks_datasets])
    
    def __getitem__(self, index):
        batch_task_dataset = []
        batch_task_name = []
        batch_task_mask = []
        for task_name, dataset in self.tasks_datasets:
            if index >= len(dataset):
                dataset_item = dataset[0]
                dataset_mask = torch.zeros(1, dtype=torch.bool)
            else:
                dataset_item = dataset[index]
                dataset_mask = torch.ones(1, dtype=torch.bool)

            batch_task_dataset.append(dataset_item)
            batch_task_name.append(task_name)
            batch_task_mask.append(dataset_mask)

        return torch.concatenate(batch_task_mask), np.concatenate(batch_task_name), torch.concatenate(batch_task_dataset)

