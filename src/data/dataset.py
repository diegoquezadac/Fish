from torch.utils.data import DataLoader,Dataset, WeightedRandomSampler

class CustomDataset(Dataset):
    def __init__(self, X, y, tokenizer):
        self.x_train = X
        self.y_train = y
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

    def get_sampler(self):
        class_weights = [0.1, 0.9] # Allows balanced batches
        sample_weights = [0] * len(self)
        for idx, (text, label) in enumerate(self):
            sample_weights[idx] = class_weights[label]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        return sampler