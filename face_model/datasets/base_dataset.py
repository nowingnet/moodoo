from abc import ABC
from torch.utils.data import Dataset

class BaseDataset(Dataset, ABC):
    def __init__(self):
        self.name = "BaseDataset"