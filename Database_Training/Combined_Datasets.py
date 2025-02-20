from torch.utils.data import Dataset
import torch

class Combined_Datasets(Dataset):
    def __init__(self, precip_dataset, surface_map_dataset):
        self.precip_dataset = precip_dataset
        self.surface_map_dataset = surface_map_dataset
    
    def __len__(self):
        return len(self.precip_dataset)
    
    def __getitem__(self, idx):
        precip_map = self.precip_dataset[idx]
        surface_maps = self.surface_map_dataset[idx]
        if isinstance(surface_maps, list):
            surface_maps_stack = torch.stack(surface_maps)
        else:
            raise ValueError(f'Expected to get Surface Maps to be a list but got {type(surface_maps)}')
        return surface_maps_stack, precip_map