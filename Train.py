import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from Database_Training import CoCoRaHS_Precip_Obs
from Database_Training import Surface_Maps_Training
from Model import AI_Processor
import datetime

end_date = datetime.datetime(2015, 12, 31, 21)
start_date = datetime.datetime(1998, 6, 10, 00)
transform = None
precip_dataset = CoCoRaHS_Precip_Obs(start_date, end_date, transform)
Surface_map_dataset = Surface_Maps_Training(start_date, end_date, transform)