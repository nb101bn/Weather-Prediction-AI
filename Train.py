import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from Database_Training.CoCoRaHS_Precip_Obs import CoCoRaHS_Dataset
from Database_Training.Surface_Maps_Training import SurfaceMap_Dataset
from Database_Training.Combined_Datasets import Combined_Datasets
from Model.AI_Processor import CNN
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

def save_output_images(predictions, epoch, batch_idx, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, pred in enumerate(predictions):
        pred_image = pred.detach().cpu().numpy()
        pred_image = (pred_image - pred_image.min())  / (pred_image.max() - pred_image.min)
        plt.imshow(pred_image, camp='viridis')
        plt.axis('off')
        plt.savefig(f'{output_dir}\\pred_epoch{epoch}_batch{batch_idx}_image{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

end_date = datetime.datetime(2015, 12, 31, 21)
start_date = datetime.datetime(2006, 1, 1, 00)
transform = transforms.ToTensor()
hours_back = 6
precip_dataset = CoCoRaHS_Dataset(start_date, end_date, transform)
Surface_map_dataset = SurfaceMap_Dataset(start_date, end_date, hours_back, transform)
combined_dataset = Combined_Datasets(precip_dataset, Surface_map_dataset)
data_loader = DataLoader(combined_dataset, batch_size=8, shuffle=True, pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    output_dir = f'C:\\Users\\natha\\Documents\\SavedOutput'
    for batch_idx, (surface_maps, precip_maps) in enumerate(data_loader):
        surface_maps = surface_maps.to(device)
        precip_maps = precip_maps.to(device)
        optimizer.zero_grad()
        outputs = model(surface_maps)
        loss = criterion(outputs, precip_maps)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        running_loss += loss.item()
        if batch_idx % 10 == 0:
            save_output_images(outputs, epoch, batch_idx, output_dir)
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(data_loader)}')
    avg_running_loss = running_loss/len(data_loader)
    torch.save(model.state_dict(), f'Checkpoint_Epoch_{epoch+1}.pth')
    print(f'Epoch {epoch+1}/{num_epochs}, average loss: {avg_running_loss}')
    torch.save(model.state_dict(), f'Checkpoint_Epoch_{epoch+1}.pth')
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_surface_maps, val_precip_maps in data_loader:
            val_surface_maps = val_surface_maps.to(device)
            val_precip_maps = val_precip_maps.to(device)
            val_outputs = model(val_surface_maps)
            loss = criterion(val_outputs, val_precip_maps)
            val_loss += loss.item()
    avg_val_loss = val_loss/len(data_loader)
    print(f'Validation Loss after after epoch {epoch+1}/{num_epochs}: {avg_val_loss}')
    model.train()
    '''
    model.train()   
    running_loss = 0.0
    for surface_maps, precip_maps in data_loader:
        optimizer.zero_grad()
        outputs = model(surface_maps)
        loss = criterion(outputs, precip_maps)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1/num_epochs}, Loss: {running_loss/len(data_loader)}')
    '''