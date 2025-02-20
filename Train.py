import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from Database_Training import CoCoRaHS_Precip_Obs
from Database_Training import Surface_Maps_Training
from Database_Training import Combined_Datasets
from Model import AI_Processor
import datetime
import matplotlib as plt
import os

def save_output_images(predictions, epoch, batch_idx, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, pred in enumerate(predictions):
        pred_image = pred.detach().cpu().numpy
        pred_image = (pred_image-pred_image.min()/(pred_image.max()-pred_image.min))
        plt.imshow(pred_image, camp='viridis')
        plt.axis('off')
        plt.savefig(f'{output_dir}\\pred_epoch{epoch}_batch{batch_idx}_image{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

end_date = datetime.datetime(2015, 12, 31, 21)
start_date = datetime.datetime(1998, 6, 10, 00)
transform = None
precip_dataset = CoCoRaHS_Precip_Obs(start_date, end_date, transform)
Surface_map_dataset = Surface_Maps_Training(start_date, end_date, transform)
combined_dataset = Combined_Datasets(precip_dataset, Surface_map_dataset)
data_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AI_Processor(num_classes = 10).to(device)
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
        if batch_idx % 10 == 0:
            save_output_images(outputs, epoch, batch_idx, output_dir)
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1/num_epochs}, Loss: {running_loss/len(data_loader)}')    
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