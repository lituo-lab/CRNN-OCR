import torch
import numpy as np
from tqdm import tqdm
from model import CRNN
from torch import nn,optim
from dataset import CrnnDataset
from torchvision import transforms
from torch.utils.data import DataLoader

device = 'cuda'

train_dataset = CrnnDataset(
    image_path = './data/train',
    label_path = './data/train/label.txt',
    transform = transforms.ToTensor(),
    dict_path = './data/dicts.txt')

valid_dataset = CrnnDataset(
    image_path = './data/valid',
    label_path = './data/valid/label.txt',
    transform = transforms.ToTensor(),
    dict_path = './data/dicts.txt')

train_dataloader = DataLoader(dataset=train_dataset,batch_size=300, collate_fn = train_dataset.collate_fn)
valid_dataloader = DataLoader(dataset=valid_dataset,batch_size=100, collate_fn = valid_dataset.collate_fn)


model = CRNN(32,3,28,512).to(device)
loss_fn = nn.CTCLoss(blank=0)
optimizer = optim.Adam(model.parameters())

train_loss_record = []
valid_loss_record = []

num_epochs = 1000

for epoch in range(num_epochs):

    model.train()
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    epoch_loss = []
    for i, (image,label,label_len) in loop:
        
        image,label = image.to(device), label.to(device)
        output = model(image)
        output_len = torch.full((output.shape[1],),output.shape[0],dtype=torch.long)
        label_len,output_len = label_len.to(device), output_len.to(device)
        loss = loss_fn(output,label,output_len,label_len)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        loop.set_postfix(loss=loss.item())
    train_loss_record.append(np.mean(epoch_loss))
    
    
    model.eval()
    loop = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))
    epoch_loss = []
    with torch.no_grad():
        epoch_loss = []
        for i, (image,label,label_len) in loop:
            
            image,label = image.to(device), label.to(device)
            output = model(image)
            output_len = torch.full((output.shape[1],),output.shape[0],dtype=torch.long)
            label_len,output_len = label_len.to(device), output_len.to(device)
            loss = loss_fn(output,label,output_len,label_len)
            epoch_loss.append(loss.item())
            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(loss=loss.item())
    valid_loss_record.append(np.mean(epoch_loss))


    if (epoch+1) % 50 == 0:
        torch.save(model.state_dict(), 'model.param')
        np.save('train_loss_record.npy', train_loss_record)
        np.save('valid_loss_record.npy', valid_loss_record)


np.save('train_loss_record.npy', train_loss_record)
np.save('valid_loss_record.npy', valid_loss_record)


