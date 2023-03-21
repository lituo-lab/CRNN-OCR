import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset


def get_dicts(dict_path):
    data = open(dict_path).readlines()
    word2index_dicts = {}
    index2word_dicts = {}
    for i, d in enumerate(data):
        word2index_dicts[d.replace("\n", "")] = i
        index2word_dicts[i] = d.replace("\n", "")
    return word2index_dicts, index2word_dicts


class CrnnDataset(Dataset):
    def __init__(self, image_path=None, label_path=None, transform=None, dict_path=None):
        super().__init__()
        self.image_path = image_path
        self.data = open(label_path).readlines()
        self.transform = transform
        self.word2index_dict = get_dicts(dict_path)[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        img_name, label_str = self.data[index].replace("\n", "").split("###")
        image = cv2.imread(os.path.join(self.image_path, img_name))

        h, w, c = image.shape
        new_w, new_h = 200,32
        
        padding_w = int(new_w/new_h*h-w)
        
        if padding_w<0:
            padding_w=0

        mask = np.ones((h, padding_w, 3)).astype(np.uint8)*128
        image = np.concatenate([image,mask],axis=1)
        image = cv2.resize(image,(new_w,new_h))
        
        label = []
        for i in label_str:
            label.append(int(self.word2index_dict[i]))

        return self.transform(image), torch.LongTensor(label)

    def collate_fn(self, datas):
        image = []
        label = torch.LongTensor([])
        len_label = []
        for data in datas:
            image.append(data[0])
            label = torch.concat([label, data[1]])
            len_label.append(len(data[1]))

        return torch.stack(image), label, torch.tensor(len_label)


if __name__ == '__main__':
    
    train_dataset = CrnnDataset(
        image_path = './data/train',
        label_path = './data/train/label.txt',
        transform = transforms.ToTensor(),
        dict_path = './data/dicts.txt')
    
    for i, data_i in enumerate(train_dataset):
        
        image = data_i[0]
        label = data_i[1]
        print(i,image.shape,label.shape)
     
        index2word = get_dicts('./data/dicts.txt')[1]
        label_str = [index2word[i] for i in label.numpy()]
        label_str = ''.join(label_str)
        image_trs = image.numpy().transpose((1, 2, 0))
        
        plt.figure()
        plt.imshow(image_trs)
        plt.title(label)
        
        if i >= 5: break
        
    