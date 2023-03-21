import os
import cv2
import torch
import numpy as np
from model import CRNN
import matplotlib.pyplot as plt
from torchvision import transforms


def get_dicts(dict_path):
    data = open(dict_path).readlines()
    word2index_dicts = {}
    index2word_dicts = {}
    for i, d in enumerate(data):
        word2index_dicts[d.replace("\n", "")] = i
        index2word_dicts[i] = d.replace("\n", "")
    return word2index_dicts, index2word_dicts


if __name__ == "__main__":

    model = CRNN(32, 3, 28, 512)
    model.load_state_dict(torch.load('model.param'))

    with torch.no_grad():

        for im in os.listdir("./data/valid"):

            if im[-3:] == 'jpg':
                img = cv2.imread(os.path.join("./data/valid", im))
            else:
                continue

            h, w, c = img.shape
            new_w, new_h = 200, 32
            padding_w = int(new_w/new_h*h-w)

            if padding_w < 0:
                padding_w = 0

            mask = np.ones((h, padding_w, 3)).astype(np.uint8)*128
            image = np.concatenate([img, mask], axis=1)
            image = cv2.resize(image, (new_w, new_h))

            image = transforms.ToTensor()(image)
            image = image.unsqueeze(dim=0).float()
            output = model(image).squeeze().numpy()

            out = np.argmax(output, axis=1)

            index2word_dicts = get_dicts("./data/dicts.txt")[1]

            char_list = []
            for i in range(len(out)):
                if out[i] != 0 and (not (i > 0 and out[i - 1] == out[i])):
                    char_list.append(out[i])

            out_str = "".join([index2word_dicts[i.item()] for i in char_list])

            plt.figure()
            plt.imshow(img)
            plt.title(out_str)
            plt.show()

