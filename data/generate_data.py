import os
import random
from PIL import Image,ImageFont,ImageDraw


def make_data(data_number, path):
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    f = open(path + "/label.txt","w")
    
    data = open("./dicts.txt",'r').readlines()[1:]
    data = [i.replace("\n","") for i in data]
    
    font = ImageFont.truetype("./TIMES.TTF", 35)
    
    for i in range(data_number):
        
        text  = "".join(random.sample(data,random.randint(1,15)))
        im = Image.new("RGB", (int(len(text)*20*0.95),random.randint(43,50)), (255, 255, 255))
        dr = ImageDraw.Draw(im)
        dr.text((3, 1), text, font=font, fill="#000000")
        im.save(os.path.join(path,str(i).zfill(5)+".jpg"))
        
        f.write(str(i).zfill(5)+".jpg"+"###"+text+"\n")
    f.close()    


# train data
train_data_number = 20
train_data_path = "./train"
make_data(train_data_number,train_data_path)


# valid data
valid_data_number = 5
valid_data_path = "./valid"
make_data(valid_data_number,valid_data_path)


    

