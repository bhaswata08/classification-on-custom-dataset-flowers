import argparse
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import json


def parser_args():
    parser = argparse.ArgumentParser(description = 'Test')
    parser.add_argument('--filepath', dest = 'filepath', default = 'flowers/test/1/image_06743.jpg')
    
    parser.add_argument('--checkpoint', action = 'store', default = 'checkpoint.pth')
    parser.add_argument('--top_k', dest = 'top_k', default = '3')
    parser.add_argument('--gpu', action = 'store', default = 'gpu')
    parser.add_argument('--category_names', dest = 'catagory_names', default = 'cat_to_name.json')
    return parser.parse_args()

def process_image(image):
    
    image.thumbnail((256,256))
    #crop
    width, height = image.size
    left = (256 - 224)/2
    top = (256 - 224)/2
    right = (256 + 224)/2
    bottom = (256 + 224)/2

    image = image.crop((left, top, right, bottom))
    
    # Normalize
    image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    image = ((image - mean)/std)
    
    # Move color channels
    image = np.transpose(image, (2, 0, 1))
    
    return image


def predict(image_path, model, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    args = parser_args()
    if args.gpu == 'gpu':
        model.to('cuda')
    else:
        model.to('cpu')
        
    
    img = Image.open(image_path)
    img = process_image(img)
    img = torch.from_numpy(img)
    
    img = img.unsqueeze_(0)
    img = img.float()
    img = img.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    with torch.no_grad():
        logps = model.forward(img)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk)
    
    return top_class, top_p

def load_cat_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names

  
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained = True)
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def main():
    
    args = parser_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    
    cat_to_name = load_cat_names(args.catagory_names)
    image_path = args.filepath
    top_class, top_p = predict(image_path, model, int(args.top_k))
    
    names = [cat_to_name.get(str(index)) for index in top_class.cpu().numpy()[0]]
    probs = [x for x in top_p.cpu().numpy()[0]]
    
    
    print(names)
    print(probs)
    
    for i in range(len(names)):
        print(f'{names[i]} : {probs[i]:.3f}')
        
    

    



if __name__ == '__main__':
    main()

