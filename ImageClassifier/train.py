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
    parser = argparse.ArgumentParser(description = 'Training')
    parser.add_argument('--data_dir', action ='store')
    parser.add_argument('--arch', dest = 'arch', default = 'vgg19')
    parser.add_argument('--learning_rate', dest = 'learning_rate', default = '0.001')
    parser.add_argument('--epochs', dest = 'epochs', default = '3')
    parser.add_argument('--gpu', action = 'store', default = 'gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
        
    return parser.parse_args()
        
def train(model, criterion, optimizer, epochs, trainloader, validloader, gpu):
    
    print('hello')
    print_every = 10
    steps = 0
    
    
    for e in range(epochs):
        running_loss = 0
        
        for inputs, labels in trainloader:
            steps+=1
            if gpu == 'gpu':
                model.cuda()
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                model.cpu()
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                
                model.eval()
                valid_loss = 0
                accuracy = 0
                
                with torch.no_grad():
                    for inputs, labels in validloader:
                        if gpu == 'gpu':
                            model.cuda()
                            inputs, labels = inputs.to('cuda'), labels.to('cuda')
                        else:
                            
                            model.cpu()
                            
                        logps = model.forward(inputs)
                        batch_loss=criterion(logps, labels)
                        
                        valid_loss += batch_loss.item()
                        
                        
                        ##accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim =1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {e+1}/{epochs}.. "
                f"Train loss: {running_loss/print_every:.3f}.. "
                f"Valid loss: {valid_loss:.3f}.. "
                f"Valid accuracy: {accuracy/len(validloader):.3f}")
                        
                train_loss = running_loss/print_every
                valid_loss = valid_loss/len(validloader)
                    
                        
            running_loss = 0
            model.train()
    
def save_checkpoint(path, model, optimizer, args, classifier):
    checkpoint = {'arch': args.arch,
                 'model_state_dict' : model.state_dict(),
                 'optimizer':optimizer.state_dict(),
                 'classifier': classifier,
                 'epochs': args.epochs,
                 'learning_rate': args.learning_rate,
                 'class_to_idx' : model.class_to_idx,
                 }
        
    torch.save(checkpoint, path)
        
  
        
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained = True)
    model.classifier = checkpoint(['classifier'])
    learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])
    return model                   

def main():
    args = parser_args()

    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

    trainset = datasets.ImageFolder(train_dir, transform = train_transform)
    testset = datasets.ImageFolder(test_dir, transform = test_transform)
    validset = datasets.ImageFolder(valid_dir, transform = valid_transform)                                    


    trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=64,
                                          shuffle=True)
    testloader = torch.utils.data.DataLoader(testset,
                                          batch_size=64,
                                          shuffle=True)
    validloader = torch.utils.data.DataLoader(validset,
                                          batch_size=64,
                                          shuffle=True)
    

    
    with open('cat_to_name.json','r') as f:
        cat_to_name = json.load(f)
      
    
    model = getattr(models, args.arch)(pretrained = True)
    
    
    for param in model.parameters():
        model.requires_grad = False
        
    classifier = nn.Sequential(nn.Linear(25088,4096),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(4096, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512,102),
                             nn.LogSoftmax(dim = 1))
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr = float(args.learning_rate))
    
    if args.gpu == 'gpu':
        model.cuda()
        
    else:
        model.cpu()
    
    epochs = int(args.epochs)
    class_index = trainset.class_to_idx
    gpu = args.gpu
    print(model)
    
    from workspace_utils import keep_awake

    for i in keep_awake(range(1)):
        train(model, criterion, optimizer, epochs, trainloader, validloader, gpu)
    model.class_to_idx = class_index
#     path = args.save_dir
    save_checkpoint(args.save_dir, model, optimizer, args, classifier)  

    
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    

    
    
                
          
                        
                        
                        
                        
                    

    

