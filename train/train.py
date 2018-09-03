# Imports here
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import sys

import argparse

def get_input_args():
    """
    Retrieves and parses command line arguments. 
        9 command line arguments are created:
            data_dir - path to dataset files
            save_dir - path to save checkpoint (default- current path)
            arch - model architecture, support: 'vgg16, 'resnet152', 'densenet161'
            lr - Learning rate (default - 0.01)
            hidden_layers - sizes for hidden layers (in string), for e.g "1024,512" or "1024" 
            output_size - output size (default= 102)
            epochs - training epochs (default= 1)
            print_every - print lossess and accuracy every print_every steps during training (default = 40 steps)
            gpu - use gpu processing (default= False)
        Parameters:
            None - use module to create & store command line arguments
        Returns:
            parse_args() - data structure that stores the command line arguments object
    """
    parser = argparse.ArgumentParser(description='Get NN arguments')
    #Define arguments
    parser.add_argument('data_dir', type=str, help='mandatory data directory')
    parser.add_argument('--save_dir', default='', help='Directory to save checkpoint.')
    parser.add_argument('--arch', default='resnet152', help='architecture, options: vgg16, resnet152, densenet161')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate' )
    parser.add_argument('--hidden_layers', default='1024', type=str, help='hidden layer sizes, for e.g 1024,512')
    parser.add_argument('--output_size', default=102, type=int, help='output_size')
    parser.add_argument('--epochs', default=1, type=int, help='training epochs')
    parser.add_argument('--print_every', default=40, type=int, help='print every')
    parser.add_argument('--gpu', default=False, action='store_true', help='use GPU processing')
    
    return parser.parse_args()


def loaders(data_dir):
    """
    Load dataset. Returns dataloaders
        Parameters:
            data_dir - dataset file path 
        Returns:
            trainloaders, testloaders, validloaders, class_to_idx
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.RandomRotation((-90,90)),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                           transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])


    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform= test_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform= valid_transforms)

    #Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    
    #class to index
    class_to_idx = train_datasets.class_to_idx
    
    return trainloaders, testloaders, validloaders, class_to_idx

def build_classifier(input_size, hidden_layers, output_size):
    """
    build classifier based on input size, hidden layer size, and output size
        Parameters:
            input_size - input size (int or string)
            hidden_layers - hidden layers size in string, e.g "1024,512"
            output_size - output size (int or string)
        Returns:
            classifier 
    """
    
    from collections import OrderedDict
    
    hidden_layers_param = hidden_layers.split(',')
    hidden_layers_param = [int(h) for h in hidden_layers_param]
    hidden_layers_param.append(output_size)
    
    hidden_layers = nn.ModuleList([nn.Linear(input_size,hidden_layers_param[0])])
    hidden_layers.extend(nn.Linear(h1,h2) for h1,h2 in zip(hidden_layers_param[:-1], hidden_layers_param[1:]))
    
    layers = OrderedDict()
    
    for i in range(len(hidden_layers)):
        layer_id = i+1
        if i ==0:
            layers.update({'fc{}'.format(layer_id): hidden_layers[i]})
        else:
            layers.update({'relu{}'.format(layer_id): nn.ReLU()})
            layers.update({'dropout{}'.format(layer_id): nn.Dropout(p=0.5)})
            layers.update({'fc{}'.format(layer_id): hidden_layers[i]})
                          
    layers.update({'output': nn.LogSoftmax(dim=1)})
    classifier = nn.Sequential(layers)
    return classifier
     
    
def build_model(arch, hidden_layers, output_size):
    """
    build model based on selected architecture, hidden layer size, and output size
        Parameters:
            arch - model architecture, support : 'vgg16, 'resnet152', 'densenet161'
            hidden_layers - hidden layers size in string, e.g "1024,512"
            output_size - output size (int or string)
        Returns:
            model 
    """
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        
        #freeze model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        #update final classifier layer as per hidden_layers
        model.classifier = build_classifier(25088,hidden_layers,output_size)
        
    elif arch == 'resnet152':
        model = models.resnet152(pretrained=True)
        
        #freeze model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        #update final classifier layer as per hidden_layers
        model.fc = build_classifier(2048,hidden_layers,output_size)
    
    elif arch == 'densenet161':
        model = models.densenet161(pretrained=True)
        
        #freeze model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        #update final classifier layer as per hidden_layers
        model.classifier = build_classifier(2208,hidden_layers,output_size)
    
    else:
        print('The {} architecture was not recognized. The supported architecture were : \'vgg16\', \'resnet152\', or \'densenet161\''.format(arch))
        sys.exit()
    
    
    return model
  
    
def train(model, trainloader, validloader,criterion, optimizer, scheduler, epochs=1, print_every=40, gpu=False):
    """
    train model, return trained model, and optimizer
        Parameters:
            model - model architecture, support : 'vgg16, 'resnet152', 'densenet161'
            trainloader - trainloaders 
            validloader - validloaders
            criterion - deep learning pytorch criterion
            optimizer - optimizer for training
            scheduler - scheduler for adjusting learning rate during training
            epochs - training epochs (default=1)
            print_every - print lossess and accuracy every print_every steps (default = 40 steps)
            gpu - use GPU (default=False)
        Returns:
            model, optimizer 
    """
    
    steps = 0
    running_loss = 0
    
    if gpu and torch.cuda.is_available():
        device = 'cuda:0'
        print('\nTraining with GPU ')
        print('===================')
    elif gpu and not torch.cuda.is_available():
        device = 'cpu'
        print('\nGPU is not detected, continue training with CPU')
        print('================================================')
    else:
        device = 'cpu'
        print('\nTraining with CPU')
        print('==================')
    
    model.to(device)

    for e in range(epochs):
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            #clear gradients
            optimizer.zero_grad()

            #foward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, valid_acc = validation(model, validloader, criterion, device)
                
                
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                  
                print("Epochs: {}/{}..".format(e+1, epochs),
                      "lr: {}..".format(current_lr),
                     "Train loss: {:.4f}..".format(running_loss/print_every),
                      "Val loss: {:.4f}..".format(valid_loss),
                     'Val Acc: {:.2f} % '.format( valid_acc))
                    
                running_loss = 0
                model.train()
                scheduler.step(valid_loss)
    
    return model, optimizer
                
def validation(model, loader, criterion, device):
    """
     model inference during training 
        Parameters:
            model - model architecture, support : 'vgg16, 'resnet152', 'densenet161'
            loader - dataloaders for e.g validloaders
            criterion - deep learning criterion
            device - processing in cpu ('cpu') or gpu ('cuda:0')
        Returns:
            loss, accuracy 
    """
    loss = 0
    correct = 0
    total_n = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        loss += criterion(output,labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        correct += equality.type(torch.FloatTensor).sum().item()
        total_n += labels.size(0)
    
    return loss/len(loader), 100*correct/total_n 

def save(model,arch, save_dir, input_size, hidden_layers, output_size, epochs, optimizer,scheduler,criterion,class_to_idx):
    """
    saving trained model, return None
        Parameters:
            model - model architecture, support : 'vgg16, 'resnet152', 'densenet161'
            arch - model architecture, support : 'vgg16, 'resnet152', 'densenet161'
            save_dir - saving file path for 'checkpoint.pth'
            input_size - int or string input size
            hidden_layers - string of hidden layer size, for e.g "1024, 512"
            output_size - int or string output_size
            epochs - training epoch
            optimizer - deeplearning optimizer
            scheduler - learning rate adjuster
            criterion - deep learning criterion
            class_to_idx - mapping class label to index number
        Returns:
            None - use for saving trained model
    """
    checkpoint = {'epochs': epochs,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler,
                  'criterion': criterion,
                  'class_to_idx' : class_to_idx,
                  'input_size': input_size,
                  'hidden_layers': hidden_layers,
                  'output_size': output_size,
                  'arch': arch
        }
    if len(save_dir) == 0:
        save_path = save_dir + 'checkpoint.pth'
    else:
        save_path = save_dir + '/checkpoint.pth'
    torch.save(checkpoint,save_path)

    
def main():
    in_args = get_input_args()
    trainloaders, testloaders, validloaders, class_to_idx = loaders(in_args.data_dir)
     
    #build Model
    model = build_model(in_args.arch, in_args.hidden_layers, in_args.output_size)

    #define criterion, optimizer, and scheduler
    criterion = nn.NLLLoss()
    
    #define optimizer and scheduler
    if in_args.arch == 'resnet152':
        optimizer = optim.SGD(model.fc.parameters(), lr=in_args.lr, momentum=0.9)
    else:
        optimizer = optim.SGD(model.classifier.parameters(), lr=in_args.lr, momentum=0.9)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    #train
    model, optimizer = train(model, trainloaders, validloaders,criterion, optimizer, scheduler, epochs=in_args.epochs, print_every=in_args.print_every, gpu=in_args.gpu)
    
    #input_size dict based on architecture
    input_size = {'vgg16': 25088, 'resnet152': 2048, 'densenet121': 2208}
    
    #save model
    save(model,in_args.arch,in_args.save_dir,input_size[in_args.arch], in_args.hidden_layers, in_args.output_size, 
         in_args.epochs, optimizer,scheduler,criterion,class_to_idx)
     
    pass

if __name__ == '__main__':
    main()
    
    
    