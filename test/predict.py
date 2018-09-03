# Imports here
import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from train import build_classifier, build_model


def get_input_args():
    """
        5 command line arguments are created:
            input - path to image file to predict
            checkpoint - path to checkpoint file
            top_k - Top k label with most probabilities (default- 1)
            cat - path to json file for mapping flower names (default- None)
            gpu - select GPU processing (default - False)
        Returns:
            parse_args() - store data structure
    """
    parser = argparse.ArgumentParser(description='Get arguments')
    
    #Define arguments
    parser.add_argument('input', type=str, help='image file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file to load')
    parser.add_argument('--top_k', default=1, type=int, help='default top_k results')
    parser.add_argument('--gpu', default=False, action='store_true', help='use GPU processing')
    parser.add_argument('--cat', default='', type=str, help='default category name json file path' )
    return parser.parse_args()

def process_image(image):
    ''' 
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    from PIL import Image
    import torch
    
    img = Image.open(image)
    #resize to 256 pixels
    img = img.resize((256,256))
    width, height = img.size
    new_width = 224
    new_height = 224
    
    #crop center 22x224
    left = np.ceil((width - new_width)/2.)
    top = np.ceil((height - new_height)/2.)
    right = np.floor((width + new_width)/2.)
    bottom = np.floor((height + new_height)/2.)
    img = img.crop((left,top,right,bottom))
    
    #convert to array --> shape (224,224,3)
    img = np.array(img)
    
    #normalizing to range [0,1]
    img = img /255.0
    
    #normalizing to specified mean and std
    norm_mean = np.array([0.485,0.456,0.406])
    norm_std = np.array([0.229, 0.224, 0.225])
    img = (img - norm_mean) / norm_std
    
    #return a transpose to shape (3,224,224) for pytorch input
    return torch.from_numpy(img.transpose(2,0,1))
    
def predict(image_path, model, gpu=None, topk=1):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
     
    model.eval()
    image = process_image(image_path)
    image = image.unsqueeze(0).float()
    with torch.no_grad():
        if gpu and torch.cuda.is_available():
            model.cuda()
            image = image.cuda()
            print('\nGPU processing')
            print('===============')
        elif gpu and not torch.cuda.is_available():
            print('\nGPU is not detected, continue with CPU PROCESSING')
            print('==================================================')
        else:
            print('\nCPU processing')
            print('===============')
        
        output = model.forward(image)
        ps = torch.exp(output)
        probs, classes = torch.topk(ps,topk)
        probs = np.array(probs).flatten()
        classes = np.array(classes).flatten()
        
        #idx_to_class
        idx_to_class = { v:k for k,v in model.class_to_idx.items()}
        classes = [idx_to_class[cls] for cls in classes]
   
    return probs, classes

def print_predict( probs, classes,cat_names):
    """
    Prints predictions. Returns Nothing
    Parameters:
        classes - list of predicted classes
        probs - list of probabilities 
    Returns:
        None - Use to print predictions
    """
    import json

    if cat_names:  
        with open(cat_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[k] for k in classes]
    predictions = list(zip(classes, probs))
    
    print("The predicted image is....\n")
    for i in range(len(predictions)):
        print('{} : {:.3%}'.format(predictions[i][0], predictions[i][1]))
    print("\n")
    pass

def load_model(checkpoint_path):
    """
    Load model. Returns loaded model
    Parameters:
        checkpoint_path - checkpoint file path 
    Returns:
        model - loaded model
    """
        
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    input_size = checkpoint['input_size']
    hidden_layers = checkpoint['hidden_layers']
    output_size = checkpoint['output_size']
    arch = checkpoint['arch']
    
    model = build_model(arch, hidden_layers, output_size)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    #define optimizer and scheduler
    if arch == 'resnet152':
        optimizer = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)
    else:
        optimizer = optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9)
        
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.optimizer = optimizer
    
    model = model.eval()
    
    return model

def main():
    in_args = get_input_args()
    model = load_model(in_args.checkpoint)
    probs, classes = predict(in_args.input, model,in_args.gpu, in_args.top_k)
    print_predict(probs, classes, in_args.cat)
    pass

if __name__ == '__main__':
    main()
    