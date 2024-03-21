import pandas as pd, numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms as T
import matplotlib.pyplot as plt
import torch, argparse, timm
from data import CustomData
from utils import tensor_2_im, display
from dataloader import get_dl
from train import train, train_setup
import torch.nn as nn
from inference import *
from plots import *





def run(args):
    
    mean, std = [0.2250, 0.2250, 0.2250], [0.2505, 0.2505, 0.2505]
    tfs = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean = mean, std = std)])
    ds = CustomData(root=args.data_path, transformation=tfs)
    print("shapes of data: ", ds[0][0].shape)


    # Visualize
    display(ds=ds, rows=args.rows, im_num=args.number_im )
    
    # dataloader
    tr_dl, val_dl, ts_dl, classes = get_dl(root=args.data_path, transformations=tfs, bs=args.batch)
    
    
    # train 

    model = timm.create_model(args.model, pretrained=True, num_classes = len(ds.classes))
    device, model, optimizer, criterion, epochs = train_setup(model)
    results = train(model, tr_dl, val_dl, criterion, epochs, optimizer, device)
    
    # Plots
    plot_results(results['tr_loss_cs'], results['val_loss_cs'], results['tr_acc_sc'], results['val_acc_sc'])


    # Inference
        # Load the model
    model.load_state_dict(torch.load(args.inference))
    # Pass the dataset instead of the DataLoader
    inference(model.to(device), device, ts_dl, 20, 4, list(classes.keys()), 224)
if __name__ =="__main__":
    #main
    parser = argparse.ArgumentParser(description="Skin Lesion Project")
    parser.add_argument("-dt","--data_path", type=str, default="D:/Data/Datasets/Skin_lesion_Pixel_data/meta_deta.csv", help="Path to data")
    
    # visualize
    parser.add_argument("-nm", "--number_im", type=str, default=20, help="number of images")
    parser.add_argument("-rw", "--rows", type=str, default=4, help="rows of images")
    parser.add_argument("-bs", "--batch", type=str, default=32, help="batch_size")  
    
    # Train
    parser.add_argument("-m","--model", default="resnet18", help="Model")
    
    parser.add_argument("-mi", "--num_image", type=str, default= 20 , help="numbe images")
    parser.add_argument("-in", "--inference", type=str, default= "D:/portfolio/Classification/Skin_lesion/Images/model_best_model.pth" , help="numbe images")


    
    
    args = parser.parse_args()
    run(args)