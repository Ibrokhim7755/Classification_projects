import matplotlib.pyplot as plt
from torchvision import transforms as T
import random, numpy as np


def tensor_2_im(t, type = "rgb"):
    gray = T.Compose([T.Normalize(mean = [ 0.], std = [1/0.5]), T.Normalize(mean = [-0.5], std = [1])])
    rgb = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.2505, 1/0.2505, 1/0.2505]),
                         T.Normalize(mean = [ -0.2250, -0.2250, -0.2250 ], std = [ 1., 1., 1. ])])
    inp = gray if type == "gray" else rgb
    return(inp(t)*255).detach().squeeze().cpu().permute(1,2,0).numpy().astype(np.uint8) if type == "gray" else (inp(t)*255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def display(ds, rows, im_num):
    
    """
    Display a grid of randomly selected images from a dataset.

    Args:
        ds (CustomData): Custom dataset object containing images and labels.
        rows (int): Number of rows in the grid.
        im_num (int): Total number of images to display.
    """
    
    indices = np.random.randint(low=0, high=len(ds), size=im_num)

    plt.figure(figsize=(20, 10))
    for i, idx in enumerate(indices):
        im, gt = ds[idx]
        plt.subplot(rows, im_num // rows, i+1)
        plt.imshow(tensor_2_im(im))
        plt.title(f'GT > {ds.classes[gt]}')
        plt.axis('off')
        plt.savefig("D:/portfolio/Classification/Skin_lesion/Images/sample.png")
    plt.show()




