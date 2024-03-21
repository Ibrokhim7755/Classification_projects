from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np, random, matplotlib.pyplot as plt, torch
from torchvision import transforms as T
from tqdm import tqdm
import cv2


def tensor_2_im(t, type="rgb"):
    """
    Convert a PyTorch tensor to an image.

    Args:
    - t (torch.Tensor): Input tensor.
    - type (str): Type of the image. Options: "rgb" or "gray".

    Returns:
    - numpy.ndarray: Image in numpy array format.
    """
    gray = T.Compose([T.Normalize(mean=[0.], std=[1/0.5]), T.Normalize(mean=[-0.5], std=[1])])
    rgb = T.Compose([T.Normalize(mean=[0., 0., 0.], std=[1/0.2505, 1/0.2505, 1/0.2505]),
                     T.Normalize(mean=[-0.2250, -0.2250, -0.2250], std=[1., 1., 1.])])
    inp = gray if type == "gray" else rgb
    return (inp(t)*255).detach().squeeze().cpu().permute(1, 2, 0).numpy().astype(np.uint8) if type == "gray" else (inp(t)*255).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)


def inference(model, device, data, num_im, row, class_name=None, im_dim=None):
    """
    Perform inference using the provided model on the given data and visualize the results.

    Args:
    - model: PyTorch model for inference.
    - device (torch.device): Device to run the inference on.
    - data (torch.utils.data.Dataset): Dataset for inference.
    - num_im (int): Number of images to visualize.
    - row (int): Number of rows in the visualization grid.
    - class_name (list or None): List of class names for labeling the predictions. Default is None.
    - im_dim (int or None): Dimension for resizing the heatmap. Default is None.

    Returns:
    - None
    """
    #os.makedirs(save_prefix, exist_ok=True)
    Acc = 0
    preds, ims, gts = [], [], []
    for idx, batch in enumerate(tqdm(data)):
        im, gt = batch
        im, gt = im.to(device), gt.to(device)
        pred = model(im)
    
        pred_class = torch.argmax(pred, dim=1)
        Acc += (pred_class == gt).sum().item()
        ims.append(im); preds.append(pred_class.item()); gts.append(gt.item())
    print(f"Accuracy of the model on the test dataset -> {Acc/len(data): .3f}")
    plt.figure(figsize=(20, 10))
    index = [random.randint(0, len(ims)-1) for _ in range(num_im)]
    
    for i, idx in enumerate(index):
        im = ims[idx].squeeze(); gt = gts[idx]; pred = preds[idx]
  
        #GradCAM
        orginal_im = tensor_2_im(im) / 255
        cam = GradCAMPlusPlus(model=model, target_layers=[model.layer4[-1]])
        grayscale_cam = cam(input_tensor=im.unsqueeze(0))[0, :]
        heat_map = show_cam_on_image(img=orginal_im, mask=grayscale_cam, image_weight=0.1, use_rgb="jet")
        
        #start plot
        plt.subplot(row, num_im//row, i+1)
        plt.imshow(tensor_2_im(im), cmap="gray")
        plt.axis("off")
        plt.imshow(cv2.resize(heat_map, (im_dim, im_dim), interpolation=cv2.INTER_LINEAR), alpha=0.3, cmap='jet')
        plt.axis("off")
        plt.savefig("D:/portfolio/Classification/Skin_lesion/Images/inference/inference.png")
        color = ("green" if {class_name[int(gt)]} == {class_name[int(pred_class)]} else 'red')
        if class_name:
            plt.title(f"GT -> {class_name[gt]}; Pred- > {class_name[pred_class]}", color=color)
        else:
            plt.title(f"GT -> {gt}; PRED -> {pred}")
