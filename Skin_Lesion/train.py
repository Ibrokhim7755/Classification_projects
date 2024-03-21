import torch, os, numpy as np
from tqdm import tqdm



# Function for setting up training parameters
def train_setup(model):
    """
    Set up the training environment including device, optimizer, criterion, and epochs.

    Args:
        model: PyTorch model to be trained.

    Returns:
        device (torch.device): Device for training (CUDA if available, otherwise CPU).
        model (torch.nn.Module): Model moved to the selected device.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion: Loss function for training.
        epochs: Number of epochs for training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cuda")
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 10
    return device, model, optimizer, criterion, epochs


# Function for training the model
def train(model, tr_dl, val_dl, loss_fn, epochs, opt, device, threshold=0.001):
    """
    Train the model and validate it on a validation set.

    Args:
        model: PyTorch model to be trained.
        tr_dl: Training data loader.
        val_dl: Validation data loader.
        loss_fn: Loss function for training.
        epochs: Number of epochs for training.
        opt: Optimizer for updating model parameters.
        device: Device for training (CUDA or CPU).
        threshold (float): Threshold for considering improvement in validation loss.

    Returns:
        dict: Dictionary containing training and validation accuracy and loss scores.
    """
    tr_acc_sc, tr_loss_cs, val_acc_sc, val_loss_cs = [], [], [], []
    best_loss = np.inf
    for epoch in range(epochs):
        print(f"{epoch + 1} - epoch is starting ....")
        tr_loss, tr_acc = 0, 0
        for idx, batch in enumerate(tqdm(tr_dl)):
            im, gt = batch
            im, gt = im.to(device), gt.to(device)

            # Convert input image tensor to the appropriate data type
            im = im.float()  # Convert to float if the input type is not already float

            pred = model(im)
            loss = loss_fn(pred, gt)
            tr_loss += loss.item()

            pred_class = torch.argmax(pred, dim=1)
            tr_acc += (pred_class == gt).sum().item()
            # Perform optimization steps
            opt.zero_grad()
            loss.backward()
            opt.step()

        tr_loss /= len(tr_dl)
        tr_acc /= len(tr_dl.dataset)
        tr_acc_sc.append(tr_acc)
        tr_loss_cs.append(tr_loss)

        model.eval()
        with torch.no_grad():
            val_loss, val_acc = 0, 0
            for idx, batch in enumerate(tqdm(val_dl)):
                im, gt = batch
                im, gt = im.to(device), gt.to(device)

                # Convert input image tensor to the appropriate data type
                im = im.float()  # Convert to float if the input type is not already float

                pred = model(im)
                loss = loss_fn(pred, gt)

                val_loss += loss.item()
                pred_class = torch.argmax(pred, dim=1)
                val_acc += (pred_class == gt).sum().item()

            val_acc /= len(val_dl.dataset)
            val_loss /= len(val_dl)
            val_acc_sc.append(val_acc)
            val_loss_cs.append(val_loss)

            print(f"{epoch + 1} - epoch Train process results:\n")
            print(f"{epoch + 1} - epoch Train Accuracy score       - > {tr_acc:.3f}")
            print(f"{epoch + 1} - epoch Train epoch loss score      - > {tr_loss:.3f}")
            print(f"{epoch + 1} - epoch Validation process results:\n")
            print(f"{epoch + 1} - epoch Validation Accuracy score  - > {val_acc:.3f}")
            print(f"{epoch + 1} - epoch Validation epoch loss score - > {val_loss:.3f}")

            if val_loss < (best_loss + threshold):
                best_loss = val_loss
                # os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), "D:/portfolio/Classification/Skin_lesion/Images/model_best_model.pth")

    return {"tr_acc_sc": tr_acc_sc, "tr_loss_cs": tr_loss_cs, "val_acc_sc": val_acc_sc, "val_loss_cs": val_loss_cs}
