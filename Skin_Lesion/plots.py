import matplotlib.pyplot as plt



def plot_results(train_losses, val_losses, train_accs, val_accs):
    """
    Plot the training and validation loss curves, as well as the training and validation accuracy curves.

    Args:
    - train_losses (list): List of training losses for each epoch.
    - val_losses (list): List of validation losses for each epoch.
    - train_accs (list): List of training accuracies for each epoch.
    - val_accs (list): List of validation accuracies for each epoch.
    """
    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training accuracy')
    plt.plot(val_accs, label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig("D:/portfolio/Classification/Skin_lesion/Plots/learning_curves.png")
    
    plt.tight_layout()
    plt.show()
