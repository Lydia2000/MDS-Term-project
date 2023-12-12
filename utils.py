import pickle        
import matplotlib.pyplot as plt
import numpy as np

def load(dir):

    with open(dir, "rb") as f:  # Python 3: open(..., "rb")
        var = pickle.load(f)
        print(f"Load model from: {dir}")
    return var

def save(var, dir, verbose=False):

    with open(dir, "wb") as f:  # Python 3: open(..., "wb")
        pickle.dump(var, f)
        print(f"Model saved to: {dir}")

def plot_loss(training_losses, validation_losses):

    fig, ax = plt.subplots(figsize=(12, 8))
    line1, = ax.plot(training_losses, label='train_loss')
    line2, = ax.plot(validation_losses, label='val_loss')

    ax.legend()
    ax.grid(True)
    ax.set_xlabel('epochs')
    ax.set_ylabel('MSE')
    ax.set_xlim(0, len(training_losses)) 

    plt.savefig('loss.png')
    plt.show()

def plot_prediction(y_pred, y_true):

    # Plot predicted values and true value
    fig, ax = plt.subplots(figsize = (14,8))
    ax.plot(np.arange(y_pred.shape[0]), y_pred.cpu().numpy(), label = 'predictions', c = 'salmon') # y_pred.shape[0] -> batch_size
    ax.plot(np.arange(y_true.shape[0]), y_true.cpu().numpy(), label = 'true values', c = 'lightseagreen')

    # set labels and grid
    ax.set_xlabel('Test Engine Units', fontsize = 16)
    ax.set_ylabel('RUL', fontsize = 16)
    ax.grid(True)
    ax.legend()
    
    plt.savefig('prediction.png')
    plt.show()
    print('Prediction image saved to: prediction.png')