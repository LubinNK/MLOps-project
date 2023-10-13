import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from model import CNN_new
from utils import get_loader, load_mnist


def train_epoch(model, optimizer, train_loader, criterion, device):
    """
    for each batch
    performs forward and backward pass and parameters update

    Input:
    model: instance of model (example defined above)
    optimizer: instance of optimizer (defined above)
    train_loader: instance of DataLoader

    Returns:
    nothing

    Do not forget to set net to train mode!
    """
    # your code here
    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        x_pred = model(x)

        # train_step
        loss = criterion(x_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_loss_acc(loader, model, criterion, device):
    """
    Evaluates loss and accuracy on the whole dataset

    Input:
    loader:  instance of DataLoader
    model: instance of model (examle defined above)

    Returns:
    (loss, accuracy)

    Do not forget to set net to eval mode!
    """
    # your code here
    model.eval()
    loss, accur = 0.0, 0.0
    # так как батчи могут быть разных размеров нужно считать и количество батчей, и суммарное число элементов
    batches_count = 0
    full_count = 0
    with torch.no_grad():
        for x, y in loader:
            full_count += len(y)
            batches_count += 1
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss += criterion(y_pred, y).item()
            accur += torch.sum(
                torch.argmax(y_pred, dim=-1) == y
            )  # it will be normed before return
    accur = float(accur) / full_count
    loss = loss / batches_count  # уcреднили по батчам
    return loss, accur


def train(
    model, opt, train_loader, test_loader, criterion, n_epochs, device, verbose=True
):
    """
    Performs training of the model and prints progress

    Input:
    model: instance of model (example defined above)
    opt: instance of optimizer
    train_loader: instance of DataLoader
    test_loader: instance of DataLoader (for evaluation)
    n_epochs: int

    Returns:
    4 lists: train_log, train_acc_log, val_log, val_acc_log
    with corresponding metrics per epoch
    """
    train_log, train_acc_log = [], []
    val_log, val_acc_log = [], []

    for epoch in range(n_epochs):
        train_epoch(model, opt, train_loader, criterion, device)
        train_loss, train_acc = evaluate_loss_acc(
            train_loader, model, criterion, device
        )
        val_loss, val_acc = evaluate_loss_acc(test_loader, model, criterion, device)

        train_log.append(train_loss)
        train_acc_log.append(train_acc)

        val_log.append(val_loss)
        val_acc_log.append(val_acc)

        if verbose:
            print(
                (
                    "Epoch [%d/%d], Loss (train/val): %.4f/%.4f,"
                    + " Acc (train/val): %.4f/%.4f"
                )
                % (epoch + 1, n_epochs, train_loss, val_loss, train_acc, val_acc)
            )

    return train_log, train_acc_log, val_log, val_acc_log


def save_all(model, model_parameters, save_name):
    model_dict = model.state_dict()
    tmp_save = [model_dict, model_parameters]
    torch.save(tmp_save, save_name)


def main():
    n_epochs = 3
    save_name = "data/best_model.xyz"
    if len(sys.argv) > 1:
        n_epochs = int(sys.argv[1])
        if len(sys.argv) > 2:
            save_name = sys.argv[2]
    # datasets load
    X_train, y_train = load_mnist(train=True)
    # permute train
    idxs = np.random.permutation(np.arange(X_train.shape[0]))
    X_train, y_train = X_train[idxs], y_train[idxs]

    # for final model:
    # train_loader_full = get_loader(X_train, y_train)
    # for validation purposes:
    train_loader = get_loader(X_train[:25000], y_train[:25000])
    val_loader = get_loader(X_train[25000:30000], y_train[25000:30000])

    # define model
    model_parameters = {"k": 4}
    model = CNN_new(**model_parameters)
    opt = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    train(
        model, opt, train_loader, val_loader, criterion, n_epochs, device, verbose=True
    )

    save_all(model, model_parameters, save_name)


if __name__ == "__main__":
    main()
