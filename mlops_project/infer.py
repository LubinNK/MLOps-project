import sys

import numpy as np
import pandas as pd
import torch
from model import CNN_new
from utils import get_loader, load_mnist


def load_model(filename):
    [model_state_dict, model_parameters] = torch.load(filename)
    model = CNN_new(**model_parameters)
    model.load_state_dict(model_state_dict)
    return model


def validate(model, test_loader, device):
    model.eval()
    accur = 0.0
    batches_count = 0
    counter = 0
    model_answer = []
    with torch.no_grad():
        for x, y in test_loader:
            counter += len(y)
            batches_count += 1
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            model_answer.append((np.argmax(y_pred.cpu().numpy(), axis=-1)))
            accur += torch.sum(torch.argmax(y_pred, dim=-1) == y)
    accur = float(accur) / counter
    model_answer = np.hstack(model_answer)
    return model_answer, accur


def main():
    model_name = "data/best_model.xyz"
    save_name = "data/test_results.csv"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        if len(sys.argv) > 2:
            save_name = sys.argv[2]
    X_test, y_test = load_mnist(train=False)
    test_loader = get_loader(X_test, y_test)

    model = load_model(model_name)
    device = torch.device("cpu")
    model = model.to(device)
    model_answer, accuracy = validate(model, test_loader, device)
    data = pd.DataFrame(model_answer)
    data.to_csv(save_name, sep="\t", encoding="utf-8")
    print(f"Test Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
