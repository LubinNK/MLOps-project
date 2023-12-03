import os

import hydra
import numpy as np
import pandas as pd
import torch
from conf.config import Config, Model
from hydra.core.config_store import ConfigStore
from model import CNN_new

from data import get_loader, load_mnist


def load_model(model_path, model_name):
    filename = model_path + model_name
    assert os.path.isfile(filename), "file do not exist"
    [model_state_dict, model_parameters] = torch.load(filename)
    conf = Model(**model_parameters)
    model = CNN_new(conf)
    model.load_state_dict(model_state_dict)
    return model


def validate(model, test_loader, device):
    model.eval()
    accur = 0.0
    batches_count = 0
    full_count = 0
    model_answer = []
    with torch.no_grad():
        for x, y in test_loader:
            full_count += len(y)
            batches_count += 1
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            model_answer.append((np.argmax(y_pred.cpu().numpy(), axis=-1)))
            accur += torch.sum(torch.argmax(y_pred, dim=-1) == y)
    accur = float(accur) / full_count
    model_answer = np.hstack(model_answer)
    return model_answer, accur


cs = ConfigStore.instance()
cs.store(name="infer_config", node=Config)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Config):
    print("prepare data for test")
    X_test, y_test = load_mnist(cfg.data, train=False)
    print(X_test.shape)
    test_loader = get_loader(X=X_test, y=y_test, batch_size=cfg.infer.batch_size)

    model = load_model(model_name=cfg.infer.model_name, model_path=cfg.infer.model_path)

    device = torch.device("cpu")
    model = model.to(device)

    print("make computations")
    model_answer, accuracy = validate(model, test_loader, device)
    data = pd.DataFrame(model_answer)

    save_path = cfg.infer.infer_save_path + cfg.infer.infer_name
    data.to_csv(save_path, sep="\t", encoding="utf-8")
    print(f"testing accuracy: {accuracy}")
    print(f"save exit at {save_path}")


if __name__ == "__main__":
    main()
