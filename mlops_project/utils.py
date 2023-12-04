import os

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import torch
from conf.config import Model, OnnxParameters
from mlflow.models import infer_signature
from model import CNN_new


def load_model(model_path, model_name):
    filename = model_path + model_name
    assert os.path.isfile(filename), "file do not exist"
    [model_state_dict, model_parameters] = torch.load(filename)
    conf = Model(**model_parameters)
    model = CNN_new(conf)
    model.load_state_dict(model_state_dict)
    return model


def save_all(model, model_parameters, save_path, save_name):
    model_dict = model.state_dict()
    tmp_save = [model_dict, model_parameters]
    torch.save(tmp_save, save_path + save_name)


def convert_to_onnx(model, conf: OnnxParameters):
    model.eval()
    input_tensor = torch.randn(1, *conf.input_shape)
    torch.onnx.export(
        model,
        input_tensor,
        conf.onnx_path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["IMAGES"],
        output_names=["CLASS_PROBS"],
        dynamic_axes={"IMAGES": {0: "BATCH_SIZE"}, "CLASS_PROBS": {0: "BATCH_SIZE"}},
    )

    original_emb = model(input_tensor).detach().numpy()
    ort_input = {
        "IMAGES": input_tensor.numpy(),
    }

    ort_session = ort.InferenceSession(conf.onnx_path)
    onnx_embedding = ort_session.run(None, ort_input)[0]

    assert np.allclose(
        original_emb, onnx_embedding, atol=1e-5
    ), "something wrond with onnx model"

    onnx_model = onnx.load(conf.onnx_path)
    with mlflow.start_run():
        signature = infer_signature(input_tensor.numpy(), original_emb)
        mlflow.onnx.save_model(
            onnx_model, conf.mlflow_onnx_export_path, signature=signature
        )
