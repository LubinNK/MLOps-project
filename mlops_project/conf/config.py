from dataclasses import dataclass
from typing import List


@dataclass
class Data:
    name: str
    path: str
    sourse: str
    train_data_file: str
    train_labels_file: str
    test_data_file: str
    test_labels_file: str


@dataclass
class ModelParameters:
    k: int


@dataclass
class OnnxParameters:
    export_to_onnx: bool
    onnx_path: str
    input_shape: List
    mlflow_onnx_export_path: str


@dataclass
class Model:
    name: str
    save_path: str
    save_name: str
    parameters: ModelParameters
    onnx_parameters: OnnxParameters


@dataclass
class Training:
    log_every_n_steps: int
    num_workers: int
    train_part: float
    batch_size: int
    epochs: int
    optimizer: str
    device: str


@dataclass
class Infer:
    model_name: str
    model_path: str
    infer_save_path: str
    infer_name: str
    batch_size: int


@dataclass
class MLflow:
    experiment_name: str
    tracking_uri: str


@dataclass
class Loggers:
    mlflow: MLflow


@dataclass
class Config:
    data: Data
    model: Model
    training: Training
    infer: Infer
    loggers: Loggers
