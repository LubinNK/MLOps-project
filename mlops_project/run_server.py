"""""
import hydra
import mlflow
import onnx
from conf.config import Config
from hydra.core.config_store import ConfigStore


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Config):
    onnx_model = onnx.load("./models/best_model.onnx")

    with mlflow.start_run():
        signature = infer_signature()

    mlflow.onnx.load_model("./models/best_model.onnx")
    onnx_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)


predictions = onnx_pyfunc.predict(X.numpy())
print(predictions)
""" ""
