import fire
from conf.config import Config
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from utils import convert_to_onnx, load_model

cs = ConfigStore.instance()
cs.store(name="config", group="first", node=Config)


def main(model_path, save_path):
    with initialize(version_base="1.3", config_path="conf"):
        cfg = compose(
            config_name="config",
            overrides=[f"model.onnx_parameters.onnx_path={save_path}"],
        )
    model = load_model(model_path=model_path, model_name="")
    convert_to_onnx(model, cfg.model.onnx_parameters)


if __name__ == "__main__":
    fire.Fire(main)
