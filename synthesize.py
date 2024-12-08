import warnings
import os, urllib
import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="synthesize")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    path = os.path.join(os.path.expanduser('~'), ".cache/wv_mos/wv_mos.ckpt")
    print(path)
    if (not os.path.exists(path)):
        print("Downloading the checkpoint for WV-MOS")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(
            "https://zenodo.org/record/6201162/files/wav2vec2.ckpt?download=1",
            path
        )
        print('Weights downloaded in: {} Size: {}'.format(path, os.path.getsize(path)))
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup text_encoder
    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    generator = instantiate(config.generator).to(device)
    print(generator)

    # get metrics
    metrics = {"inference": []}
    for metric_config in config.metrics.get("inference", []):
        # use text_encoder in metrics
        metrics["inference"].append(
            instantiate(metric_config)
        )

    # save_path for model predictions
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        generator=generator,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
