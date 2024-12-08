import warnings
import os, urllib
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import itertools

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="hifigan")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    path = os.path.join(os.path.expanduser('~'), ".cache/wv_mos/wv_mos.ckpt")

    if (not os.path.exists(path)):
        print("Downloading the checkpoint for WV-MOS")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(
            "https://zenodo.org/record/6201162/files/wav2vec2.ckpt?download=1",
            path
        )
        print('Weights downloaded in: {} Size: {}'.format(path, os.path.getsize(path)))

    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup text_encoder
    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    generator = instantiate(config.generator).to(device)
    mpd = instantiate(config.mpd).to(device)
    msd = instantiate(config.msd).to(device)
    logger.info(generator)
    logger.info(mpd)
    logger.info(msd)

    # get function handles of loss and metrics
    loss_function_generator = instantiate(config.loss_function_generator).to(device)
    loss_function_discriminator = instantiate(config.loss_function_discriminator).to(device)
    loss_function_features = instantiate(config.loss_function_features).to(device)
    loss_function_mel_spec = instantiate(config.loss_function_mel_spec).to(device)

    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            # use text_encoder in metrics
            metrics[metric_type].append(
                instantiate(metric_config)
            )

    # build optimizer, learning rate scheduler
    trainable_params_discriminator = filter(lambda p: p.requires_grad, itertools.chain(mpd.parameters(), msd.parameters()))
    trainable_params_generator = filter(lambda p: p.requires_grad, generator.parameters())
    optimizer_discriminator = instantiate(
        config.optimizer_discriminator,
        params=trainable_params_discriminator,
    )
    optimizer_generator = instantiate(
        config.optimizer_generator,
        params=trainable_params_generator,
    )
    lr_scheduler_generator = instantiate(config.lr_scheduler_generator, optimizer=optimizer_generator)
    lr_scheduler_discriminator = instantiate(config.lr_scheduler_discriminator, optimizer=optimizer_discriminator)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        generator=generator,
        msd=msd,
        mpd=mpd,
        criterion_generator=loss_function_generator,
        criterion_discriminator=loss_function_discriminator,
        criterion_features=loss_function_features,
        criterion_mel_spec=loss_function_mel_spec,
        metrics=metrics,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
        lr_scheduler_discriminator=lr_scheduler_discriminator,
        lr_scheduler_generator=lr_scheduler_generator,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
