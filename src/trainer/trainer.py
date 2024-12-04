from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
        generated_audio = self.generator(batch['spectrogram'])
        batch['generated_audio'] = generated_audio

        generated_audio_spec = self.mec_spec(generated_audio)

        mpd_real_outputs, _ = self.mpd(batch['audio'])
        mpd_fake_outputs, _ = self.mpd(generated_audio.detach())
        msd_real_outputs, _ = self.msd(batch['audio'])
        msd_fake_outputs, _ = self.msd(generated_audio.detach())

        d_loss_mpd = self.criterion_discriminator(mpd_real_outputs, mpd_fake_outputs)
        d_loss_msd = self.criterion_discriminator(msd_real_outputs, msd_fake_outputs)

        batch['discriminator_loss'] = d_loss_mpd + d_loss_msd

        if self.is_train:
            self.optimizer_discriminator.zero_grad()
            batch['discriminator_loss'].backward()
            self._clip_grad_norm_disc()
            self.optimizer_discriminator.step()

        _, msd_real_fmaps = self.msd(batch['audio'])
        _, mpd_real_fmaps = self.mpd(batch['audio'])
        mpd_fake_outputs, mpd_fake_fmaps = self.mpd(generated_audio)
        msd_fake_outputs, msd_fake_fmaps = self.msd(generated_audio)
        g_loss_adv_mpd = self.criterion_generator(mpd_fake_outputs)
        g_loss_adv_msd = self.criterion_generator(msd_fake_outputs)
        batch['generator_loss'] = g_loss_adv_mpd + g_loss_adv_msd

        fm_loss_msd = self.criterion_features(msd_real_fmaps, msd_fake_fmaps)
        fm_loss_mpd = self.criterion_features(mpd_real_fmaps, mpd_fake_fmaps)
        batch['features_loss'] = fm_loss_mpd + fm_loss_msd
        batch['mel_spec_loss'] = self.criterion_mel_spec(batch['spectrogram'], generated_audio_spec)

        batch['loss'] = batch['generator_loss'] + 2 * batch['features_loss'] + 45 * batch['mel_spec_loss']

        if self.is_train:
            self.optimizer_generator.zero_grad()
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm_gen()
            self.optimizer_generator.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
            self.log_audio(**batch)
        # else:
        #     # Log Stuff
        #     self.log_spectrogram(**batch)
        #     self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)
    
    def log_audio(self, audio, generated_audio, **batch):
        self.writer.add_audio("audio", audio[0], 22050)
        self.writer.add_audio("generated_audio", generated_audio[0], 22050)

    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    ):
        # TODO add beam search
        # Note: by improving text encoder and metrics design
        # this logging can also be improved significantly

        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))

        rows = {}
        for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
            target = self.text_encoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
