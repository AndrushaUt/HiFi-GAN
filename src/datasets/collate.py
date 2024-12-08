import torch
from torch.nn.utils.rnn import pad_sequence

from dataclasses import dataclass

import torch
from torch import nn

import torchaudio

import librosa


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):

    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            center=False,
            pad=(config.n_fft - config.hop_length) // 2
        )
        self.mel_spectrogram.spectrogram.power = config.power

        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """

        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel


def collate_fn(dataset_items: list[dict]) -> dict[str, torch.Tensor | list]:
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result = {}
    spectrograms = []
    audios = []

    mel_spec = MelSpectrogram(MelSpectrogramConfig())
    need_pad = dataset_items[0]['need_pad']
    if need_pad == 'test':
        result["spectrogram"] = pad_sequence([item["spectrogram"].squeeze(0).permute(1, 0) for item in dataset_items], batch_first=True).permute(0, 2, 1)
        result['path'] = [item['path'] for item in dataset_items]

        return result
    if not need_pad:
        for item in dataset_items:
            spectrograms.append(mel_spec(item["audio"]).squeeze(1))
            audios.append(item["audio"])
        result["audio"] = torch.vstack([item["audio"] for item in dataset_items]).unsqueeze(1)
        result['spectrogram'] = torch.vstack(spectrograms)
    else:
        for item in dataset_items:
            spectrograms.append(mel_spec(item["audio"]).squeeze(0).permute(1, 0))
            audios.append(item["audio"].squeeze(0))
        result["audio"] = pad_sequence(audios, batch_first=True)
        result['spectrogram'] = pad_sequence(spectrograms, batch_first=True).permute(0, 2, 1)

    result["audio_path"] = [item["audio_path"] for item in dataset_items]


    return result
