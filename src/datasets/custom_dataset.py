import torchaudio

from pathlib import Path

from src.datasets.base_dataset import BaseDataset


class CustomDatasetWav(BaseDataset):
    def __init__(self, wavs_dir, *args, **kwargs):
        data = []
        for wav_path in Path(wavs_dir).iterdir():
            entry = {}
            if wav_path.suffix in [".mp3", ".wav", ".flac"]:
                entry["audio_path"] = str(wav_path)
                entry["audio_len"] = self._calculate_length(wav_path)
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
    
    def _calculate_length(self, audio_path: Path) -> float:
        audio_info = torchaudio.info(str(audio_path))
        return audio_info.num_frames / audio_info.sample_rate
