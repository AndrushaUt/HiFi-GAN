from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch    
from collections import OrderedDict
from torch import nn
import os
from src.metrics.base_metric import BaseMetric
from torch import Tensor


import torchaudio

def extract_prefix(prefix, weights):
    result = OrderedDict()
    for key in weights:
        if key.find(prefix) == 0:
            result[key[len(prefix):]] = weights[key]
    return result     


class Wav2Vec2ConvEncoder:

    def __init__(self, device="cuda"):
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").feature_extractor
        self.encoder.eval()
        self.encoder = self.encoder.to(self.device)
        self.preprocessor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.preprocessor._sample_rate = 16000

    def __call__(self, x):
        # x - [bs, 1, time]
        x = x[:, 0]
        input_values = (x - x.mean(-1)[:, None]) / (x.std(-1)[:, None] + 1e-6)
        hidden_states = self.encoder(input_values.to(self.device))
        return hidden_states
    
class Wav2Vec2FullEncoder:

    def __init__(self, device="cuda"):
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.encoder.eval()
        self.encoder = self.encoder.to(self.device)
        self.preprocessor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.preprocessor._sample_rate = 16000

    def __call__(self, x):
        x = x[:, 0]
        input_values = (x - x.mean(-1)[:, None]) / (x.std(-1)[:, None] + 1e-6)
        hidden_states = self.encoder(input_values.to(self.device)).last_hidden_state
        return hidden_states.transpose(-2, -1)
    
    
class Wav2Vec2MOS(nn.Module):
    def __init__(self, freeze=True, cuda=True):
        super().__init__()
        path = os.path.join(os.path.expanduser('~'), ".cache/wv_mos/wv_mos.ckpt")

        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.freeze = freeze
        
        self.dense = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
        if self.freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.load_state_dict(extract_prefix('model.', torch.load(path, map_location=self.device)['state_dict']))
        self.eval()
        self.cuda_flag = cuda
        if cuda:
            self.cuda()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        
    def forward(self, x):
        x = self.encoder(x)['last_hidden_state']
        x = self.dense(x)
        x = x.mean(dim=[1,2], keepdims=True)
        return x
        
    def calculate_one(self, audio):
        resampled = torchaudio.functional.resample(audio.squeeze(0), 22050, 16000)
        x = self.processor(resampled, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        with torch.no_grad():
            if self.cuda_flag:
                x = x.cuda()
            res = self.forward(x).mean()
        return res.cpu().item()

class MosMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mos_meter = Wav2Vec2MOS(cuda=torch.cuda.is_available())
        self.values = []

    def __call__(
        self, wavs: Tensor, **kwargs
    ):
        for wav in wavs:
            mos = self.mos_meter.calculate_one(wav)
            self.values.append(mos)

        return mos
