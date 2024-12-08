from pathlib import Path

from src.datasets.base_dataset import BaseDataset

from tqdm import tqdm


class CustomDatasetText(BaseDataset):
    def __init__(self, text_dir, part, text=None, *args, **kwargs):
        data = []
        if not text:
            for text_path in tqdm(Path(text_dir).iterdir(), desc="Reading dataset"):
                entry = {}
                if text_path.suffix in [".txt"]:
                    with open(text_path, 'r') as file:
                        entry["text"] = file.read().strip()
                        entry['text_path'] = text_path
                if len(entry) > 0:
                    data.append(entry)
        else:
            data.append({"text": text.strip(), 'text_path': Path('myfile.txt')})
        super().__init__(data, part=part, *args, **kwargs)
