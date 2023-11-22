from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class CosentTrainDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data_files, max_len: int = 128):
        self.tokenizer = tokenizer
        dataset = load_dataset('json', data_files=data_files, split="train")
        print(f"{dataset=}")
        self.data = self.convert_to_rank(dataset)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text2id(self, text: str):
        return self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text2id(line[0]), line[1]

    def convert_to_rank(self, dataset):
        """
        Flatten the dataset to a list of tuples
        """
        data = []
        for line in dataset:
            data.append((line['sentence1'], line['label']))
            data.append((line['sentence2'], line['label']))
        return data
