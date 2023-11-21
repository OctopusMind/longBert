import json
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel


# 自定义数据集类
class OctopusDataset(Dataset):
    def __init__(self, train_data_path, tokenizer):
        self.tokenizer = tokenizer
        with open(train_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.sentences1 = []
        self.sentences2 = []
        self.label = []
        for one in data:
            self.sentences1.append(one["sentence1"])
            self.sentences2.append(one["sentence2"])
            if int(one["label"]) == 0:
                self.label.append(0)
            else:
                self.label.append(1)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        sentence1 = self.sentences1[index]
        sentence2 = self.sentences2[index]
        label = self.label[index]
        tokenizer_kwargs = {}
        tokenizer_kwargs['padding'] = tokenizer_kwargs.get('padding', True)
        tokenizer_kwargs['max_length'] = tokenizer_kwargs.get('max_length', 8192)
        tokenizer_kwargs['truncation'] = tokenizer_kwargs.get('truncation', True)
        encoded_input = self.tokenizer(
            [sentence1, sentence2],
            return_tensors='pt',
            **tokenizer_kwargs,
        )
        return encoded_input["input_ids"], encoded_input["token_type_ids"], encoded_input["attention_mask"], label


class OctopusFinetuning:
    def __init__(self, args):
        self.args = args
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, use_auth_token=True)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)
        dataset = OctopusDataset(self.args.train_data_path, tokenizer=self.model.tokenizer)
        self.dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.loss_fn = nn.MSELoss()

    def collate_fn(self, batch):
        inputs = [item for items in batch for item in items[0]]
        token_type_ids = [item for items in batch for item in items[1]]
        attention_mask = [item for items in batch for item in items[2]]
        labels = [item[3] for item in batch]

        max_length = max(len(input_ids) for input_ids in inputs)
        padded_inputs = []
        padded_token_type_ids = []
        padded_attention_mask = []

        for input_ids, tt_ids, am in zip(inputs, token_type_ids, attention_mask):
            pad_length = max_length - len(input_ids)
            padded_inputs.append(input_ids.tolist() + [0] * pad_length)
            padded_token_type_ids.append(tt_ids.tolist() + [0] * pad_length)
            padded_attention_mask.append(am.tolist() + [0] * pad_length)

        return torch.tensor(padded_inputs), torch.tensor(padded_token_type_ids), torch.tensor(
            padded_attention_mask), torch.tensor(labels)

    def train(self):
        for _ in range(self.args.epoch):
            for batch_targets in self.dataloader:
                token_embs = self.model(input_ids=batch_targets[0].squeeze(0),
                                        token_type_ids=batch_targets[1].squeeze(0),
                                        attention_mask=batch_targets[2].squeeze(0))
                # 计算余弦相似度
                cos_sim = torch.cosine_similarity(token_embs[1][::2], token_embs[1][1::2], dim=1)
                loss = self.loss_fn(cos_sim, batch_targets[-1].float())
                print(loss)
                # 清除梯度
                self.optimizer.zero_grad()

                # 反向传播
                loss.backward()

                # 更新参数
                self.optimizer.step()
        torch.save(self.model.state_dict(), self.args.save_model_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        '--train_data_path', default='data/train_data.json')
    parser.add_argument(
        '--model_path', default='OctopusMind/Long-bert-8k-zh')
    parser.add_argument(
        '--save_model_path', default='pytorch_model.bin')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epoch', default=1)
    parser.add_argument('--lr', default=0.00001)
    args = parser.parse_args()
    octopus_finetuning = OctopusFinetuning(args)
    octopus_finetuning.train()
