
<h1 align="center">
    Long Bert Chinese
    <br>
</h1>

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/OctopusMind/long-bert-chinese/blob/main/README.md">简体中文</a> 
    </p>
</h4>

<p >
<br>
</p>

**Long Bert**: A long text similarity model that supports a length of 8192 tokens.
Based on bert-base-chinese, it changes the original BERT positional encoding to ALiBi positional encoding, allowing BERT to support a sequence length of 8192.

### News
* Support `CoSENT` fine-tuning. see [train folder](./train)
* The model has been uploaded to [Huggingface](https://huggingface.co/OctopusMind/LongBert)


### Usage
```python

from numpy.linalg import norm
from transformers import AutoModel
model_path = "OctopusMind/longbert-8k-zh"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

sentences = ['我是问蚂蚁借呗为什么不能提前结清欠款', "为什么借呗不能选择提前还款"]
embeddings = model.encode(sentences)
cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
print(cos_sim(embeddings[0], embeddings[1]))
```

### CoSENT fine-tuning data format

```json
[
    {
        "sentence1": "一个男人在吹一支大笛子。",
        "sentence2": "一个人在吹长笛。",
        "label": 3
    },
    {
        "sentence1": "三个人在下棋。",
        "sentence2": "两个人在下棋。",
        "label": 2
    },
    {
        "sentence1": "一个女人在写作。",
        "sentence2": "一个女人在游泳。",
        "label": 0
    }
]
```



### Fine-tuning
```bash
cd train/
```
```bash
python cosent_finetune.py \
        --data_dir ../data/train_data.json \
        --output_dir ./outputs/my-model \
        --max_seq_length 1024 \
        --num_epochs 10 \
        --batch_size 64 \
        --learning_rate 2e-5
```

## Contributions
Feel free to make contributions to this module by submitting pull requests or raising issues in the repository.

## License
This project is licensed under the [Apache-2.0 License](./LICENSE).
