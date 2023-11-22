
<h1 align="center">
    Long Bert Chinese
    <br>
</h1>

<h4 align="center">
    <p>
        <b>简体中文</b> |
        <a href="https://github.com/OctopusMind/long-bert-chinese/blob/main/README_EN.md">English</a> 
    </p>
</h4>

<p >
<br>
</p>

**Long Bert**: 长文本相似度模型，支持8192token长度。
基于bert-base-chinese，将原始BERT位置编码更改成ALiBi位置编码，使BERT可以支持8192的序列长度。

### News
* 支持`CoSENT`微调
* 模型已上传至 [Huggingface](https://huggingface.co/OctopusMind/LongBert)


### 使用
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

## 微调
### 数据格式

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

### CoSENT 微调

至`train/`路径下
```bash
cd train/
```
进行 CoSENT 微调
```bash
python cosent_finetune.py \
        --data_dir ../data/sts-b.json \
        --output_dir ./outputs/STS-B-model \
        --max_seq_length 1024 \
        --num_epochs 10 \
        --batch_size 64 \
        --learning_rate 2e-5
```



## 贡献
欢迎通过提交拉取请求或在仓库中提出问题来为此模块做出贡献。

## License
本项目遵循[Apache-2.0开源协议](./LICENSE)