
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
* 模型已上传至 [Huggingface](https://huggingface.co/OctopusMind/LongBert)


### 使用
```python
from numpy.linalg import norm
from transformers import AutoModel
model_path = "OctopusMind/longbert-8k-zh"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
embeddings = model.encode(['今天天气怎么样？', '你觉得现在天气好吗'])
print(cos_sim(embeddings[0], embeddings[1]))

```

经过数据集[]() 微调，效果如下：

### 实验对比

