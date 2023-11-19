
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
* The model has been uploaded to [Huggingface](https://huggingface.co/OctopusMind/LongBert)


### Usage
```python

from numpy.linalg import norm
from transformers import AutoModel
model_path = "OctopusMind/LongBert"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
embeddings = model.encode(['How is the weather today?', 'Do you think the weather is good now?'])
print(cos_sim(embeddings[0], embeddings[1]))
```

### Experimental Comparison
