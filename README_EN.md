
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
model_path = "OctopusMind/longbert-8k-zh"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
embeddings = model.encode(['他愤怒地看看血，接着把一条手巾浸湿，抹去了鬓角上的血迹。',
                           '血已经快干了，弄脏了他的手心；他恶狠狠地望了望血迹，然后浸湿了一条毛巾，擦去了太阳穴上的血。'])
print(cos_sim(embeddings[0], embeddings[1]))
```

## Training data format

```json
[
   {
      "sentence1": "蜂巢取快递验证码摁错怎么办",
      "sentence2": "找寄件人改号码",
      "label": "0"
   },
   {
      "sentence1": "生产过后怎么还有一层肚子",
      "sentence2": "<br><img><br>凡是生过孩子的必会留下痕迹”,相信许多年轻妈咪对这句话感同身受。而产后肚子大就是女性生没生孩子的重要标志,今天我们讲了为什么女性生完孩子肚子还是很大的原因,也说了新妈妈们产后如何快速恢复肚子,让我们远离“妈妈肚”,生完孩子还是小姐姐。产后半年是恢复身材的关键时间,新妈妈时间这么紧,时间哪里找!!赶紧行动吧!!下一位辣妈就是你!!!zy(为什么产后肚子还很大?产后如何恢复肚子)",
      "label": "1"
   },
   {
      "sentence1": "大学怎么网上选宿舍",
      "sentence2": "本人大一新生 一枚<br>今天早上八点开始选宿舍,我手机上选到了心怡的宿舍和床位之后就洗碗去了。当时我填的时候宿舍里只有一个人已经选定了床位,我是第二个“入住”的。等我洗碗回来,发现我的QQ炸了<br>四个小姐姐轮流来消息发长文要我退出去,强行礼貌,信息轰炸,希望我去和别的专业的三个人拼宿舍,把位置让给她们 因为我的存在让她们不能住一个寝室了。<br>可是我事先并不知道她们已经商定好了宿舍。我也很为难啊<br>我不退的话,我就是恶人,破坏人家本来的计划。搞不好开学后还会被宿舍其他的小姐姐孤立。<br>我退出的话,我提前蹲点挤进去抢宿舍就是为了当个活菩萨?帮陌生人占个位置的?<br>不止女生宿舍,男生也是这个情况。提前一两天已经开始商量谁谁谁一个宿舍,结果抢宿舍被哪个不知情的小盆友占了位置,就开始在群里各种污言秽语的辱骂<br>男生好歹还在班群里说了谁谁谁一起一个宿舍。女生的话,事先完全没有在大群里讲到几个要好的已经有安排了。<br>难道真的就不存在什么先来后到吗?了因为你们人多,你们拉帮结派迅速,就要给你们提供优先权,你们就是强势群体吗?<br>真的很委屈啊",
      "label": "1"
   }]
```



### Experimental Comparison
