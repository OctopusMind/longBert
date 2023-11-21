from numpy.linalg import norm
from transformers import AutoModel
model_path = "OctopusMind/longbert-8k-zh"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
embeddings = model.encode(['他愤怒地看看血，接着把一条手巾浸湿，抹去了鬓角上的血迹。',
                           '血已经快干了，弄脏了他的手心；他恶狠狠地望了望血迹，然后浸湿了一条毛巾，擦去了太阳穴上的血。'])
print(cos_sim(embeddings[0], embeddings[1]))
