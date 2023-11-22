from longbert_utils import LongBertPredictor

# load huggingface model
model_path = "OctopusMind/longbert-8k-zh"  # "./outputs/my-model"
encoder = LongBertPredictor(model_path)

# load local model
# model_path =  "./outputs/my-model"
# encoder = LongBertPredictor(model_name)

# use similarity method
print(encoder.similarity(['在香港哪里买手表好', '我喜欢你'], ['香港买手表哪里好', '我中意你']))

# use encode method
sentences = ['她是一个非常慷慨的女人，拥有自己的一大笔财产。', '她有很多钱，但她是个慷慨的女人。']
embeddings = encoder.encode(sentences)
print(f"{sentences}: {encoder.cos_sim(embeddings[0], embeddings[1])}")
