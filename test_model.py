from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
sentence = "Hello from mani"
embedding = model.encode(sentence)
print("Embedding size:", len(embedding))
