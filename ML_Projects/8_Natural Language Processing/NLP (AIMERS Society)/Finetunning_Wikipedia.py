import wikipedia #to train wikipedia pages
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch

#load Wikipedia articles
topics=['Artficial intelligence', 'Deep learning', 'Natural language processing']
texts=[]
for topic in topics:
    article=wikipedia.page(topic)
    texts.append(article.content)

#load GPT-2 tokenizer and model
model_name= 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenize texts and prepare for fine-tuning
inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Fine-tune the model
optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()
for epoch in range(3):
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch[0].to(device=device)
        attention_mask = batch[1].to(device=device)
        labels = input_ids.clone().detach()
        labels[labels == tokenizer.pad_token_id] = -100
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

# Generate some text using the fine-tuned model
prompt = "Artificial intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device=device)
output = model.generate(input_ids, max_length=100, do_sample=True)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)