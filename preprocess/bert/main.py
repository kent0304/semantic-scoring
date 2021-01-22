import torch 
from transformers import BertTokenizer, BertForPreTraining

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased')

text = "A train traveling through the * next to a dirt road."
candidate = ["countryside", "hell", "school", "zoo"]

# トークン分割
tokens = tokenizer.tokenize(text)
masked_index = tokens.index("*")

tokens[masked_index] = "[MASK]"
tokens = ["[CLS]"] + tokens + ["[SEP]"]
print(tokens)


# BERTで予測
ids = tokenizer.convert_tokens_to_ids(tokens)
ids = torch.tensor(ids).reshape(1,-1)  # バッチサイズ1の形に整形

with torch.no_grad():
    outputs = model(ids)
predictions = outputs.prediction_logits[0]

_, predicted_indexes = torch.topk(predictions[masked_index], k=30)

predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())
print(predicted_tokens)

# for i, v in enumerate(predicted_tokens):
#     if v in candidate:
#         print(i, v)
#         break