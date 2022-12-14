from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from datasets import load_from_disk
import torch
import sys

model_name = "google/pegasus-multi_news"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
print(device)


def generate_summary(document):
    result = []
    for text in document['document']:
        batch = tokenizer(''.join(text), truncation=True, padding="longest", return_tensors="pt").to(device)
        translated = model.generate(**batch)
        result.extend(tokenizer.batch_decode(translated, skip_special_tokens=True))
    document['pred'] = result
    return document


shard = load_from_disk("/scratch/kd1860/DSGA_1006_capstone/dataset/multi_news_test_clustered/shard_" + sys.argv[1])

print("start")
print("device:", device)
result = shard.map(generate_summary)
print("end")


result.save_to_disk("/scratch/kd1860/DSGA_1006_capstone/dataset/multi_news_test_pegasus/shard_"+ sys.argv[1])