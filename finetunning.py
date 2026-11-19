import torch.nn as nn
from finetune_adapter import Network
import torch
import pandas as pd
import json
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import EarlyStoppingCallback
from transformers import TrainingArguments

a=Network()
model, tokenizer = a.method()

class InputAdapter(nn.Module):

    def __init__(self,hidden_size=4096,bottleneck=256):#bottle neck adapters

        super().__init__()

        self.layernorm=nn.LayerNorm(hidden_size,)
        self.down=nn.Linear(hidden_size,bottleneck) #4096 x 256(out of total we choosed this much)
        self.activation=nn.GELU()#non linearlity
        self.up=nn.Linear(bottleneck,hidden_size) #256 X 4096 (converted it back)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self,x):

        residual=x
        x=self.layernorm(x)
        x=self.down(x)
        x=self.activation(x)
        x=self.up(x)
        x=x+residual
        return residual + 0.1 * x

class EmbeddingWithAdapter(nn.Module):

    def __init__(self,embedding,adapter):

        super().__init__()
        self.embedding=embedding
        self.adapter=adapter

    def forward(self,input_ids):

        x=self.embedding(input_ids)
        x=self.adapter(x)
        return x

adapter=InputAdapter(hidden_size=model.config.hidden_size,bottleneck=256).to(model.device)
adapter = adapter.to(device=model.device,dtype=torch.bfloat16)
original_embedding = model.model.embed_tokens
model.model.embed_tokens = EmbeddingWithAdapter(original_embedding,adapter) #Replace the embedding inside Llama

dataset_path = "/hri/storage/user/lmangam/test/outputs/exper32/injected_dataset.csv"
df = pd.read_csv(dataset_path)
df = df.dropna()

print(f"Complete rows : {len(df)}")
train_df, test_df = train_test_split(df,test_size=0.2,random_state=42,shuffle=True)

print("Train :", len(train_df))
print("Test  :", len(test_df))

def row_to_text(row):

    row_json = json.dumps(row.to_dict(), indent=4)

    return {"text": row_json}

train_dataset = Dataset.from_list([row_to_text(row) for _, row in train_df.iterrows()])

test_dataset = Dataset.from_list([row_to_text(row) for _, row in test_df.iterrows()])

def tokenize_function(example):

    encoding = tokenizer(
        example["text"],
        truncation=True,)

    # causal LM
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

train_dataset = train_dataset.map(tokenize_function,remove_columns=["text"])

test_dataset = test_dataset.map(tokenize_function,remove_columns=["text"])
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False,)




training_args = TrainingArguments(

    output_dir="./adapter",
    overwrite_output_dir=True,
    learning_rate=5e-5,          # Slightly safer than 1e-4
    num_train_epochs=10,         # Maximum epochs

    per_device_train_batch_size=4,

    per_device_eval_batch_size=2,
    gradient_accumulation_steps = 2,

    save_strategy="epoch",

    eval_strategy="epoch",

    logging_strategy="steps",

    logging_steps=1,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    save_total_limit=1,          # Keep only the best/latest checkpoint

    bf16=True,

    report_to="none",

    remove_unused_columns=False,
    save_safetensors=True,
)
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=4,
            early_stopping_threshold=1e-3
        )
    ]
)

trainer.train()
import pandas as pd

history = pd.DataFrame(trainer.state.log_history)

print("\n================ Training History ================\n")

print(history)
torch.save(adapter.state_dict(),"/hri/storage/user/lmangam/test/adapter.pt")