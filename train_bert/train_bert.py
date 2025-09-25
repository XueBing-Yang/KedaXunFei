#!/usr/bin/env python
# finetune_bert.py


import torch
import torch_npu
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel

# ——— 配置区 ———
BERT_PATH    = '/cog15/gywang22/code/train_bert/AI-ModelScope/bert-base-cased'    # 本地模型目录，包含 config.json/pytorch_model.bin/vocab.txt
DATA_PATH    = '/cog15/gywang22/code/train_bert/train_bert_4473.json'         # JSONL 数据，每行 {"input": "...", "target": "..."}
SAVE_PATH    = '/cog15/gywang22/code/train_bert/output_v4'                # 模型输出目录
MAX_LEN      = 512                                # 序列截断/填充长度
BATCH_SIZE   = 32
LR           = 2e-4
EPOCHS       = 5
TEST_SIZE    = 0.2                                # 训练/测试集比例
RANDOM_SEED  = 42

# ——— 固定随机种子 ———
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(RANDOM_SEED)

# ——— 读入并划分数据 ———
df = pd.read_json(DATA_PATH, lines=True)

# 自动把 target 两类映射成 0/1
label_list = df['target'].unique().tolist()
label2id = {lab: i for i, lab in enumerate(label_list)}
df['label'] = df['target'].map(label2id)

train_df, test_df = train_test_split(
    df[['input','label']],
    test_size=TEST_SIZE,
    stratify=df['label'],
    random_state=RANDOM_SEED
)

# ——— 分词器 & Dataset ———
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

class MyDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.encodings = [
            tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=max_len,
                return_tensors='pt'
            )
            for text in df['input'].tolist()
        ]
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.encodings[idx]
        # squeeze 掉 batch 维度
        input_ids     = enc['input_ids'].squeeze(0)
        attention_mask= enc['attention_mask'].squeeze(0)
        # 有的 tokenizer 可能不返回 token_type_ids
        token_type_ids= enc.get('token_type_ids', torch.zeros_like(input_ids))
        if isinstance(token_type_ids, torch.Tensor):
            token_type_ids = token_type_ids.squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, token_type_ids, label

train_dataset = MyDataset(train_df, tokenizer, MAX_LEN)
test_dataset  = MyDataset(test_df,  tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# ——— 模型定义 ———
class BertClassifier(nn.Module):
    def __init__(self, bert_path, num_labels=2):
        super().__init__()
        self.bert    = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(0.1)
        self.linear  = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        pooled = out.pooler_output                    # [CLS] 向量
        x = self.dropout(pooled)
        x = self.linear(x)
        # x = self.sigmoid(x)                              # 可以去掉 ReLU 直接输出 logits
        return x

if torch.npu.is_available():
    device = torch.device("npu")
    print(f"Using NPU: {torch.npu.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("NPU not available, using CPU instead")
model  = BertClassifier(BERT_PATH, num_labels=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

# ——— 训练 & 测试循环 ———
best_acc = 0.0
os.makedirs(SAVE_PATH, exist_ok=True)

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    total_corr = 0
    for input_ids, attn_mask, tt_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
        input_ids     = input_ids.to(device)
        attn_mask     = attn_mask.to(device)
        tt_ids        = tt_ids.to(device)
        labels        = labels.to(device)

        logits = model(input_ids, attn_mask, tt_ids)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_corr += (preds == labels).sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc  = total_corr / len(train_dataset)

    # 验证
    model.eval()
    total_corr = 0
    with torch.no_grad():
        for input_ids, attn_mask, tt_ids, labels in tqdm(test_loader, desc=f"Epoch {epoch} [test]"):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            tt_ids    = tt_ids.to(device)
            labels    = labels.to(device)

            logits = model(input_ids, attn_mask, tt_ids)
            preds  = logits.argmax(dim=1)
            total_corr += (preds == labels).sum().item()

    test_acc = total_corr / len(test_dataset)
    print(f"Epoch {epoch} → train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")

    # 保存最优模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, "best_model.pt"))

# 保存最后一次训练权重
torch.save(model.state_dict(), os.path.join(SAVE_PATH, "last_model.pt"))