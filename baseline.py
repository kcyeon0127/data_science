# =========================
# CTR: Tabular + Seq(LSTM)
# =========================
# - tqdm 추가 버전
# =========================

import os
import gc
import math
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ======================
# Config & Reproducible
# ======================
CFG = {
    "BATCH_SIZE": 4096,
    "EPOCHS": 10,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 1e-6,
    "SEED": 42,
    "VAL_SIZE": 0.2,
    "LSTM_HIDDEN": 64,
    "MLP_HIDDEN": [256, 128],
    "DROPOUT": 0.2,
    "GRAD_CLIP": 5.0,
    "USE_AMP": True,
    "EARLY_STOP_PATIENCE": 3,
    "USE_SCHEDULER": False,
    "SEQ_MAXLEN": 1024,
    "DO_DOWNSAMPLE": True,
    "NEG_POS_RATIO": 2,
}

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG["SEED"])

# ======================
# Data Loading
# ======================
TRAIN_PATH = "./train.parquet"
TEST_PATH = "./test.parquet"
SUB_PATH = "./sample_submission.csv"

print("Loading data...")
all_train = pd.read_parquet(TRAIN_PATH, engine="pyarrow")
test = pd.read_parquet(TEST_PATH, engine="pyarrow")
print("Train shape:", all_train.shape)
print("Test shape:", test.shape)

# Down-sampling
if CFG["DO_DOWNSAMPLE"]:
    clicked_1 = all_train[all_train["clicked"] == 1]
    clicked_0_full = all_train[all_train["clicked"] == 0]
    n_neg = min(len(clicked_0_full), len(clicked_1) * CFG["NEG_POS_RATIO"])
    clicked_0 = clicked_0_full.sample(n=n_neg, random_state=CFG["SEED"])
    train = pd.concat([clicked_1, clicked_0], axis=0).sample(frac=1, random_state=CFG["SEED"]).reset_index(drop=True)
else:
    train = all_train.sample(frac=1, random_state=CFG["SEED"]).reset_index(drop=True)

print("Train shape (used):", train.shape)
print("Class dist:", dict(train["clicked"].value_counts().sort_index()))

# ======================
# Columns
# ======================
target_col = "clicked"
seq_col = "seq"
ALL_EXCLUDE = {target_col, seq_col, "ID"}
feature_cols = [c for c in train.columns if c not in ALL_EXCLUDE and c in test.columns]

# ======================
# Dataset
# ======================
def robust_parse_seq(s, maxlen: int) -> np.ndarray:
    if pd.isna(s):
        return np.array([0.0], dtype=np.float32)
    s = str(s).replace(",", " ")
    toks = [t for t in s.split() if t]
    out = []
    for t in toks:
        try: out.append(float(t))
        except: pass
    if len(out) == 0: out = [0.0]
    if len(out) > maxlen: out = out[-maxlen:]
    return np.array(out, dtype=np.float32)

class ClickDataset(Dataset):
    def __init__(self, df, feature_cols, seq_col, target_col=None, has_target=True, seq_maxlen=1024):
        self.X = df[feature_cols].astype(float).fillna(0).values
        self.seq_strings = df[seq_col].astype(str).values
        self.has_target = has_target
        self.seq_maxlen = seq_maxlen
        if has_target: self.y = df[target_col].astype(np.float32).values
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        seq = torch.from_numpy(robust_parse_seq(self.seq_strings[idx], self.seq_maxlen))
        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return x, seq, y
        return x, seq

def collate_fn_train(batch):
    xs, seqs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long).clamp(min=1)
    return xs, seqs_padded, lens, ys

def collate_fn_infer(batch):
    xs, seqs = zip(*batch)
    xs = torch.stack(xs)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long).clamp(min=1)
    return xs, seqs_padded, lens

# ======================
# Model
# ======================
class TabularSeqModel(nn.Module):
    def __init__(self, d_features, lstm_hidden=64, hidden_units=[256,128], dropout=0.2):
        super().__init__()
        self.bn_x = nn.BatchNorm1d(d_features)
        self.lstm = nn.LSTM(1, lstm_hidden, batch_first=True)
        in_dim = d_features + lstm_hidden
        layers=[]
        for h in hidden_units:
            layers += [nn.Linear(in_dim,h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim=h
        layers += [nn.Linear(in_dim,1)]
        self.mlp=nn.Sequential(*layers)
    def forward(self, x_feats, x_seq, seq_lengths):
        x = self.bn_x(x_feats)
        x_seq = x_seq.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(x_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _,(h_n,_) = self.lstm(packed)
        h=h_n[-1]
        z=torch.cat([x,h],dim=1)
        return self.mlp(z).squeeze(1)

# ======================
# Train Utils
# ======================
def compute_pos_weight(y_np: np.ndarray) -> float:
    pos=(y_np==1).sum(); neg=(y_np==0).sum()
    return max(1.0, neg/max(1,pos))

@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    losses,probs,labels=[],[],[]
    for xs,seqs,lens,ys in tqdm(loader, desc="Valid", leave=False):
        xs,seqs,lens,ys=xs.to(device),seqs.to(device),lens.to(device),ys.to(device)
        logits=model(xs,seqs,lens)
        loss=loss_fn(logits,ys)
        losses.append(loss.item()*ys.size(0))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(ys.cpu().numpy())
    if not probs: return math.nan, math.nan, math.nan
    probs=np.concatenate(probs); labels=np.concatenate(labels)
    avg_loss=float(np.sum(losses)/len(labels))
    auc=roc_auc_score(labels,probs) if len(np.unique(labels))>1 else math.nan
    ll=log_loss(labels,probs,eps=1e-7)
    return avg_loss,auc,ll

def train_one_epoch(model, loader, optimizer, device, loss_fn, scaler=None, grad_clip=5.0):
    model.train(); running=0.0;n=0
    for xs,seqs,lens,ys in tqdm(loader, desc="Train", leave=False):
        xs,seqs,lens,ys=xs.to(device),seqs.to(device),lens.to(device),ys.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.cuda.amp.autocast():
                logits=model(xs,seqs,lens)
                loss=loss_fn(logits,ys)
            scaler.scale(loss).backward()
            if grad_clip: 
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            logits=model(xs,seqs,lens); loss=loss_fn(logits,ys)
            loss.backward()
            if grad_clip: nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
            optimizer.step()
        running+=loss.item()*ys.size(0); n+=ys.size(0)
    return running/max(1,n)

# ======================
# Training
# ======================
tr_df,va_df=train_test_split(train,test_size=CFG["VAL_SIZE"],random_state=CFG["SEED"],shuffle=True,stratify=train[target_col])

train_dataset=ClickDataset(tr_df,feature_cols,seq_col,target_col,True,CFG["SEQ_MAXLEN"])
val_dataset=ClickDataset(va_df,feature_cols,seq_col,target_col,True,CFG["SEQ_MAXLEN"])

train_loader=DataLoader(train_dataset,batch_size=CFG["BATCH_SIZE"],shuffle=True,collate_fn=collate_fn_train)
val_loader=DataLoader(val_dataset,batch_size=CFG["BATCH_SIZE"],shuffle=False,collate_fn=collate_fn_train)

model=TabularSeqModel(len(feature_cols),CFG["LSTM_HIDDEN"],CFG["MLP_HIDDEN"],CFG["DROPOUT"]).to(device)
pos_weight_t=torch.tensor(compute_pos_weight(tr_df[target_col].values.astype(int)),device=device)
loss_fn=lambda logits,y: nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=pos_weight_t)
optimizer=optim.AdamW(model.parameters(),lr=CFG["LEARNING_RATE"],weight_decay=CFG["WEIGHT_DECAY"])
scaler=torch.cuda.amp.GradScaler() if (CFG["USE_AMP"] and device=="cuda") else None

best_auc=-1; no_improve=0; best_path="./best_tabular_seq.pt"

for epoch in range(1,CFG["EPOCHS"]+1):
    tr_loss=train_one_epoch(model,train_loader,optimizer,device,loss_fn,scaler,CFG["GRAD_CLIP"])
    va_loss,va_auc,va_ll=evaluate(model,val_loader,device,loss_fn)
    print(f"[Epoch {epoch}] train={tr_loss:.5f} val={va_loss:.5f} AUC={va_auc:.5f} logloss={va_ll:.5f}")
    if va_auc>best_auc:
        best_auc=va_auc; no_improve=0
        torch.save({"state":model.state_dict()} , best_path)
        print(f"  -> Saved new best AUC={best_auc:.5f}")
    else:
        no_improve+=1
        if no_improve>=CFG["EARLY_STOP_PATIENCE"]:
            print("Early stopping."); break

# ======================
# Inference
# ======================
state=torch.load(best_path,map_location=device)
model.load_state_dict(state["state"]); model.eval()

test_ds=ClickDataset(test,feature_cols,seq_col,has_target=False,seq_maxlen=CFG["SEQ_MAXLEN"])
test_ld=DataLoader(test_ds,batch_size=CFG["BATCH_SIZE"],shuffle=False,collate_fn=collate_fn_infer)

preds=[]
with torch.no_grad():
    for xs,seqs,lens in tqdm(test_ld,desc="Inference"):
        xs,seqs,lens=xs.to(device),seqs.to(device),lens.to(device)
        logits=model(xs,seqs,lens)
        preds.append(torch.sigmoid(logits).cpu().numpy())
test_preds=np.concatenate(preds)

submit=pd.read_csv(SUB_PATH); submit["clicked"]=test_preds
submit.to_csv("./baseline_submit.csv",index=False)
print("Saved baseline_submit.csv")
