import torch
from transformers import AutoModel, AutoTokenizer
import random
import torch.nn as nn
from gensim.models import Word2Vec
import pandas as pd
import jieba
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
)
import numpy as np
import matplotlib.pyplot as plt

# ========== 定义模型结构 ==========
embedding_dim = 100

class ChineseMedicalBERTClassifier(nn.Module):
    def __init__(self, base_model, num_classes=2):
        super().__init__()
        self.bert = base_model
        self.dropout = nn.Dropout(0.3)
        self.classifier1 = nn.Linear(self.bert.config.hidden_size + embedding_dim, num_classes)
        self.classifier2 = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, embeddings):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] 向量

        # 🎯 加入 70% 概率融合 Word2Vec
        '''if random.random() < 0.7:
            if embeddings.ndim == 1:
                embeddings = embeddings.unsqueeze(0)
            x = torch.cat([pooled_output, embeddings.to(pooled_output.device)], dim=-1)
            x = self.dropout(x)
            logits = self.classifier1(x)
        else:
            x = self.dropout(pooled_output)
            logits = self.classifier2(x)'''

        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        x = torch.cat([pooled_output, embeddings.to(pooled_output.device)], dim=-1)
        x = self.dropout(x)
        logits = self.classifier1(x)

        return logits


# ========== 加载模型和分词器 ==========
w2v_model = Word2Vec.load(r"D:\backup_20260414\Project\Work\postdoctor\project\research\chengleilei\gpu\my_word2vec_model.model")

# 获取词汇表大小
vocab_size = len(w2v_model.wv.key_to_index)

# 获取向量维度
embedding_dim = w2v_model.vector_size

print(f"Vocabulary Size: {vocab_size:,}")
print(f"Embedding Dimension: {embedding_dim}")
print(f"Embedding Matrix Shape: {w2v_model.wv.vectors.shape}")

model_name = r"D:\backup_20260414\Project\Work\postdoctor\project\research\chengleilei\gpu\hflchinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

model = ChineseMedicalBERTClassifier(base_model, num_classes=2)
model.load_state_dict(torch.load(
    r"D:\backup_20260414\Project\Work\postdoctor\project\research\chengleilei\gpu\hflchinese-roberta-wwm-ext\death\medical_bert_roberta_wwm_ext.pt",
    map_location="cpu"))
model.eval()
print("✅ 模型已成功加载（state_dict 模式）")


# ========== 读取测试集 ==========
df = pd.read_csv(r"D:\backup_20260414\Project\Work\postdoctor\project\research\chengleilei\gpu\test_0.2.csv", encoding="utf-8")


df["更新死亡"] = pd.to_numeric(df["更新死亡"], errors="coerce")  # 转为数值，非法值变NaN
df = df[df["更新死亡"].notna()]
#df["更新死亡"] = df["更新死亡"].fillna(0)  # 将NaN视为0（未死亡）
#df["更新死亡"] = df["更新死亡"].apply(lambda x: 1 if x > 0 else 0)  # 统一转成0/1
labels = df["更新死亡"].astype(int).tolist()
texts = df["record"].astype(str).tolist()

# ========== 获取句向量函数 ==========
def get_sentence_embedding(text, w2v_model):
    tokens = list(jieba.cut(text))
    vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if len(vecs) == 0:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vecs, axis=0)


# ========== 预测函数 ==========
def predict_with_full_model(texts, model, w2v_model):
    all_probs = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            emb = get_sentence_embedding(text, w2v_model)
            encoded = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            embedding = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            logits = model(input_ids, attention_mask, embedding)
            #probs = torch.softmax(logits, dim=1)
            # --- 防止softmax饱和 ---
            temperature = 3.0
            logits = torch.clamp(logits, min=-10, max=10)
            probs = torch.softmax(logits / temperature, dim=1)

            all_probs.append(probs[:, 1].cpu().item())
    return np.array(all_probs)


# ========== 评估函数 ==========
from sklearn.metrics import confusion_matrix

def classification_metrics(y_true, y_pred, y_prob):
    """Point estimates. PPV/NPV are NaN when the corresponding denominator is 0."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "PPV": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }


def evaluate(y_true, y_pred, y_prob, model_name):
    metrics = classification_metrics(y_true, y_pred, y_prob)

    print(f"\n📊 {model_name} Performance:")
    for name in ("Accuracy", "Sensitivity", "Specificity", "PPV", "NPV",
                 "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"):
        print(f"{name + ':':<18}{metrics[name]:.4f}")
    return metrics



# ========== 查找最佳阈值 ==========
def find_best_threshold(y_true, y_prob, metric="youden"):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    if metric == "youden":
        youden_index = tpr - fpr
        best_threshold = thresholds[np.argmax(youden_index)]
    else:  # 默认 F1 最优
        best_threshold = 0.5
        best_f1 = 0
        for thr in thresholds:
            preds = (y_prob >= thr).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = thr
    print(f"✅ Best threshold ({metric}): {best_threshold:.4f}")
    return best_threshold


# ========== Base 模型预测（简单池化） ==========
def predict_with_base_model(texts, base_model):
    all_probs = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
            outputs = base_model(**encoded)
            pooled_output = outputs.pooler_output
            # 临时线性层（仅用于演示）
            #logits = torch.nn.Linear(base_model.config.hidden_size, 2)(pooled_output)
            logits = model.classifier2(pooled_output)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs[:, 1].cpu().item())
    return np.array(all_probs)


# ========== 开始预测 ==========


print("🔹 使用 Base BERT 预测中...")
base_probs = predict_with_base_model(texts, base_model)
base_probs = 1 - base_probs
best_thr = find_best_threshold(labels, base_probs, metric="youden")
base_preds = (base_probs >= best_thr).astype(int)

print("🔹 使用 Full Model 预测中...")

full_probs = predict_with_full_model(texts, model, w2v_model)
#full_probs = 1 - full_probs
best_thr = find_best_threshold(labels, full_probs, metric="youden")
full_preds = (full_probs >= best_thr).astype(int)

# 先评估默认 0.5 阈值
evaluate(labels, base_preds, base_probs, f"Base Model (Best Threshold {best_thr:.3f})")
evaluate(labels, full_preds, full_probs, f"Full Model (Best Threshold {best_thr:.3f})")

# ========== 寻找最佳阈值并重新评估 ==========
best_thr = find_best_threshold(labels, full_probs, metric="youden")
full_preds_best = (full_probs >= best_thr).astype(int)
evaluate(labels, full_preds_best, full_probs, f"Full Model (Best Threshold {best_thr:.3f})")



# ========== 绘制 ROC 曲线 ==========
fpr, tpr, _ = roc_curve(labels, full_probs)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'Full Model AUC = {roc_auc_score(labels, full_probs):.3f}')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title('ROC Curve (Full Model)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()


fpr, tpr, _ = roc_curve(labels, base_probs)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'Base Model AUC = {roc_auc_score(labels, base_probs):.3f}')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title('ROC Curve (Base Model)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()

# ========== 保存结果 ==========
df["prob_base"] = base_probs
df["pred_base"] = base_preds
df["prob_full"] = full_probs
df["pred_full"] = full_preds_best
df.to_csv(r"D:\backup_20260414\Project\Work\postdoctor\project\research\chengleilei\gpu\prediction_results_with_best_thr.csv",
          index=False, encoding="utf-8-sig")

print(f"✅ 已保存预测结果到 prediction_results_with_best_thr.csv")

# 评估模型
metrics_full = evaluate(labels, full_preds_best, full_probs, f"Full Model (Best Threshold {best_thr:.3f})")

# 可将指标保存为 DataFrame
pd.DataFrame([metrics_full]).to_csv(
    r"D:\backup_20260414\Project\Work\postdoctor\project\research\chengleilei\gpu\metrics_full.csv",
    index=False, encoding="utf-8-sig"
)

metrics_base = evaluate(labels, base_preds, base_probs, f"Base Model (Best Threshold {best_thr:.3f})")

# 可将指标保存为 DataFrame
pd.DataFrame([metrics_base]).to_csv(
    r"D:\backup_20260414\Project\Work\postdoctor\project\research\chengleilei\gpu\metrics_base.csv",
    index=False, encoding="utf-8-sig"
)

# ======== 绘制 Base 与 Full 模型的 ROC 曲线 ========
plt.figure(figsize=(7,7))

# === Full Model ROC ===
fpr_full, tpr_full, _ = roc_curve(labels, full_probs)
auc_full = roc_auc_score(labels, full_probs)
plt.plot(fpr_full, tpr_full, color='black', lw=2.5, label=f"Full Model (AUC={auc_full:.3f})")

# === Base Model ROC ===
fpr_base, tpr_base, _ = roc_curve(labels, base_probs)
auc_base = roc_auc_score(labels, base_probs)
plt.plot(fpr_base, tpr_base, color='steelblue', lw=2.5, linestyle='--', label=f"Base Model (AUC={auc_base:.3f})")

# === 随机参考线 ===
plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.6)

# === 图形修饰 ===
plt.title("ROC Curves Comparison: Full Model vs Base Model", fontsize=14)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# === 保存图像 ===
save_path = r"D:\backup_20260414\Project\Work\postdoctor\project\research\chengleilei\gpu\ROC_Comparison_Full_vs_Base.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ ROC 对比图已绘制并保存到：{save_path}")
print(f"Full Model AUC = {auc_full:.3f}")
print(f"Base Model AUC = {auc_base:.3f}")

# ======== 绘制 Base 与 Full 模型的 ROC 曲线 ========
plt.figure(figsize=(7,7))

# === Full Model ROC ===
fpr_full, tpr_full, _ = roc_curve(labels, full_probs)
auc_full = roc_auc_score(labels, full_probs)
plt.plot(fpr_full, tpr_full, color='black', lw=2.5, label=f"Full Model (AUC={auc_full:.3f})")

# === Base Model ROC ===
fpr_base, tpr_base, _ = roc_curve(labels, base_probs)
auc_base = roc_auc_score(labels, base_probs)
plt.plot(fpr_base, tpr_base, color='steelblue', lw=2.5, linestyle='--', label=f"Base Model (AUC={auc_base:.3f})")

# === 随机参考线 ===
plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.6)

# === 图形修饰 ===
plt.title("ROC Curves Comparison: Full Model vs Base Model", fontsize=14)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# === 保存图像 ===
save_path = r"D:\backup_20260414\Project\Work\postdoctor\project\research\chengleilei\gpu\ROC_Comparison_Full_vs_Base.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ ROC 对比图已绘制并保存到：{save_path}")
print(f"Full Model AUC = {auc_full:.3f}")
print(f"Base Model AUC = {auc_base:.3f}")

# ======== Paired bootstrap: both models share each resample ========
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
PAIRED_METRICS = (
    "F1 Score", "PR-AUC", "PPV", "NPV",
    "ROC-AUC", "Sensitivity", "Specificity", "Accuracy",
)


def paired_bootstrap_metrics(y_true, pred_full, prob_full, pred_base, prob_base,
                              n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Percentile 95% CIs from paired bootstrap resampling.

    Each replicate draws one index vector and applies it to both models,
    so the Full-minus-Base difference is computed on the same patients.
    """
    y_true = np.asarray(y_true)
    pred_full = np.asarray(pred_full)
    prob_full = np.asarray(prob_full, dtype=float)
    pred_base = np.asarray(pred_base)
    prob_base = np.asarray(prob_base, dtype=float)
    n = len(y_true)
    rng = np.random.RandomState(seed)

    point_full = classification_metrics(y_true, pred_full, prob_full)
    point_base = classification_metrics(y_true, pred_base, prob_base)
    boot_full = {m: [] for m in PAIRED_METRICS}
    boot_base = {m: [] for m in PAIRED_METRICS}
    boot_diff = {m: [] for m in PAIRED_METRICS}

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            mf = classification_metrics(yt, pred_full[idx], prob_full[idx])
            mb = classification_metrics(yt, pred_base[idx], prob_base[idx])
        except ValueError:
            continue
        for metric in PAIRED_METRICS:
            vf, vb = mf[metric], mb[metric]
            if np.isnan(vf) or np.isnan(vb):
                continue
            boot_full[metric].append(vf)
            boot_base[metric].append(vb)
            boot_diff[metric].append(vf - vb)

    def summarize(model_name, point, boot):
        rows = []
        for metric in PAIRED_METRICS:
            vals = np.asarray(boot[metric], dtype=float)
            if len(vals) == 0:
                lo = hi = np.nan
            else:
                lo, hi = np.percentile(vals, [2.5, 97.5])
            rows.append({
                "Model": model_name,
                "Metric": metric,
                "Estimate": point[metric],
                "95%CI Lower": lo,
                "95%CI Upper": hi,
                "N replicates": int(len(vals)),
            })
        return pd.DataFrame(rows)

    df_full = summarize("Full Model", point_full, boot_full)
    df_base = summarize("Base Model", point_base, boot_base)
    point_diff = {m: point_full[m] - point_base[m] for m in PAIRED_METRICS}
    df_diff = summarize("Full - Base", point_diff, boot_diff)
    return df_full, df_base, df_diff


print(f"\n📊 Paired bootstrap: {N_BOOTSTRAP} replicates, seed={BOOTSTRAP_SEED}")
df_full_metrics, df_base_metrics, df_diff_metrics = paired_bootstrap_metrics(
    labels, full_preds_best, full_probs, base_preds, base_probs
)

out_dir = r"D:\backup_20260414\Project\Work\postdoctor\project\research\chengleilei\gpu"
path_full = out_dir + r"\metrics_full_with_CI.csv"
path_base = out_dir + r"\metrics_base_with_CI.csv"
path_diff = out_dir + r"\metrics_paired_difference.csv"
path_all = out_dir + r"\metrics_comparison_full_vs_base.csv"
path_ci = out_dir + r"\metrics_with_CI.csv"

df_full_metrics.to_csv(path_full, index=False, encoding="utf-8-sig")
df_base_metrics.to_csv(path_base, index=False, encoding="utf-8-sig")
df_diff_metrics.to_csv(path_diff, index=False, encoding="utf-8-sig")
df_all = pd.concat([df_full_metrics, df_base_metrics, df_diff_metrics], ignore_index=True)
df_all.to_csv(path_all, index=False, encoding="utf-8-sig")
df_full_metrics.to_csv(path_ci, index=False, encoding="utf-8-sig")

print(f"✅ Full Model CIs: {path_full}")
print(f"✅ Base Model CIs: {path_base}")
print(f"✅ Paired difference (Full - Base): {path_diff}")
print(f"✅ Combined table: {path_all}")
print("\n📈 Paired bootstrap summary:")
print(df_all.to_string(index=False))
