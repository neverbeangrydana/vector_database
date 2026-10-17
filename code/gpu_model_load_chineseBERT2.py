import torch
from transformers import AutoModel, AutoTokenizer
import random
import torch.nn as nn
from gensim.models import Word2Vec
import pandas as pd
import jieba
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, average_precision_score, confusion_matrix,
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
        self.classifier = nn.Linear(self.bert.config.hidden_size+embedding_dim, num_classes)
        self.classifier2 = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.w2v_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input_ids,attention_mask,embeddings):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] 表征
        '''if random.random() < 0.7:
            # 假设 embeddings 是 list，需要转换
            embeddings = [torch.tensor(e, dtype=torch.float32) for e in embeddings]
            embeddings = torch.stack(embeddings, dim=0)

            x = torch.cat([pooled_output, embeddings], dim=-1)  # [B, 1068]
            x = self.dropout(x)
            logits = self.classifier(x)
            return logits
        else:
            x = self.dropout(pooled_output)
            logits = self.classifier2(x)
            return logits'''
        embeddings = self.w2v_norm(embeddings)

        embeddings = [torch.tensor(e, dtype=torch.float32) for e in embeddings]
        embeddings = torch.stack(embeddings, dim=0)

        x = torch.cat([pooled_output, embeddings], dim=-1)  # [B, 1068]
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


# ========== 加载模型和分词器 ==========
w2v_model = Word2Vec.load(r"D:\backup_20260414\Project\Work\postdoctor\project\research\chengleilei\gpu\my_word2vec_model.model")
model_name = r"D:\backup_20260414\Project\Work\postdoctor\project\research\chengleilei\gpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

model = ChineseMedicalBERTClassifier(base_model, num_classes=2)
model.load_state_dict(torch.load(
    r"D:\backup_20260414\Project\\Work\\postdoctor\\project\\research\\chengleilei\\gpu\\chineseBert+ VectorDB\\death\\medical_bert_death_epoch1.pt",
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
def _safe_div(num, den):
    return float(num / den) if den > 0 else 0.0


def compute_metrics(y_true, y_pred, y_prob):
    """阈值相关指标用 y_pred，AUC / PR-AUC 用 y_prob。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": _safe_div(tp, tp + fn),
        "Specificity": _safe_div(tn, tn + fp),
        "PPV": _safe_div(tp, tp + fp),
        "NPV": _safe_div(tn, tn + fn),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1 Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "PR-AUC": float(average_precision_score(y_true, y_prob)),
    }


def evaluate(y_true, y_pred, y_prob, model_name):
    m = compute_metrics(y_true, y_pred, y_prob)

    print(f"\n📊 {model_name} Performance:")
    print(f"Accuracy:        {m['Accuracy']:.4f}")
    print(f"Sensitivity:     {m['Sensitivity']:.4f}")
    print(f"Specificity:     {m['Specificity']:.4f}")
    print(f"PPV:             {m['PPV']:.4f}")
    print(f"NPV:             {m['NPV']:.4f}")
    print(f"Recall:          {m['Recall']:.4f}")
    print(f"F1 Score:        {m['F1 Score']:.4f}")
    print(f"AUC:             {m['AUC']:.4f}")
    print(f"PR-AUC:          {m['PR-AUC']:.4f}")
    return m



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
#base_probs = 1 - base_probs
base_thr = find_best_threshold(labels, base_probs, metric="youden")
base_preds = (base_probs >= base_thr).astype(int)

print("🔹 使用 Full Model 预测中...")

full_probs = predict_with_full_model(texts, model, w2v_model)
#full_probs = 1 - full_probs
full_thr = find_best_threshold(labels, full_probs, metric="youden")
full_preds = (full_probs >= full_thr).astype(int)
best_thr = full_thr
full_preds_best = full_preds

evaluate(labels, base_preds, base_probs, f"Base Model (Best Threshold {base_thr:.3f})")
evaluate(labels, full_preds_best, full_probs, f"Full Model (Best Threshold {full_thr:.3f})")



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

metrics_base = evaluate(labels, base_preds, base_probs, f"Base Model (Best Threshold {base_thr:.3f})")

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

# ======== Paired bootstrap（同一批受试者索引，1000 次） ========
# 每次重抽样同时取 Full 与 Base 的预测，差值 CI 与 p 值因此是配对的。
# 阈值锁定为各模型在全测试集上的 Youden 阈值，不在每次重抽样里重选。
PAIRED_METRICS = [
    "F1 Score", "PR-AUC", "PPV", "NPV",
    "AUC", "Sensitivity", "Specificity", "Accuracy",
]
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


def paired_bootstrap(y_true, pred_full, prob_full, pred_base, prob_base,
                     n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    y_true = np.asarray(y_true)
    pred_full = np.asarray(pred_full)
    prob_full = np.asarray(prob_full, dtype=float)
    pred_base = np.asarray(pred_base)
    prob_base = np.asarray(prob_base, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    point_full = compute_metrics(y_true, pred_full, prob_full)
    point_base = compute_metrics(y_true, pred_base, prob_base)

    boots_full = {m: [] for m in PAIRED_METRICS}
    boots_base = {m: [] for m in PAIRED_METRICS}
    n_skipped = 0

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if np.unique(yt).size < 2:
            n_skipped += 1
            continue
        try:
            mf = compute_metrics(yt, pred_full[idx], prob_full[idx])
            mb = compute_metrics(yt, pred_base[idx], prob_base[idx])
        except Exception:
            n_skipped += 1
            continue
        if any(np.isnan(mf[m]) or np.isnan(mb[m]) for m in PAIRED_METRICS):
            n_skipped += 1
            continue
        for m in PAIRED_METRICS:
            boots_full[m].append(mf[m])
            boots_base[m].append(mb[m])
        if (i + 1) % 200 == 0:
            print(f"  paired bootstrap {i + 1}/{n_bootstrap}")

    model_rows = []
    diff_rows = []
    for m in PAIRED_METRICS:
        bf = np.asarray(boots_full[m], dtype=float)
        bb = np.asarray(boots_base[m], dtype=float)
        if bf.size == 0:
            raise RuntimeError(f"paired bootstrap produced no valid replicates for {m}")
        diff = bf - bb
        obs_diff = point_full[m] - point_base[m]
        flo, fhi = np.percentile(bf, [2.5, 97.5])
        blo, bhi = np.percentile(bb, [2.5, 97.5])
        dlo, dhi = np.percentile(diff, [2.5, 97.5])
        # 以 bootstrap 均值重定中心，得到原假设“差值为 0”下的双侧 p 值
        null_diff = diff - diff.mean()
        p_value = (np.sum(np.abs(null_diff) >= abs(obs_diff)) + 1) / (diff.size + 1)

        model_rows.append({
            "Model": "Full Model",
            "Metric": m,
            "Point": point_full[m],
            "Bootstrap Mean": float(bf.mean()),
            "95% CI Lower": float(flo),
            "95% CI Upper": float(fhi),
            "N Replicates": int(bf.size),
        })
        model_rows.append({
            "Model": "Base Model",
            "Metric": m,
            "Point": point_base[m],
            "Bootstrap Mean": float(bb.mean()),
            "95% CI Lower": float(blo),
            "95% CI Upper": float(bhi),
            "N Replicates": int(bb.size),
        })
        diff_rows.append({
            "Metric": m,
            "Full": point_full[m],
            "Base": point_base[m],
            "Difference (Full-Base)": obs_diff,
            "95% CI Lower": float(dlo),
            "95% CI Upper": float(dhi),
            "p-value": float(p_value),
            "N Replicates": int(diff.size),
        })

    return pd.DataFrame(model_rows), pd.DataFrame(diff_rows), n_skipped


print(f"\n📊 Paired bootstrap resampling: {N_BOOTSTRAP} replicates (seed={BOOTSTRAP_SEED})")
df_paired, df_diff, n_skipped = paired_bootstrap(
    labels, full_preds_best, full_probs, base_preds, base_probs
)
print(f"有效重复次数: {int(df_paired['N Replicates'].iloc[0])}，跳过单类别样本: {n_skipped}")

df_full_metrics = df_paired[df_paired["Model"] == "Full Model"].reset_index(drop=True)
df_base_metrics = df_paired[df_paired["Model"] == "Base Model"].reset_index(drop=True)


def _fmt_estimate(value):
    rounded = round(float(value), 2)
    if rounded == 0:
        return "0"
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def write_comparison_table(df_full, df_base, df_diff, path, model_label="ChineseBERT2"):
    """与 metrics_comparison_full_vs_base_ROBERTa.csv 相同的列和分组版式。"""
    display_name = {"AUC": "ROC-AUC"}
    groups = [
        ("Full Model", df_full.set_index("Metric"), "Point"),
        ("Base Model", df_base.set_index("Metric"), "Point"),
        ("Full - Base", df_diff.set_index("Metric"), "Difference (Full-Base)"),
    ]
    rows = []
    first_row = True
    for model_name, frame, estimate_col in groups:
        first_in_group = True
        for metric in PAIRED_METRICS:
            rec = frame.loc[metric]
            rows.append({
                model_label: "",
                "Model": model_name if first_in_group else "",
                "Metric": display_name.get(metric, metric),
                "Estimate": _fmt_estimate(rec[estimate_col]),
                "95%CI Lower": _fmt_estimate(rec["95% CI Lower"]),
                "95%CI Upper": _fmt_estimate(rec["95% CI Upper"]),
                "N replicates": str(int(rec["N Replicates"])) if first_row else "",
            })
            first_row = False
            first_in_group = False
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


out_dir = r"D:\backup_20260414\Project\Work\postdoctor\project\research\chengleilei\gpu"
path_full = out_dir + r"\metrics_full_with_CI.csv"
path_base = out_dir + r"\metrics_base_with_CI.csv"
path_paired = out_dir + r"\metrics_paired_bootstrap.csv"
path_diff = out_dir + r"\metrics_paired_difference.csv"
path_all = out_dir + r"\metrics_comparison_full_vs_base.csv"
path_ci = out_dir + r"\metrics_with_CI.csv"

df_full_metrics.to_csv(path_full, index=False, encoding="utf-8-sig")
df_base_metrics.to_csv(path_base, index=False, encoding="utf-8-sig")
df_paired.to_csv(path_paired, index=False, encoding="utf-8-sig")
df_diff.to_csv(path_diff, index=False, encoding="utf-8-sig")
write_comparison_table(df_full_metrics, df_base_metrics, df_diff, path_all)
df_full_metrics.to_csv(path_ci, index=False, encoding="utf-8-sig")

print(f"✅ Full Model 指标: {path_full}")
print(f"✅ Base Model 指标: {path_base}")
print(f"✅ 配对 bootstrap 明细: {path_paired}")
print(f"✅ Full − Base 差值、95%CI 与 p 值: {path_diff}")

print("\n📈 各模型点估计与 95% CI（paired bootstrap）:")
print(df_paired.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n📈 Full − Base（同一批重抽样索引）:")
print(df_diff.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

