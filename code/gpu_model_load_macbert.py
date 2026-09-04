import torch
from transformers import AutoModel, AutoTokenizer
import random
import torch.nn as nn
from gensim.models import Word2Vec
import pandas as pd
import jieba
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

# ========== 定义模型结构 ==========
embedding_dim = 100

class ChineseMedicalBERTClassifier(nn.Module):
    def __init__(self, base_model, num_classes=2):
        super().__init__()
        self.bert = base_model
        self.dropout = nn.Dropout(0.3)
        self.classifier1 = nn.Linear(self.bert.config.hidden_size+embedding_dim, num_classes)
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
        logits = self.classifier1(x)
        return logits

# ========== 加载模型和分词器 ==========
w2v_model = Word2Vec.load(r"D:\Project\Work\postdoctor\project\research\chengleilei\gpu\my_word2vec_model.model")
model_name = r"D:\Project\Work\postdoctor\project\research\chengleilei\gpu\hflchinese-macbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

model = ChineseMedicalBERTClassifier(base_model, num_classes=2)
model.load_state_dict(torch.load(
    r"D:\Project\Work\postdoctor\project\research\chengleilei\gpu\hflchinese-macbert-base\death\chinese-macbert-base0.pt",
    map_location="cpu"))
model.eval()
print("✅ 模型已成功加载（state_dict 模式）")


# ========== 读取测试集 ==========
df = pd.read_csv(r"D:\Project\Work\postdoctor\project\research\chengleilei\gpu\test_0.1.csv", encoding="utf-8")


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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate(y_true, y_pred, y_prob, model_name):
    # === 1. 基础指标 ===
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    # === 2. 混淆矩阵 ===
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # === 3. 医学类指标 ===
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 与 recall 一致
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0          # 阳性预测值
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0          # 阴性预测值

    # === 4. 输出 ===
    print(f"\n📊 {model_name} Performance:")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"PPV (Precision): {ppv:.4f}")
    print(f"NPV:          {npv:.4f}")
    print(f"Recall:       {rec:.4f}")
    print(f"F1-score:     {f1:.4f}")
    print(f"AUC:          {auc:.4f}")

    # === 5. 返回所有指标 ===
    return {
        "Accuracy": acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc
    }



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
df.to_csv(r"D:\Project\Work\postdoctor\project\research\chengleilei\gpu\prediction_results_with_best_thr.csv",
          index=False, encoding="utf-8-sig")

print(f"✅ 已保存预测结果到 prediction_results_with_best_thr.csv")

# 评估模型
metrics_full = evaluate(labels, full_preds_best, full_probs, f"Full Model (Best Threshold {best_thr:.3f})")

# 可将指标保存为 DataFrame
pd.DataFrame([metrics_full]).to_csv(
    r"D:\Project\Work\postdoctor\project\research\chengleilei\gpu\metrics_full.csv",
    index=False, encoding="utf-8-sig"
)

metrics_base = evaluate(labels, base_preds, base_probs, f"Base Model (Best Threshold {best_thr:.3f})")

# 可将指标保存为 DataFrame
pd.DataFrame([metrics_full]).to_csv(
    r"D:\Project\Work\postdoctor\project\research\chengleilei\gpu\metrics_base.csv",
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
save_path = r"D:\Project\Work\postdoctor\project\research\chengleilei\gpu\ROC_Comparison_Full_vs_Base.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ ROC 对比图已绘制并保存到：{save_path}")
print(f"Full Model AUC = {auc_full:.3f}")
print(f"Base Model AUC = {auc_base:.3f}")

import scipy.stats as st
import seaborn as sns

# ======== 修正版：Bootstrap CI，防止AUC为空 ========
def bootstrap_ci(y_true, y_prob, metric_func, n_bootstrap=1000, seed=42):
    np.random.seed(seed)
    boot_values = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        y_t = np.array(y_true)[idx]
        y_p = np.array(y_prob)[idx]

        # 跳过只有一个类别的样本
        if len(np.unique(y_t)) < 2:
            continue

        try:
            val = metric_func(y_t, y_p)
            if not np.isnan(val):
                boot_values.append(val)
        except:
            continue

    # 如果boot_values为空，则直接计算总体指标
    if len(boot_values) == 0:
        print("⚠️ Warning: bootstrap_ci() failed, fallback to direct computation.")
        try:
            val = metric_func(y_true, y_prob)
            return val, val, val
        except:
            return 0.0, 0.0, 0.0

    mean_val = np.mean(boot_values)
    ci_lower, ci_upper = np.percentile(boot_values, [2.5, 97.5])
    return mean_val, ci_lower, ci_upper


# ======== 计算核心指标 + 置信区间 ========
print("\n📊 计算 AUC, Sensitivity, Specificity, Accuracy 的95%CI...")

metrics_ci = {}
metrics_ci["AUC"] = bootstrap_ci(labels, full_probs, roc_auc_score)
metrics_ci["Sensitivity"] = bootstrap_ci(labels, full_preds_best, lambda y_true, y_pred: recall_score(y_true, y_pred))
metrics_ci["Specificity"] = bootstrap_ci(labels, full_preds_best, lambda y_true, y_pred:
                                         confusion_matrix(y_true, y_pred).ravel()[0] /
                                         np.sum(confusion_matrix(y_true, y_pred).ravel()[0:2]))
metrics_ci["Accuracy"] = bootstrap_ci(labels, full_preds_best, accuracy_score)

df_metrics_ci = pd.DataFrame([
    {"Metric": k, "Mean": v[0], "95%CI Lower": v[1], "95%CI Upper": v[2]}
    for k, v in metrics_ci.items()
])

print(df_metrics_ci)
df_metrics_ci.to_csv(
    r"D:\Project\Work\postdoctor\project\research\chengleilei\gpu\metrics_with_CI.csv",
    index=False, encoding="utf-8-sig"
)
print("✅ 已保存 metrics_with_CI.csv")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

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
save_path = r"D:\Project\Work\postdoctor\project\research\chengleilei\gpu\ROC_Comparison_Full_vs_Base.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ ROC 对比图已绘制并保存到：{save_path}")
print(f"Full Model AUC = {auc_full:.3f}")
print(f"Base Model AUC = {auc_base:.3f}")

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, roc_auc_score

# ======== 修正版 Bootstrap CI 函数 ========
def bootstrap_ci(y_true, y_pred, y_prob, metric_func, n_bootstrap=1000, seed=42):
    np.random.seed(seed)
    boot_values = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        y_t = np.array(y_true)[idx]
        y_p = np.array(y_pred)[idx]
        y_pb = np.array(y_prob)[idx]

        # 跳过只有一个类别的样本
        if len(np.unique(y_t)) < 2:
            continue

        try:
            val = metric_func(y_t, y_p, y_pb)
            if not np.isnan(val):
                boot_values.append(val)
        except:
            continue

    # 如果boot_values为空，则直接计算总体指标
    if len(boot_values) == 0:
        print("⚠️ Warning: bootstrap_ci() failed, fallback to direct computation.")
        try:
            val = metric_func(y_true, y_pred, y_prob)
            return val, val, val
        except:
            return 0.0, 0.0, 0.0

    mean_val = np.mean(boot_values)
    ci_lower, ci_upper = np.percentile(boot_values, [2.5, 97.5])
    return mean_val, ci_lower, ci_upper


# ======== 计算四个核心指标的函数 ========
def compute_metrics_with_ci(y_true, y_pred, y_prob, model_name):
    print(f"\n📊 计算 {model_name} 的 AUC, Sensitivity, Specificity, Accuracy 及其95%CI...")

    def auc_metric(y_t, y_p, y_pb):
        return roc_auc_score(y_t, y_pb)
    def sens_metric(y_t, y_p, y_pb):
        return recall_score(y_t, y_p)
    def spec_metric(y_t, y_p, y_pb):
        tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    def acc_metric(y_t, y_p, y_pb):
        return accuracy_score(y_t, y_p)

    metrics_ci = {}
    metrics_ci["AUC"] = bootstrap_ci(y_true, y_pred, y_prob, auc_metric)
    metrics_ci["Sensitivity"] = bootstrap_ci(y_true, y_pred, y_prob, sens_metric)
    metrics_ci["Specificity"] = bootstrap_ci(y_true, y_pred, y_prob, spec_metric)
    metrics_ci["Accuracy"] = bootstrap_ci(y_true, y_pred, y_prob, acc_metric)

    df_metrics_ci = pd.DataFrame([
        {"Model": model_name, "Metric": k,
         "Mean": v[0], "95%CI Lower": v[1], "95%CI Upper": v[2]}
        for k, v in metrics_ci.items()
    ])

    return df_metrics_ci


# ======== 分别计算 Full Model 与 Base Model 的指标 ========
df_full_metrics = compute_metrics_with_ci(labels, full_preds_best, full_probs, "Full Model")
df_base_metrics = compute_metrics_with_ci(labels, base_preds, base_probs, "Base Model")

# ======== 保存各自指标 ========
path_full = r"D:\Project\Work\postdoctor\project\research\chengleilei\gpu\metrics_full_with_CI.csv"
path_base = r"D:\Project\Work\postdoctor\project\research\chengleilei\gpu\metrics_base_with_CI.csv"
df_full_metrics.to_csv(path_full, index=False, encoding="utf-8-sig")
df_base_metrics.to_csv(path_base, index=False, encoding="utf-8-sig")

print(f"✅ 已保存 Full Model 指标至：{path_full}")
print(f"✅ 已保存 Base Model 指标至：{path_base}")

# ======== 汇总保存到一个总表 ========
df_all = pd.concat([df_full_metrics, df_base_metrics], ignore_index=True)
path_all = r"D:\Project\Work\postdoctor\project\research\chengleilei\gpu\metrics_comparison_full_vs_base.csv"
df_all.to_csv(path_all, index=False, encoding="utf-8-sig")
print(f"✅ 已生成汇总文件：{path_all}")

print("\n📈 各模型指标摘要：")
print(df_all)
