import pandas as pd
output_path = r"D:\Project\Work\postdoctor\project\research\chengleilei\gpu\remaining_data\remaining_data_readmission.csv"
data = pd.read_csv(output_path, sep=",",encoding="gb18030")

print(data.head())

# === 2️⃣ 随机抽取100条 ===
sample_data = data.sample(n=int(len(data)*0.2), random_state=42)   # 可修改 random_state 保证可复现
remaining_data = data.drop(sample_data.index)       # 剩余部分

print(f"🎯 抽取样本: {len(sample_data)} 条, 剩余: {len(remaining_data)} 条")

# === 3️⃣ 保存为两个新文件 ===
sample_path = r"D:\Project\Work\postdoctor\project\research\chengleilei\gpu\remaining_data\sample_100.csv"
remaining_path = r"D:\Project\Work\postdoctor\project\research\chengleilei\gpu\remaining_data\remaining_data_readmission_0.8.csv"

sample_data.to_csv(sample_path, index=False, encoding="utf-8-sig")
remaining_data.to_csv(remaining_path, index=False, encoding="utf-8-sig")

print(f"✅ 已保存两个文件：\n  - 抽样数据：{sample_path}\n  - 剩余数据：{remaining_path}")
