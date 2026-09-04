# 读取CSV文件并提取指定变量列
library(dplyr)

# 1. 读取数据（请根据你的文件路径修改）
data <- read.csv("D:\\Project\\Work\\postdoctor\\project\\research\\chengleilei\\gpu\\remaining_data\\remaining_data.csv", header = TRUE, stringsAsFactors = FALSE, encoding = "UTF-8")

# 2. 指定需要保留的变量名称
selected_vars <- c(
  "Euroscore", "抢救记录", "住院天数", "是否再入院", "年龄", "入院年份",
  "是否上海人", "总费用", "更新死亡", "科室编码", "是否外地人", "更新医保",
  "性别", "高血压", "糖尿病", "高血脂", "心衰",
  "心律失常_室性类", "心律失常_心房颤动类", "急性心肌梗死", "冠心病",
  "心肌桥", "瓣膜病", "医生资历", "是否支架", "是否搭桥"
)

# 3. 提取这些列（若有列名不存在会自动忽略）
data_selected <- data %>% select(all_of(selected_vars))

data_selected$年龄 <- as.numeric(data_selected$年龄) 

data_selected$高血压 <- as.numeric(data_selected$高血压)

data_selected$性别 <- ifelse(data_selected$性别=="男",0,1) 

# 4. 查看结果
head(data_selected)

# 生成summary table

# 加载必要包
library(dplyr)
library(readr)


# 2️⃣ 定义统计函数
summary_table <- data_selected %>%
  summarise(across(
    everything(),
    ~ {
      # 如果列是数值型（numeric/integer）
      if (is.numeric(.x)) {
        # 判断是否二分类（只有两个唯一值）
        unique_vals <- unique(na.omit(.x))
        if (length(unique_vals) == 2 && all(unique_vals %in% c(0, 1))) {
          total <- sum(.x == 1, na.rm = TRUE)
          percent <- mean(.x == 1, na.rm = TRUE) * 100
          return(sprintf("%d (%.1f%%)", total, percent))
        } else {
          return(sprintf("%.2f ± %.2f", mean(.x, na.rm = TRUE), sd(.x, na.rm = TRUE)))
        }
      } else {
        # 非数值列则计算频数和百分比（如是字符二分类）
        unique_vals <- unique(na.omit(.x))
        if (length(unique_vals) == 2) {
          main_cat <- unique_vals[1]
          total <- sum(.x == main_cat, na.rm = TRUE)
          percent <- mean(.x == main_cat, na.rm = TRUE) * 100
          return(sprintf("%s: %d (%.1f%%)", main_cat, total, percent))
        } else {
          return("非二分类变量")
        }
      }
    }
  ))

# 3️⃣ 转换为长格式
summary_table_long <- data.frame(
  Variable = names(summary_table),
  Summary = t(summary_table)
)
rownames(summary_table_long) <- NULL

# 4️⃣ 保存为 CSV 文件
write.csv(summary_table_long, "D:\\Project\\Work\\postdoctor\\project\\research\\chengleilei\\gpu\\summary_table.csv", 
          row.names = FALSE, fileEncoding = "UTF-8")

# 5️⃣ 查看结果
print(summary_table_long)


#画变量概览图

# ============================================
# 📊 医疗数据分布可视化脚本（中文变量→英文首字母大写）
# 输入：原始数据表（每列一个变量）
# 输出：直方图 + 二值变量百分比图 + 英文版CSV
# ============================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(readr)
library(stringr)

# ==== 1️⃣ 读取数据 ====
df <- read_csv("D:\\Project\\Work\\postdoctor\\project\\research\\chengleilei\\gpu\\remaining_data\\remaining_data.csv")

df$性别 = ifelse(df$性别=="男",0,1)

# ==== 2️⃣ 变量翻译映射 ====
translate <- c(
  "年龄" = "Age",
  "性别" = "Gender",
  "Euroscore" = "Euroscore",
  "科室编码" = "Department_Code",
  "更新医保" = "Insurance",
  "医生资历" = "Doctor_Experience",
  "住院天数" = "Length_Of_Stay",
  "总费用" = "Total_Cost",
  "高血压" = "Hypertension",
  "糖尿病" = "Diabetes",
  "高血脂" = "Hyperlipidemia",
  "心衰" = "Heart_Failure",
  "心律失常_室性类" = "Ventricular_Arrhythmia",
  "心律失常_心房颤动类" = "Atrial_Fibrillation",
  "急性心肌梗死" = "Acute_Myocardial_Infarction",
  "冠心病" = "Coronary_Heart_Disease",
  "心肌桥" = "Myocardial_Bridge",
  "瓣膜病" = "Valvular_Disease",
  "是否支架" = "PCI",
  "是否搭桥" = "CABG",
  "抢救记录" = "Resuscitation",
  "是否再入院" = "Readmission",
  "更新死亡" = "Death"
)

# ==== 3️⃣ 只保留目标列并翻译 ====
df <- df %>% select(any_of(names(translate)))
for (v in names(translate)) {
  if (v %in% names(df)) {
    names(df)[names(df) == v] <- translate[[v]]
  }
}

# ==== 将整张表所有变量都转为 numeric ====
df <- df %>% mutate(across(everything(), ~ suppressWarnings(as.numeric(.))))

#df = na.omit(df)
# ==== 4️⃣ 手动指定变量类型 ====
continuous_vars <- c("Age", "Euroscore", "Length_Of_Stay", "Total_Cost")
binary_vars <- setdiff(names(df), continuous_vars)

cat("📊 连续变量:\n", paste(continuous_vars, collapse = ", "), "\n")
cat("📊 二值变量:\n", paste(binary_vars, collapse = ", "), "\n")

# ==== 5️⃣ 连续变量直方图（横坐标完全自适应）====
plots_cont <- list()

for (v in continuous_vars) {
  dat <- df %>% filter(!is.na(.data[[v]]))
  
  # 自动计算范围（排除极端值）
  q_low <- quantile(dat[[v]], 0.01, na.rm = TRUE)
  q_high <- quantile(dat[[v]], 0.99, na.rm = TRUE)
  range_val <- c(q_low, q_high)
  
  # 根据范围动态确定bin数量（10~60）
  data_range <- diff(range_val)
  n_bins <- min(max(round(data_range / (sd(dat[[v]], na.rm = TRUE) / 2)), 10), 60)
  binwidth_val <- data_range / n_bins
  
  # 自动确定x轴刻度
  breaks_seq <- pretty(range_val, n = 6)
  
  # 绘制直方图
  p <- ggplot(dat, aes_string(x = v)) +
    geom_histogram(
      fill = "lightblue", color = "white",
      binwidth = binwidth_val, boundary = 0, closed = "left"
    ) +
    geom_vline(aes(xintercept = mean(get(v), na.rm = TRUE)),
               linetype = "dashed", color = "white") +
    geom_vline(aes(xintercept = median(get(v), na.rm = TRUE)),
               linetype = "dotted", color = "white") +
    scale_x_continuous(
      limits = range_val,
      breaks = breaks_seq,
      expand = c(0, 0)
    ) +
    theme_minimal(base_size = 12) +
    ggtitle(paste("Histogram of", v)) +
    labs(x = v, y = "Count") +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.x = element_text(angle = 0, vjust = 0.5)
    )
  
  plots_cont[[v]] <- p
}



# ==== 6️⃣ 二值变量百分比柱状图 ====
binary_df <- df %>%
  select(any_of(binary_vars)) %>%
  # 👇 统一类型：转为字符，避免 numeric 与 character 混合出错
  mutate(across(everything(), ~ as.character(.))) %>%
  pivot_longer(
    cols = everything(),
    names_to = "Variable",
    values_to = "Value"
  ) %>%
  mutate(Value = as.factor(Value))

# 汇总统计
binary_summary <- binary_df %>%
  group_by(Variable, Value) %>%
  summarise(Count = n(), .groups = "drop") %>%
  group_by(Variable) %>%
  mutate(Percent = Count / sum(Count) * 100)

# ==== 6️⃣ 二值变量百分比柱状图（指定颜色）====

binary_summary$Variable <- factor(
  binary_summary$Variable,
  levels = rev(c(
    "Gender", "Department_Code", "Insurance", "Doctor_Experience",
    "Hypertension", "Diabetes", "Hyperlipidemia", "Heart_Failure",
    "Ventricular_Arrhythmia", "Atrial_Fibrillation", "Acute_Myocardial_Infarction",
    "Coronary_Heart_Disease", "Myocardial_Bridge", "Valvular_Disease",
    "PCI", "CABG", "Resuscitation", "Readmission", "Death"
  )
))

p_binary <- ggplot(binary_summary, aes(x = Variable, y = Percent, fill = Value)) +
  geom_bar(stat = "identity", position = "stack") +
  geom_text(aes(label = sprintf("%.1f%%", Percent)), 
            position = position_stack(vjust = 0.5), size = 3, color = "white") +
  coord_flip() +
  labs(y = "Percentage", x = NULL, title = "Binary Variables (%)") +
  scale_fill_manual(
    values = c(
      "0" = "#A1D99B",  # 浅绿色
      "1" = "#31A354",  # 深绿色
      "NA" = "gray80"   # 灰色
    ),
    name = "Value"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    plot.title = element_text(hjust = 0.5, face = "bold")
  )


# ==== 7️⃣ 拼接最终图 ====
final_plot <- wrap_plots(plots_cont, ncol = 2) / p_binary +
  plot_annotation(title = "Distribution of Baseline Variables",
                  theme = theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold")))


# ==== 9️⃣ 保存 ====
ggsave("D:\\Project\\Work\\postdoctor\\project\\research\\chengleilei\\gpu\\remaining_data\\variable_distribution_panel_translated.png", final_plot, width = 14, height = 12, dpi = 300)
ggsave("D:\\Project\\Work\\postdoctor\\project\\research\\chengleilei\\gpu\\remaining_data\\variable_distribution_panel_translated.pdf", final_plot, width = 14, height = 12)

cat("✅ 图像已保存为:\n - variable_distribution_panel_translated.png\n - variable_distribution_panel_translated.pdf\n")

# ==== 8️⃣ 保存输出 ====
write_csv(df, "remaining_data_translated_selected.csv")
ggsave("variable_distribution_final.png", final_plot, width = 14, height = 12, dpi = 300)
ggsave("variable_distribution_final.pdf", final_plot, width = 14, height = 12)

cat("✅ 已保存输出文件:\n",
    "- remaining_data_translated_selected.csv\n",
    "- variable_distribution_final.png\n",
    "- variable_distribution_final.pdf\n")



