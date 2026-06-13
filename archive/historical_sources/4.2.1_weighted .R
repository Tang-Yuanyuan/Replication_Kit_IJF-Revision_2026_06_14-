
graphics.off()
rm(list = ls())
cat("\014") 

# .custom_lib <- "D:/R-4.5.2/Packages"
# if (dir.exists(.custom_lib)) .libPaths(c(.custom_lib, .libPaths()))
library(MASS)
library(dplyr)

# ----------------------------------------------------------------------------
# 1. 读取 Python 传来的数据
# ----------------------------------------------------------------------------

train_df <- read.csv("D:/RUC/B1WTA_new/TEST/empirical4.2_result/train_data.csv")
test_df  <- read.csv("D:/RUC/B1WTA_new/TEST/empirical4.2_result/test_data.csv")

# ----------------------------------------------------------------------------
# 2. 处理数据
# ----------------------------------------------------------------------------

prepare_data <- function(data) {
  data$location <- as.factor(data$location)
  data$province <- as.factor(data$province)
  data$weekday  <- as.factor(data$weekday)
  data$heard_about_global_warming <- as.factor(data$heard_about_global_warming)
  data$know_about_low_carbon      <- as.factor(data$know_about_low_carbon)
  data$know_about_carbon_neutrality <- as.factor(data$know_about_carbon_neutrality)
  data$know_about_carbon_policy     <- as.factor(data$know_about_carbon_policy)
  
  data$location <- relevel(data$location, ref = "city")
  data$heard_about_global_warming <- relevel(data$heard_about_global_warming, ref = "no")
  
  levels_know <- c("never", "heard but do not know", "heard and know", "familiar")
  data$know_about_low_carbon <- factor(data$know_about_low_carbon, levels = levels_know)
  data$know_about_carbon_neutrality <- factor(data$know_about_carbon_neutrality, levels = levels_know)
  data$know_about_carbon_policy <- factor(data$know_about_carbon_policy, levels = levels_know)
  
  return(data)
}

train_df <- prepare_data(train_df)
test_df  <- prepare_data(test_df)

# ----------------------------------------------------------------------------
# 3. 训练模型 (同步切换为加权 Logit)
# ----------------------------------------------------------------------------

base_demos <- "ifpollution + living_area_ln + age_ln + 
              is_bachelor + 
              location +   
              female + 
              married + 
              income_level + 
              youth + older_adults + 
              partymember +  
              province + weekday + ifsunny"

base_all <- "ifpollution + living_area_ln + age_ln + 
              is_bachelor + 
              location +   
              female + 
              married + 
              income_level + 
              youth + older_adults + 
              partymember +  
              province + weekday + ifsunny + 
              heard_about_global_warming + know_about_low_carbon + know_about_carbon_neutrality + know_about_carbon_policy"

# 分割子集并进行权重局部归一化 (与 Python 逻辑对齐)
train_df_car <- subset(train_df, publictrans < 5)
train_df_car$weights <- train_df_car$weights / mean(train_df_car$weights)

train_df_elec <- subset(train_df, conditionernumber == 1)
train_df_elec$weights <- train_df_elec$weights / mean(train_df_elec$weights)

train_df_green <- subset(train_df, energy_consume2020 > 1000)
train_df_green$weights <- train_df_green$weights / mean(train_df_green$weights)

# --- 定义加权训练函数，减少冗余代码并确保 weights 被正确调用 ---
train_weighted_logit <- function(formula_str, data_sub) {
  glm(as.formula(formula_str), 
      data = data_sub, 
      family = binomial(link = "logit"),
      weights = weights) # <--- 核心修改：注入权重
}

# --- CAR 组 ---
model_demo_car <- train_weighted_logit(paste("y_car ~", base_demos, " + caruse"), train_df_car)
model_all_car  <- train_weighted_logit(paste("y_car ~", base_all, " + caruse"), train_df_car)

# --- ELEC 组 ---
model_demo_elec <- train_weighted_logit(paste("y_elec ~", base_demos, " + conditioner1month"), train_df_elec)
model_all_elec  <- train_weighted_logit(paste("y_elec ~", base_all, " + conditioner1month"), train_df_elec)

# --- GREEN 组 ---
model_demo_green <- train_weighted_logit(paste("y_green ~", base_demos, " + mainuseelec"), train_df_green)
model_all_green  <- train_weighted_logit(paste("y_green ~", base_all, " + mainuseelec"), train_df_green)

# ----------------------------------------------------------------------------
# 4. 预测并分组成两个独立表 (Demos 组 & All 组)
# ----------------------------------------------------------------------------

# --- (A) Demos 组：只包含基础人口统计学变量的模型预测 ---
logit_probs_demos <- data.frame(
  prob_car   = predict(model_demo_car,   newdata = test_df, type = "response"),
  prob_elec  = predict(model_demo_elec,  newdata = test_df, type = "response"),
  prob_green = predict(model_demo_green, newdata = test_df, type = "response")
)

# --- (B) All 组：包含基础变量 + 认知变量的模型预测 ---
logit_probs_all <- data.frame(
  prob_car   = predict(model_all_car,   newdata = test_df, type = "response"),
  prob_elec  = predict(model_all_elec,  newdata = test_df, type = "response"),
  prob_green = predict(model_all_green, newdata = test_df, type = "response")
)

# ----------------------------------------------------------------------------
# 5. 导出到指定目录
# ----------------------------------------------------------------------------

# 导出 Demos 组概率表
write.csv(logit_probs_demos, 
          "D:/RUC/B1WTA_new/TEST/empirical4.2_result/logit_probs_demos_weights.csv", 
          row.names = FALSE)

# 导出 All 组概率表
write.csv(logit_probs_all, 
          "D:/RUC/B1WTA_new/TEST/empirical4.2_result/logit_probs_all_weights.csv", 
          row.names = FALSE)

message("Success: Logit probability tables exported.")

