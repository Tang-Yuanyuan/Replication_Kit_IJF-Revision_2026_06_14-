graphics.off()
rm(list = ls())
cat("\014") 

.custom_lib <- "D:/R-4.5.2/Packages"
if (dir.exists(.custom_lib)) .libPaths(c(.custom_lib, .libPaths()))

library(MASS)
library(dplyr)

# ----------------------------------------------------------------------------
# 1. 读取 Python 传来的数据
# ----------------------------------------------------------------------------
train_df <- read.csv("D:/RUC/B1WTA_new/TEST/empirical4.2_result/train_data.csv")
test_df  <- read.csv("D:/RUC/B1WTA_new/TEST/empirical4.2_result/test_data_simulated_weighted.csv")

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
# 3. 训练 Logit 分类模型 (加权)
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

# 分割子集并进行权重局部归一化
train_df_car <- subset(train_df, publictrans < 5 )
train_df_car$weights <- train_df_car$weights / mean(train_df_car$weights)

train_df_elec <- subset(train_df, conditionernumber == 1)
train_df_elec$weights <- train_df_elec$weights / mean(train_df_elec$weights)

train_df_green <- subset(train_df, energy_consume2020 > 1000)
train_df_green$weights <- train_df_green$weights / mean(train_df_green$weights)

# 定义加权训练函数
train_weighted_logit <- function(formula_str, data_sub) {
  glm(as.formula(formula_str), 
      data = data_sub, 
      family = binomial(link = "logit"),
      weights = weights) # 注入权重
}

# 训练分类模型
model_demo_car <- train_weighted_logit(paste("y_car ~", base_demos, " + caruse"), train_df_car)
model_all_car  <- train_weighted_logit(paste("y_car ~", base_all, " + caruse"), train_df_car)

model_demo_elec <- train_weighted_logit(paste("y_elec ~", base_demos, " + conditioner1month"), train_df_elec)
model_all_elec  <- train_weighted_logit(paste("y_elec ~", base_all, " + conditioner1month"), train_df_elec)

model_demo_green <- train_weighted_logit(paste("y_green ~", base_demos, " + mainuseelec"), train_df_green)
model_all_green  <- train_weighted_logit(paste("y_green ~", base_all, " + mainuseelec"), train_df_green)

# ----------------------------------------------------------------------------
# 4. 预测概率并导出 (_weights)
# ----------------------------------------------------------------------------

logit_probs_demos <- data.frame(
  prob_car   = predict(model_demo_car,   newdata = test_df, type = "response"),
  prob_elec  = predict(model_demo_elec,  newdata = test_df, type = "response"),
  prob_green = predict(model_demo_green, newdata = test_df, type = "response")
)

logit_probs_all <- data.frame(
  prob_car   = predict(model_all_car,   newdata = test_df, type = "response"),
  prob_elec  = predict(model_all_elec,  newdata = test_df, type = "response"),
  prob_green = predict(model_all_green, newdata = test_df, type = "response")
)

write.csv(logit_probs_demos, 
          "D:/RUC/B1WTA_new/TEST/empirical4.3_result/logit_probs_demos_simulated_weights.csv", 
          row.names = FALSE)

write.csv(logit_probs_all, 
          "D:/RUC/B1WTA_new/TEST/empirical4.3_result/logit_probs_all_simulated_weights.csv", 
          row.names = FALSE)

# ----------------------------------------------------------------------------
# 5. WTA 模型 (Ordered Logit 加权)
# ----------------------------------------------------------------------------

wta_levels <- c("1", "2", "3", "4", "5", "6", "7")

train_df_car$wta_car     <- factor(train_df_car$wta_car, levels = wta_levels, ordered = TRUE)
train_df_elec$wta_elec    <- factor(train_df_elec$wta_elec, levels = wta_levels, ordered = TRUE)
train_df_green$wta_green  <- factor(train_df_green$wta_green, levels = wta_levels, ordered = TRUE)

# 训练序数回归模型 (加入 weights 参数)
model_demo_car_polr  <- polr(as.formula(paste("wta_car ~", base_demos, " + caruse")), 
                             data = train_df_car, weights = weights, Hess = TRUE)
model_demo_elec_polr <- polr(as.formula(paste("wta_elec ~", base_demos, " + conditioner1month")), 
                             data = train_df_elec, weights = weights, Hess = TRUE)
model_demo_green_polr <- polr(as.formula(paste("wta_green ~", base_demos, " + mainuseelec")), 
                             data = train_df_green, weights = weights, Hess = TRUE)

model_all_car_polr   <- polr(as.formula(paste("wta_car ~", base_all, " + caruse")), 
                             data = train_df_car, weights = weights, Hess = TRUE)
model_all_elec_polr  <- polr(as.formula(paste("wta_elec ~", base_all, " + conditioner1month")), 
                             data = train_df_elec, weights = weights, Hess = TRUE)
model_all_green_polr <- polr(as.formula(paste("wta_green ~", base_all, " + mainuseelec")), 
                             data = train_df_green, weights = weights, Hess = TRUE)

# 期望值计算函数
calc_expected_wta_safe <- function(model, train_data, test_data) {
  resp_var <- as.character(formula(model)[[2]])
  predictor_vars <- setdiff(all.vars(formula(model)), resp_var)
  
  for (v in predictor_vars) {
    if (is.factor(train_data[[v]])) {
      test_data[[v]] <- factor(test_data[[v]], levels = levels(train_data[[v]]))
    }
  }
  
  res <- tryCatch({
    probs <- predict(model, newdata = test_data, type = "probs", na.action = na.pass)
    levels_num <- as.numeric(colnames(probs))
    expected_score <- probs %*% levels_num
    as.vector(expected_score)
  }, error = function(e) {
    cat("\n⚠️ 预测出错:", e$message)
    return(rep(NA, nrow(test_data)))
  })
  return(res)
}

# ----------------------------------------------------------------------------
# 6. 预测 WTA 并导出 (_weights)
# ----------------------------------------------------------------------------

logit_preds_demos <- data.frame(
  pred_wta_car   = calc_expected_wta_safe(model_demo_car_polr,   train_df_car,   test_df),
  pred_wta_elec  = calc_expected_wta_safe(model_demo_elec_polr,  train_df_elec,  test_df),
  pred_wta_green = calc_expected_wta_safe(model_demo_green_polr, train_df_green, test_df)
)

logit_preds_all <- data.frame(
  pred_wta_car   = calc_expected_wta_safe(model_all_car_polr,   train_df_car,   test_df),
  pred_wta_elec  = calc_expected_wta_safe(model_all_elec_polr,  train_df_elec,  test_df),
  pred_wta_green = calc_expected_wta_safe(model_all_green_polr, train_df_green, test_df)
)

write.csv(logit_preds_demos, 
          "D:/RUC/B1WTA_new/TEST/empirical4.3_result/wta_preds_demos_simulated_weights.csv", 
          row.names = FALSE)

write.csv(logit_preds_all, 
          "D:/RUC/B1WTA_new/TEST/empirical4.3_result/wta_preds_all_simulated_weights.csv", 
          row.names = FALSE)

cat("\n✅ 加权 Logit & Ologit 模拟预测完成！")
cat("\n📂 结果已存至 empirical4.3_result 并带 _weights 后缀。")