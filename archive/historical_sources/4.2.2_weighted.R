graphics.off()
rm(list = ls())
cat("\014") 

# .custom_lib <- "D:/R-4.5.2/Packages"
# if (dir.exists(.custom_lib)) .libPaths(c(.custom_lib, .libPaths()))

library(MASS)
library(dplyr)

# ----------------------------------------------------------------------------
# 1. 读取数据
# ----------------------------------------------------------------------------
train_df <- read.csv("D:/RUC/B1WTA_new/TEST/empirical4.2_result/train_data.csv")
test_df  <- read.csv("D:/RUC/B1WTA_new/TEST/empirical4.2_result/test_data.csv")

# ----------------------------------------------------------------------------
# 2. 数据处理函数
# ----------------------------------------------------------------------------
prepare_data <- function(data) {
  wta_levels <- c("1", "2", "3", "4", "5", "6", "7")
  data$wta_car   <- factor(data$wta_car, levels = wta_levels, ordered = TRUE)
  data$wta_elec  <- factor(data$wta_elec, levels = wta_levels, ordered = TRUE)
  data$wta_green <- factor(data$wta_green, levels = wta_levels, ordered = TRUE)

  data$location <- as.factor(data$location)
  data$province <- as.factor(data$province)
  data$weekday  <- as.factor(data$weekday)
  data$heard_about_global_warming <- as.factor(data$heard_about_global_warming)
  data$know_about_low_carbon      <- as.factor(data$know_about_low_carbon)
  data$know_about_carbon_neutrality <- as.factor(data$know_about_carbon_neutrality)
  data$know_about_carbon_policy     <- as.factor(data$know_about_carbon_policy)
  
  data$location <- relevel(data$location, ref = "city")
  data$heard_about_global_warming <- factor(data$heard_about_global_warming, 
                                     levels = c("no", "yes and agree", "yes but disagree"))  
  levels_know <- c("never", "heard but do not know", "heard and know", "familiar")
  data$know_about_low_carbon <- factor(data$know_about_low_carbon, levels = levels_know)
  data$know_about_carbon_neutrality <- factor(data$know_about_carbon_neutrality, levels = levels_know)
  data$know_about_carbon_policy <- factor(data$know_about_carbon_policy, levels = levels_know)
  
  return(data)
}

train_df <- prepare_data(train_df)
test_df  <- prepare_data(test_df)

# ----------------------------------------------------------------------------
# 3. 训练加权 ologit 模型 (Ordered Logit)
# ----------------------------------------------------------------------------
base_demos <- "ifpollution + living_area_ln + age_ln + is_bachelor + location + female + married + income_level + youth + older_adults + partymember + province + weekday + ifsunny"
base_all   <- paste(base_demos, "+ heard_about_global_warming + know_about_low_carbon + know_about_carbon_neutrality + know_about_carbon_policy")

# --- 核心修改：子集划分 + 权重局部归一化 ---
train_df_car   <- subset(train_df, publictrans < 5)
train_df_car$weights   <- train_df_car$weights / mean(train_df_car$weights)

train_df_elec  <- subset(train_df, conditionernumber == 1)
train_df_elec$weights  <- train_df_elec$weights / mean(train_df_elec$weights)

train_df_green <- subset(train_df, energy_consume2020 > 1000)
train_df_green$weights <- train_df_green$weights / mean(train_df_green$weights)

# --- 核心修改：在 polr 中加入 weights 参数 ---
# 训练模型 (Demos 组)
model_demo_car   <- polr(as.formula(paste("wta_car ~", base_demos, " + caruse")), 
                        data = train_df_car, weights = weights, Hess = TRUE)
model_demo_elec  <- polr(as.formula(paste("wta_elec ~", base_demos, " + conditioner1month")), 
                        data = train_df_elec, weights = weights, Hess = TRUE)
model_demo_green <- polr(as.formula(paste("wta_green ~", base_demos, " + mainuseelec")), 
                        data = train_df_green, weights = weights, Hess = TRUE)

# 训练模型 (All 组)
model_all_car    <- polr(as.formula(paste("wta_car ~", base_all, " + caruse")), 
                        data = train_df_car, weights = weights, Hess = TRUE)
model_all_elec   <- polr(as.formula(paste("wta_elec ~", base_all, " + conditioner1month")), 
                        data = train_df_elec, weights = weights, Hess = TRUE)
model_all_green  <- polr(as.formula(paste("wta_green ~", base_all, " + mainuseelec")), 
                        data = train_df_green, weights = weights, Hess = TRUE)

# ----------------------------------------------------------------------------
# 4. 执行预测 (保持逻辑不变)
# ----------------------------------------------------------------------------
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
    message("⚠️ 预测出错: ", e$message)
    return(rep(NA, nrow(test_data)))
  })
  return(res)
}

logit_preds_demos <- data.frame(
  pred_wta_car   = calc_expected_wta_safe(model_demo_car,   train_df_car,   test_df),
  pred_wta_elec  = calc_expected_wta_safe(model_demo_elec,  train_df_elec,  test_df),
  pred_wta_green = calc_expected_wta_safe(model_demo_green, train_df_green, test_df)
)

logit_preds_all <- data.frame(
  pred_wta_car   = calc_expected_wta_safe(model_all_car,   train_df_car,   test_df),
  pred_wta_elec  = calc_expected_wta_safe(model_all_elec,  train_df_elec,  test_df),
  pred_wta_green = calc_expected_wta_safe(model_all_green, train_df_green, test_df)
)

# ----------------------------------------------------------------------------
# 5. 导出结果 (文件名与 Python 读取端对齐)
# ----------------------------------------------------------------------------
write.csv(logit_preds_demos, "D:/RUC/B1WTA_new/TEST/empirical4.2_result/wta_preds_demos_weights.csv", row.names = FALSE)
write.csv(logit_preds_all,   "D:/RUC/B1WTA_new/TEST/empirical4.2_result/wta_preds_all_weights.csv", row.names = FALSE)

message("✅ 加权 ologit 预测完成！已导出连续型期望 WTA 分值。")