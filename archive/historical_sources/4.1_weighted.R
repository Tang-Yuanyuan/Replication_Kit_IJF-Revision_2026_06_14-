graphics.off()
rm(list = ls())
cat("\014")

.custom_lib <- "D:/R-4.5.2/Packages"
if (dir.exists(.custom_lib)) .libPaths(c(.custom_lib, .libPaths()))

library(MASS)
library(dplyr)
library(survey)

df <- read.csv("D:/RUC/B1WTA_new/TEST/energy_wta.csv")

# ============================================================
# 1. 数据处理
# ============================================================

df_clean <- df %>%
  mutate(
    edu_strat = case_when(
      education == "uneducated"  ~ "Uneducated",
      education == "primary"     ~ "Primary",
      education == "junior high" ~ "Junior high",
      education == "senior high" ~ "Senior high",
      education %in% c("associate", "bachelor", "postgraduate") ~ "College or higher",
      TRUE ~ NA_character_
    ),
    loc_strat = case_when(
      location %in% c("city", "county seat") ~ "Urban",
      location == "rural"                    ~ "Rural",
      TRUE ~ NA_character_
    ),
    gender_strat = factor(as.character(female), levels = c("0", "1")),
    loc_strat    = factor(loc_strat,    levels = c("Rural", "Urban")),
    edu_strat    = factor(edu_strat,    levels = c("Uneducated", "Primary", "Junior high", "Senior high", "College or higher"))
  ) %>%
  filter(!is.na(gender_strat), !is.na(loc_strat), !is.na(edu_strat), !is.na(age))

n_clean <- nrow(df_clean)

# ============================================================
# 2. 自动化构建目标向量 (解决名称不匹配问题)
# ============================================================

# 定义一个目标值列表（方便管理）
targets <- list(
  # 截距项即总人数
  "(Intercept)" = n_clean,
  # 变量名必须与 model.matrix 生成的 dummy 变量名一致
  "gender_strat1" = n_clean * 0.488,
  "loc_stratUrban" = n_clean * 0.639,
  "edu_stratPrimary" = n_clean * 0.244,
  "edu_stratJunior high" = n_clean * 0.345,
  "edu_stratSenior high" = n_clean * 0.151,
  "edu_stratCollege or higher" = n_clean * 0.155,
  "age" = n_clean * 38.8 # 38.8
)

# 获取模型矩阵的正确顺序
mm <- model.matrix(~ gender_strat + loc_strat + edu_strat + age, data = df_clean)
pop_totals <- numeric(ncol(mm))
names(pop_totals) <- colnames(mm)

# 按照 R 自动生成的顺序填充目标值
for(n in names(pop_totals)) {
  if(n %in% names(targets)) {
    pop_totals[n] <- targets[[n]]
  } else {
    # 如果有没定义的类别，报错提醒
    stop(paste("未定义目标值的变量名:", n))
  }
}

# ============================================================
# 3. 执行校准
# ============================================================

ids <- svydesign(ids = ~1, data = df_clean)

# 使用 raking 函数，它比 linear 更稳健，且不会产生负权重
# 移除 bounds 限制以确保收敛，后续再统一 trim
cal_design_joint <- survey::calibrate(
  design     = ids,
  formula    = ~ gender_strat + loc_strat + edu_strat + age,
  population = pop_totals,
  calfun     = "raking",   # 改用 raking 提高收敛成功率
  epsilon    = 1e-7,
  maxit      = 2000
)

# ============================================================
# 4. 权重处理
# ============================================================

# 获取原始校准权重
raw_w <- weights(cal_design_joint)
mean_w <- mean(raw_w)

# 设置倍数限制
lower_bound <- mean_w * 0.29 # 0.29
upper_bound <- mean_w * 9.5 # 9.5

# 执行修剪
cal_trimmed <- survey::trimWeights(
  cal_design_joint, 
  lower = lower_bound, 
  upper = upper_bound, 
  strict = TRUE
)

# 提取并归一化
df_clean$weights <- weights(cal_trimmed)
df_clean$weights <- df_clean$weights / mean(df_clean$weights)

# 合并回原表 
df$weights <- NA
df[rownames(df_clean), "weights"] <- df_clean$weights

# ============================================================
# 5. 权重验证
# ============================================================
check_design <- svydesign(ids = ~1, weights = ~weights, data = df_clean)

# --- 教育维度 ---
res_table_edu <- prop.table(svytable(~edu_strat, check_design))
target_edu <- c(
  "Uneducated"       = 0.027,
  "Primary"          = 0.244,
  "Junior high"      = 0.345,
  "Senior high"      = 0.151,
  "College or higher"= 0.155
)
diffs_edu     <- round(abs(res_table_edu[names(target_edu)] - target_edu), 3)
mean_diff_edu <- round(mean(diffs_edu, na.rm = TRUE), 3)

# --- 性别维度 ---
res_gender  <- prop.table(svytable(~gender_strat, check_design))
diff_gender <- round(abs(res_gender["1"] - 0.488), 3)

# --- 城乡维度 ---
res_loc  <- prop.table(svytable(~loc_strat, check_design))
diff_loc <- round(abs(res_loc["Urban"] - 0.639), 3)

# --- 年龄维度 ---
weighted_age_mean <- svymean(~age, check_design)
diff_age <- round(abs(coef(weighted_age_mean) - 38.8), 3)

# --- 打印报告 ---
cat("\n================ 权重验证报告 ================\n")
cat(sprintf("【教育维度】平均绝对偏差 (MAE): %.3f\n", mean_diff_edu))
cat(sprintf("【教育维度】单项偏差: %s\n",
            paste(names(diffs_edu), diffs_edu, sep = "=", collapse = ", ")))
cat(sprintf("【性别维度】女性占比偏差: %.3f\n", diff_gender))
cat(sprintf("【城乡维度】城镇占比偏差: %.3f\n", diff_loc))
cat(sprintf("【年龄维度】加权均值: %.3f  |  目标: %.1f  |  偏差: %.3f\n",
            coef(weighted_age_mean), 38.8, diff_age))
cat("-----------------------------------------------------------\n")

#cat("\n--- 加权后精确比例 ---\n")
print(round(prop.table(svytable(~edu_strat,  check_design)) * 100, 1))
print(round(prop.table(svytable(~gender_strat, check_design)) * 100, 1))
print(round(prop.table(svytable(~loc_strat,  check_design)) * 100, 1))

# ============================================================
# 6. 权重回归
# ============================================================
df$wta_car  <- factor(df$wta_car,  levels = 1:7, ordered = TRUE)
df$wta_elec <- factor(df$wta_elec, levels = 1:7, ordered = TRUE)
df$wta_green<- factor(df$wta_green,levels = 1:7, ordered = TRUE)

df$education  <- relevel(factor(df$education),  ref = "uneducated")
df$location   <- relevel(factor(df$location),   ref = "city")
df$marriage   <- relevel(factor(df$marriage),   ref = "unmarried")
df$youth      <- relevel(factor(df$youth),      ref = "0")
df$older_adults<-relevel(factor(df$older_adults),ref = "0")
df$province   <- as.factor(df$province)
df$weekday    <- as.factor(df$weekday)
df$partymember<- relevel(factor(df$partymember),ref = "0")
df$ifsunny    <- relevel(factor(df$ifsunny),    ref = "0")
df$mainuseelec<- relevel(factor(df$mainuseelec),ref = "0")
df$heard_about_global_warming    <- relevel(factor(df$heard_about_global_warming), ref = "no")
df$know_about_low_carbon         <- factor(df$know_about_low_carbon,
                   levels = c("never","heard but do not know","heard and know","familiar"))
df$know_about_carbon_neutrality  <- factor(df$know_about_carbon_neutrality,
                   levels = c("never","heard but do not know","heard and know","familiar"))
df$know_about_carbon_policy      <- factor(df$know_about_carbon_policy,
                   levels = c("never","heard but do not know","heard and know","familiar"))

df$ifpollution <- ifelse(df$aqi >= 101, 1, 0)
df$married     <- as.numeric(df$marriage == "married")
df$age_ln      <- log(df$age)
df$living_area_ln <- log(df$living_area)
df$is_bachelor <- ifelse(df$education %in% c("bachelor", "postgraduate"), 1, 0)
df$caruse <- ifelse(df$carusetime == 0, 1, 0)


base_vars <- "ifpollution + living_area_ln + age_ln + 
              is_bachelor + location + female + married + 
              income_level + youth + older_adults + partymember + 
              province + weekday + ifsunny"


df_car_sub   <- subset(df, publictrans < 5)
df_elec_sub  <- subset(df, conditionernumber == 1)
df_green_sub <- subset(df, energy_consume2020 > 1000)

svy_car   <- svydesign(ids = ~1, weights = ~weights, data = df_car_sub)
svy_elec  <- svydesign(ids = ~1, weights = ~weights, data = df_elec_sub)
svy_green <- svydesign(ids = ~1, weights = ~weights, data = df_green_sub)


print_svyolr <- function(model, title = "") {
  cat("\n============================\n", title, "\n============================\n")
  coef_mat <- coef(summary(model))
  print(round(coef_mat, 4))
}



model_car1_post <- svyolr(as.formula(paste("wta_car ~", base_vars, "+ caruse + heard_about_global_warming")),
                     design = svy_car)
model_car2_post <- svyolr(as.formula(paste("wta_car ~", base_vars, "+ caruse + know_about_low_carbon")),
                     design = svy_car)
model_car3_post <- svyolr(as.formula(paste("wta_car ~", base_vars, "+ caruse + know_about_carbon_neutrality")),
                     design = svy_car)
model_car4_post <- svyolr(as.formula(paste("wta_car ~", base_vars, "+ caruse + know_about_carbon_policy")),
                     design = svy_car)


model_elec1_post <- svyolr(as.formula(paste("wta_elec ~", base_vars, "+ conditioner1month + heard_about_global_warming")),
                      design = svy_elec)
model_elec2_post <- svyolr(as.formula(paste("wta_elec ~", base_vars, "+ conditioner1month + know_about_low_carbon")),
                      design = svy_elec)
model_elec3_post <- svyolr(as.formula(paste("wta_elec ~", base_vars, "+ conditioner1month + know_about_carbon_neutrality")),
                      design = svy_elec)
model_elec4_post <- svyolr(as.formula(paste("wta_elec ~", base_vars, "+ conditioner1month + know_about_carbon_policy")),
                      design = svy_elec)


model_green1_post <- svyolr(as.formula(paste("wta_green ~", base_vars, "+ mainuseelec + heard_about_global_warming")),
                       design = svy_green)
model_green2_post <- svyolr(as.formula(paste("wta_green ~", base_vars, "+ mainuseelec + know_about_low_carbon")),
                       design = svy_green)
model_green3_post <- svyolr(as.formula(paste("wta_green ~", base_vars, "+ mainuseelec + know_about_carbon_neutrality")),
                       design = svy_green)
model_green4_post <- svyolr(as.formula(paste("wta_green ~", base_vars, "+ mainuseelec + know_about_carbon_policy")),
                       design = svy_green)

# ============================================================
# 7. 格式化输出
# ============================================================

# 只保留后续回归所需变量 + id
keep_vars <- c(
  "id",                          # 标识符
  # 因变量
  "wta_car", "wta_elec", "wta_green",
  # base_vars
  "ifpollution", "living_area_ln", "age_ln", "is_bachelor",
  "location", "female", "married", "income_level",
  "youth", "older_adults", "partymember",
  "province", "weekday", "ifsunny",
  # 各模型专属控制变量
  "caruse", "conditioner1month", "mainuseelec",
  # 知识/认知变量（四个模型分别用一个）
  "heard_about_global_warming", "know_about_low_carbon",
  "know_about_carbon_neutrality", "know_about_carbon_policy",
  # 子集筛选变量
  "publictrans", "conditionernumber", "energy_consume2020",
  # 权重
  "weights"
)

df <- df %>% select(any_of(keep_vars))

write.csv(df, file = "D:/RUC/B1WTA_new/TEST/energy_wta_with_post_weights.csv", 
          row.names = FALSE, fileEncoding = "UTF-8")
# 回归表格
m_wt_car   <- list(model_car1_post, model_car2_post, model_car3_post, model_car4_post)
m_wt_elec  <- list(model_elec1_post, model_elec2_post, model_elec3_post, model_elec4_post)
m_wt_green <- list(model_green1_post, model_green2_post, model_green3_post, model_green4_post)

export_weighted_results <- function(wt_list, group_name) {
  
  labels <- c("GW", "LC", "Neu", "Pol")
  
  # 内部清理函数：提取系数、标准误并打星
  tidy_model <- function(m) {
    s <- coef(summary(m))
    # 过滤掉截距项/阈值项（通常包含 "|"）
    is_threshold <- grepl("\\|", rownames(s))
    s_coef <- s[!is_threshold, , drop = FALSE]
    
    # 识别列名：svyolr 使用 "Value"，polr 使用 "Estimate"
    c_idx <- if("Value" %in% colnames(s_coef)) "Value" else "Estimate"
    
    data.frame(
      term  = rownames(s_coef),
      coef  = as.numeric(s_coef[, c_idx]),
      se    = as.numeric(s_coef[, "Std. Error"]),
      tstat = as.numeric(s_coef[, ncol(s_coef)]), # 最后一列通常是 t 或 z 值
      stringsAsFactors = FALSE
    )
  }
  
  # 获取所有模型中出现的变量并去重
  all_vars <- unique(unlist(lapply(wt_list, function(x) rownames(coef(summary(x))))))
  # 过滤掉不需要显示的变量（省份、周几、天气固定效应等）
  omit_pat <- "province|weekday|ifsunny|\\|"
  display_vars <- all_vars[!grepl(omit_pat, all_vars)]
  
  final_tab <- data.frame(Variable = display_vars, stringsAsFactors = FALSE)
  
  # 填充 4 列结果
  for (i in 1:4) {
    res <- tidy_model(wt_list[[i]])
    col_name <- labels[i]
    
    vals <- character(length(display_vars))
    for (j in seq_along(display_vars)) {
      match <- which(res$term == display_vars[j])
      if (length(match) > 0) {
        co <- res$coef[match]
        se <- res$se[match]
        ts <- res$tstat[match]
        # 显著性打星标准 (z > 2.576: 1%, 1.96: 5%, 1.645: 10%)
        stars <- ifelse(abs(ts) > 2.576, "***", 
                 ifelse(abs(ts) > 1.960, "**", 
                 ifelse(abs(ts) > 1.645, "*", "")))
        vals[j] <- sprintf("%.3f%s (%.3f)", co, stars, se)
      } else {
        vals[j] <- "" 
      }
    }
    final_tab[[col_name]] <- vals
  }
  
  cat("\n", paste(rep("-", 50), collapse=""), "\n")
  cat("GROUP:", group_name, "- Weighted Regression Results\n")
  cat(paste(rep("-", 50), collapse=""), "\n")
  print(final_tab, row.names = FALSE)
  return(final_tab)
}


table_car   <- export_weighted_results(m_wt_car,   "CAR")
table_elec  <- export_weighted_results(m_wt_elec,  "ELEC")
table_green <- export_weighted_results(m_wt_green, "GREEN")


