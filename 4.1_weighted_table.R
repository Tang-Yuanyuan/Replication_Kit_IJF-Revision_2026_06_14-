# ==============================================================================
# 汇总对比脚本：Unweighted (4.1.R) vs. Weighted (4.1weighted.R)
# ==============================================================================

# --- 第一步：环境保护 ---
# 定义临时函数屏蔽 rm()，保护内存中的模型对象
old_rm <- rm
rm <- function(...) invisible(NULL) 

# --- 第二步：运行 4.1.R 并暂存未加权模型 ---
cat("\n正在加载 4.1.R (Unweighted)...\n")
source("D:/RUC/B1WTA_new/TEST/4.1.R", encoding = "UTF-8")

m_un_car   <- list(model_car1, model_car2, model_car3, model_car4)
m_un_elec  <- list(model_elec1, model_elec2, model_elec3, model_elec4)
m_un_green <- list(model_green1, model_green2, model_green3, model_green4)

# --- 第三步：运行 4.1weighted.R 并暂存加权模型 ---
cat("\n正在加载 4.1weighted.R (Weighted)...\n")
source("D:/RUC/B1WTA_new/TEST/4.1weighted.R", encoding = "UTF-8")

m_wt_car   <- list(model_car1_post, model_car2_post, model_car3_post, model_car4_post)
m_wt_elec  <- list(model_elec1_post, model_elec2_post, model_elec3_post, model_elec4_post)
m_wt_green <- list(model_green1_post, model_green2_post, model_green3_post, model_green4_post)

# 恢复 rm 函数
rm <- old_rm 

# --- 第四步：定义统一提取与对比函数 ---
# 该函数会自动识别 polr 和 svyolr 对象的系数差异
compare_results <- function(un_list, wt_list, group_name) {
  
  labels <- c("GW", "LC", "Neu", "Pol")
  
  tidy_model <- function(m) {
    s <- coef(summary(m))
    is_threshold <- grepl("\\|", rownames(s))
    s_coef <- s[!is_threshold, , drop = FALSE]
    # 识别列名：svyolr 是 Value, polr 是 Estimate
    c_idx <- if("Value" %in% colnames(s_coef)) "Value" else "Estimate"
    data.frame(
      term  = rownames(s_coef),
      coef  = as.numeric(s_coef[, c_idx]),
      se    = as.numeric(s_coef[, "Std. Error"]),
      tstat = as.numeric(s_coef[, ncol(s_coef)]),
      stringsAsFactors = FALSE
    )
  }
  
  # 提取所有变量
  all_vars <- unique(unlist(lapply(un_list, function(x) rownames(coef(summary(x))))))
  # 剔除截距项和固定效应
  omit_pat <- "province|weekday|ifsunny|\\|"
  display_vars <- all_vars[!grepl(omit_pat, all_vars)]
  
  final_tab <- data.frame(Variable = display_vars, stringsAsFactors = FALSE)
  
  # 循环填充 8 列 (4个方案 x 2个状态)
  for (i in 1:4) {
    res_un <- tidy_model(un_list[[i]])
    res_wt <- tidy_model(wt_list[[i]])
    
    for (type in c("Un", "Wt")) {
      col_name <- paste0(labels[i], "(", type, ")")
      r <- if(type == "Un") res_un else res_wt
      
      vals <- character(length(display_vars))
      for (j in seq_along(display_vars)) {
        match <- which(r$term == display_vars[j])
        if (length(match) > 0) {
          co <- r$coef[match]; se <- r$se[match]; ts <- r$tstat[match]
          stars <- ifelse(abs(ts) > 2.576, "***", ifelse(abs(ts) > 1.960, "**", ifelse(abs(ts) > 1.645, "*", "")))
          vals[j] <- sprintf("%.3f%s (%.3f)", co, stars, se)
        } else { vals[j] <- "" }
      }
      final_tab[[col_name]] <- vals
    }
  }
  
  cat("\n", paste(rep("=", 80), collapse=""), "\n")
  cat("GROUP:", group_name, "- Comparison of Unweighted (Un) vs Weighted (Wt)\n")
  cat(paste(rep("=", 80), collapse=""), "\n")
  print(final_tab, row.names = FALSE)
  return(final_tab)
}

# --- 第五步：输出结果 ---

table_car   <- compare_results(m_un_car, m_wt_car, "CAR")
table_elec  <- compare_results(m_un_elec, m_wt_elec, "ELEC")
table_green <- compare_results(m_un_green, m_wt_green, "GREEN")

# write.csv(table_car, "D:/RUC/B1WTA_new/TEST/empirical4.1_result/Comparison_CAR.csv", row.names = FALSE)
# write.csv(table_elec, "D:/RUC/B1WTA_new/TEST/empirical4.1_result/Comparison_CAR.csv", row.names = FALSE)
# write.csv(table_green, "D:/RUC/B1WTA_new/TEST/empirical4.1_result/Comparison_CAR.csv", row.names = FALSE)

