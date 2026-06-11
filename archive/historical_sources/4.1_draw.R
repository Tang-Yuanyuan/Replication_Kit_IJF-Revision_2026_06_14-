library(ggplot2)
library(dplyr)
library(broom)

# 设定保存路径
save_path <- "D:/RUC/B1WTA_new/TEST/empirical4.1_result"
if (!dir.exists(save_path)) dir.create(save_path, recursive = TRUE)

get_plot_data <- function(model, var_prefix, label) {
  s <- summary(model)
  coef_table <- as.data.frame(s$coefficients)
  coef_table$term <- rownames(coef_table)
  
  coef_data <- coef_table %>%
    filter(grepl(var_prefix, term)) %>%
    mutate(
      term = gsub(var_prefix, "", term),
      estimate = `Value`,
      std_error = `Std. Error`,
      conf.low = estimate - 1.96 * std_error,
      conf.high = estimate + 1.96 * std_error
    ) %>%
    select(term, estimate, conf.low, conf.high)
  
  base_row <- data.frame(term = "never", estimate = 0, conf.low = 0, conf.high = 0)
  coef_data <- rbind(base_row, coef_data)
  
  levels_order <- c("never", "heard but do not know", "heard and know", "familiar")
  coef_data$term <- factor(coef_data$term, levels = levels_order)
  coef_data$group <- label
  
  return(coef_data)
}

# --- 1. 提取数据 ---
plot_df_neutrality <- rbind(
  get_plot_data(model_car3, "know_about_carbon_neutrality", "WTA Transportation"),
  get_plot_data(model_elec3, "know_about_carbon_neutrality", "WTA Home Energy"),
  get_plot_data(model_green3, "know_about_carbon_neutrality", "WTA Green Electricity")
)

plot_df_policy <- rbind(
  get_plot_data(model_car4, "know_about_carbon_policy", "WTA Transportation"),
  get_plot_data(model_elec4, "know_about_carbon_policy", "WTA Home Energy"),
  get_plot_data(model_green4, "know_about_carbon_policy", "WTA Green Electricity")
)

# --- 2. 绘制并保存 inverted_U_2 (Carbon Neutrality) ---
p2 <- ggplot(plot_df_neutrality, aes(x = term, y = estimate, color = group, group = group, shape = group)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(position = position_dodge(0.2), linewidth = 1) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), 
                width = 0.1, linewidth = 0.8, 
                position = position_dodge(0.2)) +
  geom_point(size = 4, position = position_dodge(0.2)) +
  scale_color_manual(values = c("black", "gray40", "gray70")) + 
  labs(x = NULL, y = "Marginal Effect") + # 去掉横轴名，修改纵轴名
  theme_classic() +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 15, hjust = 1)
  )

ggsave(filename = file.path(save_path, "inverted_U_2.png"), 
       plot = p2, width = 8, height = 6, dpi = 300)

# --- 3. 绘制并保存 inverted_U_3 (Carbon Policy) ---
p3 <- ggplot(plot_df_policy, aes(x = term, y = estimate, color = group, group = group, shape = group)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(position = position_dodge(0.2), linewidth = 1) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), 
                width = 0.1, linewidth = 0.8, 
                position = position_dodge(0.2)) +
  geom_point(size = 4, position = position_dodge(0.2)) +
  scale_color_manual(values = c("black", "gray40", "gray70")) + 
  labs(x = NULL, y = "Marginal Effect") + # 去掉横轴名，修改纵轴名
  theme_classic() +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 15, hjust = 1)
  )

ggsave(filename = file.path(save_path, "inverted_U_3.png"), 
       plot = p3, width = 8, height = 6, dpi = 300)

message("已完成！图片保存在: ", save_path)