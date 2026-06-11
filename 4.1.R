
graphics.off()
rm(list = ls())
cat("\014") 

.custom_lib <- "D:/R-4.5.2/Packages"
if (dir.exists(.custom_lib)) .libPaths(c(.custom_lib, .libPaths()))

library(MASS)
library(dplyr)
library(stargazer)
library(brant)
library(survey)
library(VGAM)
library(ggplot2)
library(tidyr)
library(modelsummary)

df <- read.csv("D:/RUC/B1WTA_new/TEST/energy_wta.csv")

#----------------------------------------------------------------------------
# 1.数据处理
#----------------------------------------------------------------------------

#（1）分类变量、定序

df$wta_car <- factor(df$wta_car, levels = 1:7, ordered = TRUE)
df$wta_elec <- factor(df$wta_elec, levels = 1:7, ordered = TRUE)
df$wta_green <- factor(df$wta_green, levels = 1:7, ordered = TRUE)

df$education<- as.factor(df$education)
df$location<- as.factor(df$location)
df$marriage<- as.factor(df$marriage)
df$youth<- as.factor(df$youth)
df$older_adults<- as.factor(df$older_adults)
df$province<- as.factor(df$province)
df$weekday<- as.factor(df$weekday)
df$partymember<- as.factor(df$partymember)
df$weather<- as.factor(df$weather)
df$mainuseelec<- as.factor(df$mainuseelec)
df$heard_about_global_warming <- as.factor(df$heard_about_global_warming )
df$know_about_low_carbon <- as.factor(df$know_about_low_carbon )
df$know_about_carbon_neutrality <- as.factor(df$know_about_carbon_neutrality )
df$know_about_carbon_policy <- as.factor(df$know_about_carbon_policy )

df$mainuseelec<- relevel(df$mainuseelec, ref = "0")
df$education<- relevel(df$education, ref = "uneducated")
df$location<- relevel(df$location, ref = "city")
df$marriage<- relevel(df$marriage, ref = "unmarried")
df$youth<- relevel(df$youth, ref = "0")
df$older_adults<- relevel(df$older_adults, ref = "0")
df$weather<- relevel(df$weather, ref = "0")
df$partymember<- relevel(df$partymember, ref = "0")
df$heard_about_global_warming <- relevel(df$heard_about_global_warming, ref = "no")
df$know_about_low_carbon <- factor(df$know_about_low_carbon, 
                 levels = c("never", "heard but do not know", "heard and know", "familiar"),
                 ordered = FALSE)
df$know_about_carbon_neutrality <- factor(df$know_about_carbon_neutrality, 
                 levels = c("never", "heard but do not know", "heard and know", "familiar"),
                 ordered = FALSE)
df$know_about_carbon_policy <- factor(df$know_about_carbon_policy, 
                 levels = c("never", "heard but do not know", "heard and know", "familiar"),
                 ordered = FALSE)

#（2）新建变量

df <- df %>%
  mutate(ifpollution = ifelse(aqi >= 101, 1, 0))
table(df$ifpollution)

df$married <- as.numeric(df$marriage == "married")
table(df$married )

df$age_ln <- log(df$age)
table(df$age_ln)

df$living_area_ln <- log(df$living_area)
table(df$living_area_ln)

df$is_bachelor <- ifelse(df$education %in% c("bachelor", "postgraduate"), 1, 0)
table(df$is_bachelor)

df$caruse <- ifelse(df$carusetime == 0, 1, 0)
table(df$caruse)

#(3) Table 1

# ================= 1. 连续变量/虚拟变量部分 (你的原代码) =================
table1_var <- df %>% 
  select(all_of(c("income_level", "living_area", "age", "conditioner1month", 
                  "female", "youth", "older_adults", "ifpollution", 
                  "is_bachelor", "married", "partymember", "caruse", "mainuseelec")))

table1_var <- table1_var %>%
  mutate(across(everything(), ~as.numeric(as.character(.))))

datasummary(All(table1_var) ~ N + Mean + SD + Min + Max, 
            data = table1_var,
            fmt = 3,
            title = "Table 1 - Continuous",
            output = "D:/RUC/B1WTA_new/TEST/empirical4.1_result/Table1_continuous.csv")

# ================= 2. 分类变量部分 (新增代码) =================
cat_vars <- c("location", "province", "education", "weather", 
              "heard_about_global_warming", "know_about_carbon_neutrality", 
              "know_about_carbon_policy", "know_about_low_carbon", "weekday")

cat_data <- df %>% select(all_of(cat_vars))

datasummary_skim(cat_data, type = "categorical",

                 output = "D:/RUC/B1WTA_new/TEST/empirical4.1_result/Table1_categorical.csv")

#(4) Table 2

wta_trans <- prop.table(table(df$wta_car))
wta_homeenergy <- prop.table(table(df$wta_elec))
wta_greenelec <- prop.table(table(df$wta_green))

table2_data <- rbind(wta_trans, wta_homeenergy, wta_greenelec)

write.csv(table2_data, file = "D:/RUC/B1WTA_new/TEST/empirical4.1_result/Table2.csv")

#----------------------------------------------------------------------------
# 2.回归模型
#----------------------------------------------------------------------------

#（1）打包、子集

base_vars <- "ifpollution + living_area_ln + age_ln + 
              is_bachelor + 
              location +   
              female + 
              married + 
              income_level + 
              youth + older_adults + 
              partymember +  
              province + weekday + weather"

df_car_sub <- subset(df, publictrans < 5 )# 这里我们用的是问卷问题得到的原始数据，5代表的是来往（共两次）的次数，因此一共有10次乘坐，和文章描述一致
df_elec_sub <- subset(df, conditionernumber == 1)
df_green_sub <- subset(df, energy_consume2020 > 1000)

#（2）CAR

model_car1 <- polr(as.formula(paste("wta_car ~", base_vars, " + caruse + heard_about_global_warming")), 
                data = df_car_sub, Hess = TRUE)
model_car2 <- polr(as.formula(paste("wta_car ~", base_vars, " + caruse + know_about_low_carbon")), 
                data = df_car_sub, Hess = TRUE)
model_car3 <- polr(as.formula(paste("wta_car ~", base_vars, " + caruse + know_about_carbon_neutrality")), 
                data = df_car_sub, Hess = TRUE)
model_car4 <- polr(as.formula(paste("wta_car ~", base_vars, " + caruse + know_about_carbon_policy")), 
                data = df_car_sub, Hess = TRUE)

#（3）ELEC

model_elec1 <- polr(as.formula(paste("wta_elec ~", base_vars, " + conditioner1month + heard_about_global_warming")), 
                data = df_elec_sub, Hess = TRUE)
model_elec2 <- polr(as.formula(paste("wta_elec ~", base_vars, " + conditioner1month + know_about_low_carbon")), 
                data = df_elec_sub, Hess = TRUE)
model_elec3 <- polr(as.formula(paste("wta_elec ~", base_vars, " + conditioner1month + know_about_carbon_neutrality")), 
                data = df_elec_sub, Hess = TRUE)
model_elec4 <- polr(as.formula(paste("wta_elec ~", base_vars, " + conditioner1month + know_about_carbon_policy")), 
                data = df_elec_sub, Hess = TRUE)

#（4）GREEN

model_green1 <- polr(as.formula(paste("wta_green ~", base_vars, " + mainuseelec + heard_about_global_warming")), 
                data = df_green_sub, Hess = TRUE)
model_green2 <- polr(as.formula(paste("wta_green ~", base_vars, " + mainuseelec + know_about_low_carbon")), 
                data = df_green_sub, Hess = TRUE)
model_green3 <- polr(as.formula(paste("wta_green ~", base_vars, " + mainuseelec + know_about_carbon_neutrality")), 
                data = df_green_sub, Hess = TRUE)
model_green4 <- polr(as.formula(paste("wta_green ~", base_vars, " + mainuseelec + know_about_carbon_policy")), 
                data = df_green_sub, Hess = TRUE)

#（5）模型结果表格打印
# 自动提取一组模型的 AIC 和 BIC
get_ic_lines <- function(m1, m2, m3, m4) {
  list(
    c("AIC", round(AIC(m1), 2), round(AIC(m2), 2), round(AIC(m3), 2), round(AIC(m4), 2)),
    c("BIC", round(BIC(m1), 2), round(BIC(m2), 2), round(BIC(m3), 2), round(BIC(m4), 2))
  )
}

# ================= 1. Trans Models =================
stargazer(model_car1, model_car2, model_car3, model_car4, 
          type = "text",
          model.numbers = TRUE,
          star.cutoffs = c(0.1, 0.05, 0.01),
          no.space = TRUE,
          keep = c("^location", 
                   "^heard_about_global_warming", 
                   "^know_about_low_carbon", 
                   "^know_about_carbon_neutrality", 
                   "^know_about_carbon_policy"),

          omit = c("province", "weekday", "weather", "cityanswer"),
          
          add.lines = get_ic_lines(model_car1, model_car2, model_car3, model_car4)
)

# ================= 2. Home-Energy Models =================
stargazer(model_elec1, model_elec2, model_elec3, model_elec4, 
          type = "text",
          model.numbers = TRUE,
          star.cutoffs = c(0.1, 0.05, 0.01),
          no.space = TRUE,
          keep = c("female",
                   "is_bachelor",
                   "^know_about_low_carbon", 
                   "^know_about_carbon_neutrality", 
                   "^know_about_carbon_policy"),

          omit = c("province", "weekday", "weather", "cityanswer"),
          
          add.lines = get_ic_lines(model_elec1, model_elec2, model_elec3, model_elec4)
)

# ================= 3. GreenElec Models =================
stargazer(model_green1, model_green2, model_green3, model_green4, 
          type = "text",
          model.numbers = TRUE,
          star.cutoffs = c(0.1, 0.05, 0.01),
          no.space = TRUE,
          keep = c("^location",
                   "income_level",
                   "^know_about_carbon_neutrality", 
                   "^know_about_carbon_policy"),

          omit = c("province", "weekday", "weather", "cityanswer"),
          
          add.lines = get_ic_lines(model_green1, model_green2, model_green3, model_green4)
)


#----------------------------------------------------------------------------
# 3.Brant Test
#----------------------------------------------------------------------------

# (1)CAR
brant_res <- brant(model_car1)
brant_res <- brant(model_car2)
brant_res <- brant(model_car3)
brant_res <- brant(model_car4)

# (2)ELEC
brant_res <- brant(model_elec1)
brant_res <- brant(model_elec2)
brant_res <- brant(model_elec3)
brant_res <- brant(model_elec4)

# (3)GREEN
brant_res <- brant(model_green1)
brant_res <- brant(model_green2)
brant_res <- brant(model_green3)
brant_res <- brant(model_green4)

#----------------------------------------------------------------------------
# 4.Robustness Check
#----------------------------------------------------------------------------

#（1）CAR: Replace caruse as carown 

model_car1 <- polr(as.formula(paste("wta_car ~", base_vars, " + carown + heard_about_global_warming")), 
                data = df_car_sub, Hess = TRUE)
model_car2 <- polr(as.formula(paste("wta_car ~", base_vars, " + carown + know_about_low_carbon")), 
                data = df_car_sub, Hess = TRUE)
model_car3 <- polr(as.formula(paste("wta_car ~", base_vars, " + carown + know_about_carbon_neutrality")), 
                data = df_car_sub, Hess = TRUE)
model_car4 <- polr(as.formula(paste("wta_car ~", base_vars, " + carown + know_about_carbon_policy")), 
                data = df_car_sub, Hess = TRUE)

stargazer(model_car1, model_car2, model_car3, model_car4, 
          type = "text",
          column.labels = c("Global Warming", "Low Carbon", "Neutrality", "Policy"),
          model.numbers = TRUE,
          star.cutoffs = c(0.1, 0.05, 0.01),
          no.space = TRUE,
          omit = c("province", "weekday", "weather", "weather", "cityanswer"),
          add.lines = get_ic_lines(model_car1, model_car2, model_car3, model_car4)
)

#（2）ELEC: Replace conditioner1month as conditioner1time

model_elec1 <- polr(as.formula(paste("wta_elec ~", base_vars, " + conditioner1time + heard_about_global_warming")), 
                data = df_elec_sub, Hess = TRUE)
model_elec2 <- polr(as.formula(paste("wta_elec ~", base_vars, " + conditioner1time + know_about_low_carbon")), 
                data = df_elec_sub, Hess = TRUE)
model_elec3 <- polr(as.formula(paste("wta_elec ~", base_vars, " + conditioner1time + know_about_carbon_neutrality")), 
                data = df_elec_sub, Hess = TRUE)
model_elec4 <- polr(as.formula(paste("wta_elec ~", base_vars, " + conditioner1time + know_about_carbon_policy")), 
                data = df_elec_sub, Hess = TRUE)

# ELEC Models
stargazer(model_elec1, model_elec2, model_elec3, model_elec4, 
          type = "text",
          column.labels = c("Global Warming", "Low Carbon", "Neutrality", "Policy"),
          model.numbers = TRUE,
          star.cutoffs = c(0.1, 0.05, 0.01),
          no.space = TRUE,
          omit = c("province", "weekday", "weather", "weather", "cityanswer"),
          add.lines = get_ic_lines(model_elec1, model_elec2, model_elec3, model_elec4)
)

#（3）CAR/ELEC/GREEN: Replace province as cityanswer

robust_vars <- "ifpollution + living_area_ln + age_ln + 
              is_bachelor + 
              location +   
              female + 
              married + 
              income_level + 
              youth + older_adults + 
              partymember +  
              cityanswer + weekday + weather"

model_car1 <- polr(as.formula(paste("wta_car ~", robust_vars, " + caruse + heard_about_global_warming")), 
                data = df_car_sub, Hess = TRUE)
model_car2 <- polr(as.formula(paste("wta_car ~", robust_vars, " + caruse + know_about_low_carbon")), 
                data = df_car_sub, Hess = TRUE)
model_car3 <- polr(as.formula(paste("wta_car ~", robust_vars, " + caruse + know_about_carbon_neutrality")), 
                data = df_car_sub, Hess = TRUE)
model_car4 <- polr(as.formula(paste("wta_car ~", robust_vars, " + caruse + know_about_carbon_policy")), 
                data = df_car_sub, Hess = TRUE)

model_elec1 <- polr(as.formula(paste("wta_elec ~", robust_vars, " + conditioner1month + heard_about_global_warming")), 
                data = df_elec_sub, Hess = TRUE)
model_elec2 <- polr(as.formula(paste("wta_elec ~", robust_vars, " + conditioner1month + know_about_low_carbon")), 
                data = df_elec_sub, Hess = TRUE)
model_elec3 <- polr(as.formula(paste("wta_elec ~", robust_vars, " + conditioner1month + know_about_carbon_neutrality")), 
                data = df_elec_sub, Hess = TRUE)
model_elec4 <- polr(as.formula(paste("wta_elec ~", robust_vars, " + conditioner1month + know_about_carbon_policy")), 
                data = df_elec_sub, Hess = TRUE)

model_green1 <- polr(as.formula(paste("wta_green ~", robust_vars, " + mainuseelec + heard_about_global_warming")), 
                data = df_green_sub, Hess = TRUE)
model_green2 <- polr(as.formula(paste("wta_green ~", robust_vars, " + mainuseelec + know_about_low_carbon")), 
                data = df_green_sub, Hess = TRUE)
model_green3 <- polr(as.formula(paste("wta_green ~", robust_vars, " + mainuseelec + know_about_carbon_neutrality")), 
                data = df_green_sub, Hess = TRUE)
model_green4 <- polr(as.formula(paste("wta_green ~", robust_vars, " + mainuseelec + know_about_carbon_policy")), 
                data = df_green_sub, Hess = TRUE)


# 1. CAR Models
stargazer(model_car1, model_car2, model_car3, model_car4, 
          type = "text",
          column.labels = c("Global Warming", "Low Carbon", "Neutrality", "Policy"),
          model.numbers = TRUE,
          star.cutoffs = c(0.1, 0.05, 0.01),
          no.space = TRUE,
          omit = c("province", "weekday", "weather", "weather", "cityanswer"),
          add.lines = get_ic_lines(model_car1, model_car2, model_car3, model_car4)
)

# 2. ELEC Models
stargazer(model_elec1, model_elec2, model_elec3, model_elec4, 
          type = "text",
          column.labels = c("Global Warming", "Low Carbon", "Neutrality", "Policy"),
          model.numbers = TRUE,
          star.cutoffs = c(0.1, 0.05, 0.01),
          no.space = TRUE,
          omit = c("province", "weekday", "weather", "weather", "cityanswer"),
          add.lines = get_ic_lines(model_elec1, model_elec2, model_elec3, model_elec4)
)

# 3. GREEN Models
stargazer(model_green1, model_green2, model_green3, model_green4, 
          type = "text",
          column.labels = c("Global Warming", "Low carbon", "Neutrality", "Policy"),
          model.numbers = TRUE,
          star.cutoffs = c(0.1, 0.05, 0.01),
          no.space = TRUE,
          omit = c("province", "weekday", "weather", "weather", "cityanswer"),
          add.lines = get_ic_lines(model_green1, model_green2, model_green3, model_green4)
)

