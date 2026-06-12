# Project Handoff Notes

更新时间：2026-06-12

这个项目的目标是把原来分散的实证代码整理成一个可复现的期刊提交包。当前项目根目录为：

```text
D:\RUC\revision_package
```

## 环境约定

- R 默认优先用于数据处理、统计、计量和绘图。
- Python 主要用于 Section 4.2 的 XGBoost、Optuna、模拟流程。
- R 包路径：`D:\R-4.5.2\Packages`
- Python 包路径：`D:\Python\Packages`
- Python 脚本会自动把 `D:\Python\Packages` 加入 `sys.path`。
- R 脚本通过 `R/00_setup.R` 设置 `.libPaths("D:/R-4.5.2/Packages")`。

## 主要入口

历史来源文件已经从项目根目录移到：

```text
archive/historical_sources
```

这些文件只用于追溯原始代码来源，不参与当前复现流程。正式入口只使用
`run_all.R`、`R/` 下的模块化脚本，以及 `python/empirical_4_2.py`。

### Section 4.1 及主表、附录表

运行：

```powershell
& "D:\R-4.5.2\bin\Rscript.exe" "D:\RUC\revision_package\run_all.R"
```

主要输出：

- `results/empirical4.1/figures/F_B.1.png`
- `results/empirical4.1/figures/F_B.2.png`
- `results/empirical4.1/tables/Table_3_main_transport.*`
- `results/empirical4.1/tables/Table_4_main_home_energy.*`
- `results/empirical4.1/tables/Table_5_main_green_electricity.*`
- `results/empirical4.1/tables/Table_F.7_transport_robustness.*`
- `results/empirical4.1/tables/Table_F.8_home_energy_conditioner_time.*`
- `results/empirical4.1/tables/Table_F.9_green_electricity_importance.*`
- `results/empirical4.1/tables/Table_G.10_unweighted_balance.*`
- `results/empirical4.1/tables/Table_G.11_weighted_balance_test.*`
- `results/empirical4.1/tables/Table_G.12_weighted_transport.*`
- `results/empirical4.1/tables/Table_G.13_weighted_home_energy.*`
- `results/empirical4.1/tables/Table_G.14_weighted_green_electricity.*`

该流程会生成 Section 4.2 需要的加权数据：

```text
results/energy_wta_with_post_weights.csv
```

### Section 4.2 默认可复现流程

运行：

```powershell
& "D:\Python\python.exe" "D:\RUC\revision_package\python\empirical_4_2.py" --root "D:\RUC\revision_package"
```

默认输出目录：

```text
results/empirical4.2
results/empirical4.3
```

默认设置：

- Pref Alt XGBoost：Optuna 50 trials，`TPESampler(seed=120)`
- WTA XGBoost：Optuna 100 trials，当前整理后也使用 `TPESampler(seed=120)`
- Random baseline：1000 次 Monte Carlo
- Section 4.3：200 次知识增长模拟

## 当前最重要的复现口径：只固定 WTA 旧参数

用户明确要求：第一次 Pref 模型训练不直接代入旧参数，因为它在旧代码里本来就固定了 Optuna 随机性；只需要对第二次 WTA 模型训练使用原表中的旧参数。

因此当前推荐命令是：

```powershell
& "D:\Python\python.exe" "D:\RUC\revision_package\python\empirical_4_2.py" --root "D:\RUC\revision_package" --use-legacy-wta-params --output-subdir "empirical4.2_legacy_wta" --skip-simulation
```

这条命令的含义：

- Pref Alt classifiers：仍然走带 seed 的 Optuna 搜索。
- WTA regressors：不再搜索，直接使用论文原表 D.2 中记录的 WTA 参数。
- 输出目录：`results/empirical4.2_legacy_wta`
- 跳过 Section 4.3，因为当前讨论的旧参数问题只影响 WTA 后续的配额/预算模拟，不影响 4.3 的 Pref Alt/Logit 主流程。

当前已经生成的 WTA-only 结果目录：

```text
results/empirical4.2_legacy_wta
```

其中关键文件包括：

- `Table_D.2_xgboost_hyperparameters.csv`
- `Table_D.2_xgboost_hyperparameters.tex`
- `Table_D.3_assignment_outcomes.csv`
- `Table_D.4_quota_accept_cost.csv`
- `Table_D.5_budget_participation.csv`
- `Figure_3_prediction_accuracy.png`
- `Figure_4_assignment_outcomes.png`
- `Figure_5_quota_compensation.png`
- `Figure_6_budget_participation.png`
- `legacy_params_used.txt`

注意：如果只查看 `results/empirical4.2`，不会看到 WTA-only 的更新，因为为了避免覆盖默认可复现结果，WTA-only 结果被写到了单独目录。

## WTA 旧参数

当前脚本中的 WTA 旧参数来自用户提供的 Original 截图：

| Model | n_estimators | max_depth | learning_rate | gamma | reg_lambda |
|---|---:|---:|---:|---:|---:|
| Car_Demos_Reg | 290 | 6 | 0.048 | 1.642 | 3.851 |
| Elec_Demos_Reg | 190 | 4 | 0.047 | 1.920 | 4.654 |
| Green_Demos_Reg | 50 | 6 | 0.026 | 4.153 | 1.626 |
| Car_All_Reg | 370 | 4 | 0.091 | 3.251 | 0.254 |
| Elec_All_Reg | 170 | 3 | 0.062 | 1.317 | 4.638 |
| Green_All_Reg | 270 | 3 | 0.015 | 4.714 | 0.216 |

代码位置：

```text
python/empirical_4_2.py
```

相关开关：

```text
--use-legacy-wta-params
```

## 为什么固定旧 WTA 参数后仍可能和旧结果略有差异

这是当前最需要接力者理解的一点。

旧论文表格只保存了三位小数的超参数，不是 Optuna 当时返回的完整精度。例如表里写 `learning_rate = 0.048`，真实搜索结果可能是 `0.0476...`。WTA 预测后又会执行：

```python
np.floor(model.predict(...)).astype(int)
```

因此只要预测值在整数边界附近，极小的数值差异就可能导致取整后从 3 变成 2，进而影响 Section 4.2.2 和 4.2.3 的配额/预算模拟结果。

这也解释了为什么：

- Pref 模型相关结果变化不明显或不变：Pref 模型本身有 seed，且输出是类别决策/概率排序。
- R 语言模型结果不变：R 的 logit 和 ordered logit 独立于 Python 的 XGBoost WTA 回归器。
- 差异集中在第二阶段 WTA 模型之后：WTA 预测值经过 `floor()` 后进入补偿成本和预算模拟。

如果要完全复现旧结果，需要找到当时保存的以下任一内容：

- 当时 Optuna `study.best_params` 的完整精度数值；
- 当时训练好的 XGBoost WTA 模型 pickle/joblib；
- 当时输出的 `results_xgb_all`、`results_xgb_demos` 或 WTA 预测结果文件。

仅靠论文截图中的三位小数参数，无法保证 bit-for-bit 或逐表完全一致。

## 当前脚本状态

`python/empirical_4_2.py` 当前应满足：

- 默认运行：Pref Alt 和 WTA 都走带 seed 的可复现 Optuna 搜索。
- 加 `--use-legacy-wta-params`：只有 WTA regressors 使用旧参数，Pref Alt 仍然搜索。
- 加 `--output-subdir`：Section 4.2 输出写到 `results/<output-subdir>`。
- 加 `--skip-simulation`：跳过 Section 4.3。

R 辅助脚本已经支持自定义结果子目录：

- `R/07_logit_4_2.R`
- `R/08_ologit_wta_4_2.R`
- `R/09_simulated_predictions_4_3.R`

## 容易混淆的目录

- `results/empirical4.2`：默认带 seed 的完整流程结果。
- `results/empirical4.2_legacy_wta`：只固定 WTA 旧参数的结果，当前用户希望看的就是这个。
- `results/empirical4.3`：默认 Section 4.3 知识增长模拟结果。
- `results/empirical4.1/figures`、`results/empirical4.1/tables`、`results/empirical4.1/logs`：Section 4.1 的正式输出。

已经清理掉的测试目录：

- `results/empirical4.2_legacy_params`
- `results/empirical4.2_legacy_params_check`
- `results/empirical4.2_legacy_wta_smoke`
- `results/empirical4.3_legacy_wta_smoke`
- `python/__pycache__`

不要再使用 `empirical4.2_legacy_params` 口径；用户后来明确要求只固定 WTA，不固定 Pref Alt。

## 归档历史来源

以下文件已经移入 `archive/historical_sources`：

- `4.1.R`
- `4.1_draw.R`
- `4.1_robust.R`
- `4.1_weighted.R`
- `4.1_weighted_table.R`
- `energy_wta.csv`
- `参考图1.jpg`
- `实证4.2部分的模块1.txt`

当前正式原始数据位置是 `data/raw/energy_wta.csv`，不要从根目录读取旧数据副本。

## 建议给论文中的说明口径

可以写成类似：

> The preference-alternative XGBoost classifiers were optimized using Optuna with a fixed TPESampler seed. In the original WTA XGBoost regressors, the Optuna sampler seed was not fixed. To reproduce the downstream policy simulations based on the originally reported WTA specification, we reran the WTA stage using the hyperparameters reported in Table D.2 while keeping the preference-alternative stage unchanged.

如果需要更严谨，可以补一句：

> Because the originally reported hyperparameters were rounded to three decimals, minor discrepancies may remain in simulations that depend on floored WTA predictions.

## 下一步建议

1. 如果用户需要“看到更新”，先确认她看的是否是 `results/empirical4.2_legacy_wta`。
2. 如果她希望默认目录也显示 WTA-only 结果，可以重新运行同一命令但把 `--output-subdir` 改成 `empirical4.2`。这会覆盖默认结果，不建议在没有明确确认前这么做。
3. 如果需要把 `legacy_wta` 结果复制到论文最终交付目录，应明确目标目录后再复制，避免混淆默认可复现结果。
4. 不要再使用“Pref + WTA 都固定”的旧测试口径；只使用 `--use-legacy-wta-params`。
