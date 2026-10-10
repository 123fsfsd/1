# 数据分析技能作品集配图

三张原创生成配图分别展示随机森林、回归模型与 Stata 分析工作流。
这些配图属于概念示意，图中的数值、曲线和结果表并非真实模型输出，不能作为已完成项目或实证结果的证明。

- `random_forest_portfolio.png`：随机森林概念配图。
- `regression_portfolio.png`：回归模型与诊断概念配图。
- `stata_workflow_portfolio.png`：Stata 工作流概念配图。
- `stata_portfolio.do`：使用实际数据生成 OLS 系数图和诊断图的脚本模板。

## 运行 Stata 脚本

脚本以 Stata 17 为兼容目标。本环境未安装 Stata，脚本尚未在 Stata 中验证运行。
在 Stata 中切换到你的数据所在目录，修改脚本开头的 `datafile`、`y` 和 `xvars`，再运行脚本。
`xvars` 应填写实际存在的数值变量名，用空格分隔。
脚本会清除当前内存数据；请先保存正在使用的数据。
输出位于当前工作目录下的 `stata_output`，同名输出会被覆盖。
首次生成系数图需要联网从 SSC 安装 `coefplot`。
该脚本不包含随机森林模型的训练代码。

## 报告式证明配图

- `regression_evidence_demo.png`：包含回归结果表、系数置信区间、残差诊断和 Stata 核心代码。
- `random_forest_evidence_demo.png`：包含测试集指标、特征重要性、预测效果和 Python 核心代码。
- `evidence_collage_demo.png`：两页报告式拼图，适合放在个人作品集封面或社交平台。

以上三张报告图均明确标注“演示数据 / DEMO”，不能冒充真实项目结果。替换成自己的数据并重新运行后，才可作为真实项目材料。
