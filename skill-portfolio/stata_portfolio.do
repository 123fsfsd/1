version 17.0
clear all
set more off
set linesize 120

* ====== 修改这里：你的数据文件和变量名 ======
local datafile "your_data.dta"
local y        "outcome"
local xvars    "x1 x2 x3 x4 x5"
local outdir   "stata_output"

capture mkdir "`outdir'"
use "`datafile'", clear
egen __portfolio_nmissing = rowmiss(`y' `xvars')
keep if __portfolio_nmissing == 0
drop __portfolio_nmissing

* 1) 描述统计（可作为作品集表格素材）
summarize `y' `xvars', detail

* 2) OLS 回归 + 稳健标准误
regress `y' `xvars', vce(robust)
estimates store ols_robust

* 3) 多重共线性与异方差诊断
estat vif
quietly regress `y' `xvars'
estat hettest
estat imtest, white
estimates restore ols_robust

* 4) 预测值与残差
predict fitted, xb
predict resid, resid

* 5) 残差-拟合值图
rvfplot, yline(0) ///
    title("Residuals vs fitted values") ///
    name(g_rvf, replace)
graph export "`outdir'/residuals_vs_fitted.png", replace width(2000)

* 6) 正态 Q-Q 图
qnorm resid, ///
    title("Normal Q-Q plot of residuals") ///
    name(g_qq, replace)
graph export "`outdir'/residuals_qq.png", replace width(2000)

* 7) 系数图（第一次运行会安装公开的 coefplot 命令）
capture which coefplot
if _rc {
    ssc install coefplot, replace
}
coefplot ols_robust, keep(`xvars') xline(0, lpattern(dash) lcolor(gs8)) ///
    ciopts(recast(rcap) lcolor(navy)) ///
    msymbol(circle) mcolor(navy) ///
    title("OLS coefficients with 95% confidence intervals") ///
    name(g_coef, replace)
graph export "`outdir'/ols_coefficients.png", replace width(2000)

* 8) 结果与数据快照，便于复现记录
log using "`outdir'/analysis.log", replace text
regress `y' `xvars', vce(robust)
estat vif
quietly regress `y' `xvars'
estat hettest
estimates restore ols_robust
log close

* ====== 可选：保存清理后的分析数据 ======
save "`outdir'/analysis_ready.dta", replace
