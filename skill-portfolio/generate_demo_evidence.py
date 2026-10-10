from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

OUT = Path(__file__).resolve().parent
TMP = OUT / '_charts'
TMP.mkdir(exist_ok=True)
FONT_REG = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
FONT_MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'

def font(path, size):
    return ImageFont.truetype(path, size)

def draw_text(draw, xy, text, size=30, fill='#172033', bold=False, anchor=None):
    draw.text(xy, text, font=font(FONT_BOLD if bold else FONT_REG, size), fill=fill, anchor=anchor)

def chart_paths(seed=2026):
    rng = np.random.default_rng(seed)
    n = 260
    X = rng.normal(size=(n, 5))
    names = ['Age', 'Income', 'Education', 'Experience', 'Training']
    beta = np.array([0.62, 1.18, 0.42, -0.55, 0.78])
    y = 2.1 + X @ beta + rng.normal(0, 1.05, n)
    X1 = np.column_stack([np.ones(n), X])
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    resid = y - X1 @ b
    s2 = (resid @ resid) / (n - X1.shape[1])
    cov = s2 * np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    t = b / se
    # Normal approximation is sufficient for a visual demo table.
    p = np.array([math.erfc(abs(v) / np.sqrt(2)) for v in t])
    r2 = 1 - np.sum(resid**2) / np.sum((y-y.mean())**2)
    adj = 1 - (1-r2)*(n-1)/(n-X1.shape[1])

    plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'axes.unicode_minus': False})
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
    ax.scatter(X1 @ b, resid, s=16, alpha=.55, color='#2b6cb0', edgecolors='none')
    ax.axhline(0, color='#d1495b', lw=1.5)
    ax.set_xlabel('Fitted values'); ax.set_ylabel('Residuals'); ax.set_title('Residuals vs fitted values')
    ax.grid(alpha=.18); fig.tight_layout(); residual_path = TMP/'residuals.png'; fig.savefig(residual_path, bbox_inches='tight'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.7, 5.2), dpi=160)
    co = b[1:]; lo = co - 1.96*se[1:]; hi = co + 1.96*se[1:]
    yidx = np.arange(len(names))
    ax.errorbar(co, yidx, xerr=[co-lo, hi-co], fmt='o', color='#155e75', ecolor='#0ea5a4', capsize=4, lw=2)
    ax.axvline(0, color='#d1495b', ls='--', lw=1.2)
    ax.set_yticks(yidx, names); ax.invert_yaxis(); ax.set_xlabel('Coefficient with 95% CI'); ax.set_title('OLS coefficient plot')
    ax.grid(axis='x', alpha=.18); fig.tight_layout(); coef_path = TMP/'coefficients.png'; fig.savefig(coef_path, bbox_inches='tight'); plt.close(fig)

    rf = RandomForestRegressor(n_estimators=240, max_features=0.8, min_samples_leaf=3, random_state=2026, oob_score=True, n_jobs=-1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=2026)
    rf.fit(Xtr, ytr); pred = rf.predict(Xte)
    rf_r2 = r2_score(yte, pred); rmse = mean_squared_error(yte, pred)**.5
    fig, ax = plt.subplots(figsize=(8.7, 5.2), dpi=160)
    order = np.argsort(rf.feature_importances_)[::-1]
    ax.barh(np.array(names)[order][::-1], rf.feature_importances_[order][::-1], color='#2563eb')
    ax.set_xlabel('Importance'); ax.set_title('Random forest feature importance'); ax.grid(axis='x', alpha=.18); fig.tight_layout(); rfimp_path = TMP/'rf_importance.png'; fig.savefig(rfimp_path, bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.7, 5.2), dpi=160)
    ax.scatter(yte, pred, s=24, color='#f97316', alpha=.7, edgecolors='white', linewidth=.3)
    mn, mx = min(yte.min(), pred.min()), max(yte.max(), pred.max()); ax.plot([mn,mx],[mn,mx], color='#155e75', ls='--')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted'); ax.set_title('Test set: actual vs predicted'); ax.grid(alpha=.18); fig.tight_layout(); rfpred_path = TMP/'rf_pred.png'; fig.savefig(rfpred_path, bbox_inches='tight'); plt.close(fig)
    return {'names':names, 'b':b, 'se':se, 't':t, 'p':p, 'r2':r2, 'adj':adj, 'rmse_ols':float(np.sqrt(np.mean(resid**2))), 'n':n, 'rf_r2':rf_r2, 'rmse':rmse, 'oob':rf.oob_score_, 'coef':coef_path, 'resid':residual_path, 'rfimp':rfimp_path, 'rfpred':rfpred_path}

def base_page(title, subtitle, accent='#155e75'):
    W,H=1600,2200
    im=Image.new('RGB',(W,H),'#eef1f5'); shadow=Image.new('RGBA',(W,H),(0,0,0,0)); sd=ImageDraw.Draw(shadow)
    sd.rounded_rectangle((92,70,W-74,H-64), radius=14, fill=(0,0,0,35)); shadow=shadow.filter(ImageFilter.GaussianBlur(18)); im=Image.alpha_composite(im.convert('RGBA'),shadow)
    d=ImageDraw.Draw(im); d.rounded_rectangle((72,48,W-94,H-88), radius=12, fill='white', outline='#d7dde6', width=2)
    d.rectangle((72,48,W-94,190), fill=accent)
    draw_text(d,(116,88),title,48,'white',True)
    draw_text(d,(116,146),subtitle,24,'#d9f3f1')
    d.rounded_rectangle((1120,87,1450,151), radius=12, fill='#fbbf24')
    draw_text(d,(1285,119),'演示数据 / DEMO',23,'#3b2f0b',True,anchor='mm')
    return im.convert('RGB')

def add_chart(page, path, box):
    chart=Image.open(path).convert('RGB'); chart.thumbnail((box[2]-box[0],box[3]-box[1]))
    x=box[0]+(box[2]-box[0]-chart.width)//2; y=box[1]+(box[3]-box[1]-chart.height)//2
    page.paste(chart,(x,y))

def regression_page(stats):
    page=base_page('回归分析证明页','OLS 回归｜稳健标准误｜模型诊断', '#155e75'); d=ImageDraw.Draw(page)
    draw_text(d,(122,235),'研究问题：哪些因素与结果变量相关？',30,'#172033',True)
    draw_text(d,(122,285),'模拟样本 n=260；变量名和数值仅用于展示报告结构。',22,'#64748b')
    x0,y0=122,350; row=63; cols=[0,260,515,735,950,1190]
    headers=['变量','系数','标准误','t 值','P>|t|','显著性']
    for i,h in enumerate(headers): draw_text(d,(x0+cols[i],y0),h,23,'#ffffff',True)
    d.rectangle((x0,y0-12,x0+1320,y0+43), fill='#1f6f8b')
    for i,h in enumerate(headers): draw_text(d,(x0+cols[i],y0),h,23,'#ffffff',True)
    names=['年龄','收入','教育年限','工作经验','培训次数']; vals=stats['b'][1:]; ses=stats['se'][1:]; ts=stats['t'][1:]; ps=stats['p'][1:]
    for j,name in enumerate(names):
        yy=y0+62+j*row
        if j%2==0: d.rectangle((x0,yy-10,x0+1320,yy+43), fill='#f4f8fa')
        arr=[name,f'{vals[j]:.3f}',f'{ses[j]:.3f}',f'{ts[j]:.2f}',f'{ps[j]:.3f}','***' if ps[j]<.01 else '**' if ps[j]<.05 else '*']
        for i,v in enumerate(arr): draw_text(d,(x0+cols[i],yy),v,22,'#172033',i==0)
    liney=y0+62+len(names)*row+12; d.line((x0,liney,x0+1320,liney),fill='#cbd5e1',width=2)
    draw_text(d,(x0,liney+25),f"R² = {stats['r2']:.3f}    调整 R² = {stats['adj']:.3f}    Root MSE = {stats['rmse_ols']:.3f}",22,'#475569')
    add_chart(page,stats['coef'],(130,790,800,1270)); add_chart(page,stats['resid'],(840,790,1510,1270))
    d.rounded_rectangle((122,1320,1478,1940), radius=10, fill='#0f172a')
    draw_text(d,(160,1360),'Stata do-file（核心代码）',27,'#7dd3fc',True)
    code=['clear all','set seed 2026','use "your_data.dta", clear','reg outcome x1 x2 x3 x4 x5, vce(robust)','estat vif','estat hettest','predict fitted, xb','predict resid, resid','rvfplot, yline(0)','coefplot, xline(0)']
    for i,line in enumerate(code):
        draw_text(d,(165,1415+i*43),f'{i+1:02d}  {line}',22,'#e2e8f0')
    draw_text(d,(122,2025),'说明：本页为模拟数据演示；真实证明应替换为你实际运行产生的输出。',22,'#b45309',True)
    return page

def rf_page(stats):
    page=base_page('随机森林分析证明页','Random Forest｜特征重要性｜测试集评估', '#1d4ed8'); d=ImageDraw.Draw(page)
    draw_text(d,(122,235),'任务：用非线性集成模型进行预测，并解释变量贡献。',30,'#172033',True)
    draw_text(d,(122,285),'模拟数据；训练集 75%，测试集 25%，树数量 240。',22,'#64748b')
    cards=[('测试集 R²',f"{stats['rf_r2']:.3f}"),('RMSE',f"{stats['rmse']:.3f}"),('OOB R²',f"{stats['oob']:.3f}")]
    for i,(a,b) in enumerate(cards):
        xx=122+i*430; d.rounded_rectangle((xx,350,xx+380,490),radius=12,fill='#eff6ff',outline='#bfdbfe',width=2); draw_text(d,(xx+28,380),a,24,'#1e40af',True); draw_text(d,(xx+28,420),b,42,'#172033',True)
    add_chart(page,stats['rfimp'],(120,550,800,1030)); add_chart(page,stats['rfpred'],(840,550,1510,1030))
    d.rounded_rectangle((122,1100,1478,1940), radius=10, fill='#0f172a')
    draw_text(d,(160,1140),'Python / sklearn 核心代码',27,'#93c5fd',True)
    code=['from sklearn.ensemble import RandomForestRegressor','rf = RandomForestRegressor(n_estimators=240,','    max_features=0.8, min_samples_leaf=3,','    oob_score=True, random_state=2026)','rf.fit(X_train, y_train)','pred = rf.predict(X_test)','r2_score(y_test, pred)','rf.feature_importances_']
    for i,line in enumerate(code): draw_text(d,(165,1195+i*52),f'{i+1:02d}  {line}',22,'#e2e8f0')
    draw_text(d,(122,2025),'说明：本页为模拟数据演示；真实证明应替换为你实际训练与评估的结果。',22,'#b45309',True)
    return page

def collage(reg, rf):
    W,H=1900,1400
    canvas=Image.new('RGB',(W,H),'#dcd8e9')
    canvas=canvas.filter(ImageFilter.GaussianBlur(.2))
    def place(img, xy, angle):
        thumb=img.copy(); thumb.thumbnail((720,1050)); thumb=thumb.rotate(angle,expand=True,resample=Image.Resampling.BICUBIC)
        x,y=xy; sh=Image.new('RGBA',(thumb.width+35,thumb.height+35),(0,0,0,0)); sd=ImageDraw.Draw(sh); sd.rectangle((18,18,thumb.width+20,thumb.height+20),fill=(0,0,0,45)); sh=sh.filter(ImageFilter.GaussianBlur(12)); canvas.paste(sh,(x-18,y-18),sh); canvas.paste(thumb,(x,y))
    place(reg,(120,220),-4); place(rf,(1060,170),5)
    d=ImageDraw.Draw(canvas); draw_text(d,(950,62),'DATA ANALYSIS PORTFOLIO',32,'#43355b',True,anchor='mm'); draw_text(d,(950,105),'原创版式｜模拟数据演示',20,'#6b5e7c',anchor='mm')
    return canvas

stats=chart_paths()
reg=regression_page(stats); rf=rf_page(stats)
reg.save(OUT/'regression_evidence_demo.png',quality=95); rf.save(OUT/'random_forest_evidence_demo.png',quality=95)
collage(reg,rf).save(OUT/'evidence_collage_demo.png',quality=95)
print('created', OUT/'regression_evidence_demo.png', OUT/'random_forest_evidence_demo.png', OUT/'evidence_collage_demo.png')
