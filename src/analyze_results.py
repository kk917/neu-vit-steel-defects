import sys, os
import pandas as pd, numpy as np
from scipy import stats
# usage: python src/analyze_results.py [results_dir]
os.chdir(sys.argv[1] if len(sys.argv)>1 else 'results/full_run')
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
e1=pd.read_csv('exp1_learning_curves.csv')
# average seeds within fold
f=e1.groupby(['method','k','fold'])[['acc','macro_f1','nll','ece','conf_err']].mean().reset_index()
M=['frozen_clip+LR','frozen_clip+LR_tuned','partial_finetune_clip','finetune_clip','finetune_imagenet_vit']
def ms(col):
    g=f.groupby(['method','k'])[col].agg(['mean','std']).reset_index()
    t=g.pivot(index='k',columns='method',values='mean')[M]; s=g.pivot(index='k',columns='method',values='std')[M]
    return t,s
for col in ['acc','nll','ece','conf_err']:
    t,s=ms(col); print('\n==',col,'mean over 5 folds'); print((t*(100 if col in('acc','conf_err') else 1)).round(3).to_string())
    if col=='acc': print('std over folds (pts)'); print((s*100).round(2).to_string())
def paired(a,b,col='acc'):
    rows=[]
    for k in sorted(f.k.unique()):
        x=f[(f.method==a)&(f.k==k)].sort_values('fold')[col].values; y=f[(f.method==b)&(f.k==k)].sort_values('fold')[col].values
        d=(x-y)*100; m=d.mean(); se=d.std(ddof=1)/np.sqrt(len(d)) if d.std()>0 else 0
        h=stats.t.ppf(.975,len(d)-1)*se; rows.append((k,round(m,2),round(m-h,2),round(m+h,2)))
    return pd.DataFrame(rows,columns=['k','diff_pts','lo95','hi95'])
for a,b in [('finetune_clip','frozen_clip+LR'),('finetune_clip','frozen_clip+LR_tuned'),('finetune_clip','finetune_imagenet_vit'),('partial_finetune_clip','finetune_clip')]:
    print('\npaired',a,'-',b); print(paired(a,b).to_string(index=False))
# labels needed
t,_=ms('acc')
for thr in (0.95,0.99):
    print('labels/class to reach',thr,{m:next((k for k in t.index if t.loc[k,m]>=thr),None) for m in M})
# per-class k=1 recall? use predictions
import pickle
P=pickle.load(open('predictions.pkl','rb')); y=np.repeat(np.arange(6),60)
names=['crazing','inclusion','patches','pitting','rolled-in','scratches']
for k in (1,2,5):
  for m in ['frozen_clip+LR','finetune_clip']:
    cm=np.zeros((6,6),int)
    for key,pr in P.items():
        p=key.split('|')
        if p[0]=='e1' and p[1]==m and int(p[2])==k:
            for a,b in zip(y,pr): cm[a,b]+=1
    print(k,m,dict(zip(names,(cm.diagonal()/cm.sum(1)).round(2))))
print('\nEXP2'); e2=pd.read_csv('exp2_rare_defect.csv')
g=e2.groupby(['k_rare','method'])[['rare_recall','rare_precision','acc','macro_f1']].mean().round(3); print(g.to_string())
print(e2.groupby(['rare_class','k_rare','method'])['rare_recall'].mean().unstack().round(2).to_string())
print('\nEXP3'); e3=pd.read_csv('exp3_open_set.csv')
print(e3.groupby('method')[['auroc','fpr95','acc_known']].agg(['mean','std']).round(3).to_string())
print(e3.pivot_table(index='novel_class',columns='method',values='auroc').round(3).to_string())
print('\nEXP4'); e4=pd.read_csv('exp4_robustness.csv')
for k in sorted(e4.k.unique()):
    print('k=',k); print(e4[e4.k==k].pivot_table(index='shift',columns='method',values='acc').round(3).to_string())
print('\nEXP5'); e5=pd.read_csv('exp5_sensitivity.csv')
print(e5.groupby(['k','method','param'])[['acc','nll','ece']].mean().round(3).to_string())
print('\nZS'); print(pd.read_csv('exp6_zero_shot.csv').round(3).to_string()); print(open('exp6_zero_shot_confusion.csv').read())
z=np.load('nn_similarity.npz'); s=np.concatenate([z[f'nn_sim_{i}'] for i in range(5)]); print('nn sim quantiles',np.quantile(s,[0,.05,.25,.5,.75,1]).round(4))
