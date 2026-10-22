import sys, os, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
SURF='#fcfcfb'; INK='#0b0b0b'; INK2='#52514e'; GRID='#e4e3df'
# usage: python src/make_figures.py [results_dir] [output_dir]
RES=sys.argv[1] if len(sys.argv)>1 else 'results/full_run'; OUTD=os.path.abspath(sys.argv[2] if len(sys.argv)>2 else 'figures'); os.makedirs(OUTD,exist_ok=True); os.chdir(RES)
C=dict(ft='#2a78d6', part='#eb6834', tuned='#1baf7a', imn='#eda100', frozen='#8a8985', ftw='#184f95', aug='#184f95', bal='#1baf7a')
plt.rcParams.update({'font.size':9,'axes.edgecolor':GRID,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.facecolor':SURF,'figure.facecolor':SURF,'axes.spines.top':False,'axes.spines.right':False,'grid.color':GRID,'grid.linewidth':.6,'legend.frameon':False})
def clean(ax): ax.grid(axis='y'); ax.set_axisbelow(True)
e1=pd.read_csv('exp1_learning_curves.csv'); f=e1.groupby(['method','k','fold'])[['acc','nll']].mean().reset_index()
M=[('finetune_clip','Full fine-tune (CLIP)',C['ft'],'o','-'),('partial_finetune_clip','Partial fine-tune (CLIP)',C['part'],'s','-'),
   ('finetune_imagenet_vit','Full fine-tune (ImageNet ViT)',C['imn'],'^','-'),('frozen_clip+LR_tuned','Frozen CLIP + LR (tuned C)',C['tuned'],'D','-'),
   ('frozen_clip+LR','Frozen CLIP + LR (C=10)',C['frozen'],'v','--')]
ks=sorted(f.k.unique()); x=np.arange(len(ks))
fig,ax=plt.subplots(1,2,figsize=(7.2,3.2))
for m,lab,c,mk,ls in M:
    g=f[f.method==m].groupby('k')['acc'].agg(['mean','std']).reindex(ks)*[100,100]
    ax[0].plot(x,g['mean'],ls,color=c,marker=mk,ms=4.5,lw=1.6,label=lab); ax[0].fill_between(x,g['mean']-g['std'],g['mean']+g['std'],color=c,alpha=.10,lw=0)
    n=f[f.method==m].groupby('k')['nll'].mean().reindex(ks); ax[1].plot(x,n,ls,color=c,marker=mk,ms=4.5,lw=1.6)
for a in ax: a.set_xticks(x); a.set_xticklabels(ks); a.set_xlabel('Labelled images per class (k)'); clean(a)
ax[0].set_ylabel('Test accuracy (%)'); ax[0].set_ylim(66,100.5); ax[0].set_title('(a) Accuracy, mean ± SD over 5 folds',fontsize=9,color=INK,loc='left')
ax[1].set_yscale('log'); ax[1].set_ylabel('Log-loss (lower is better)'); ax[1].set_title('(b) Log-loss, log scale',fontsize=9,color=INK,loc='left')
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=3,fontsize=7.5,bbox_to_anchor=(0.5,-0.02))
plt.tight_layout(rect=(0,0.13,1,1)); plt.savefig(OUTD+'/fig1_learning_curves.png',dpi=220); plt.close()
# fig2 rare
e2=pd.read_csv('exp2_rare_defect.csv'); g=e2.groupby(['k_rare','method'])['rare_recall'].mean().unstack()
R=[('frozen_clip+LR','Frozen LR',C['frozen']),('frozen_clip+LR_balanced','Frozen LR, balanced weights',C['bal']),('finetune_clip','Fine-tune',C['ft']),('finetune_clip_weighted','Fine-tune, class-weighted',C['ftw'])]
fig,ax=plt.subplots(figsize=(6.4,3.1)); w=.2; ks2=[1,5,20]
for i,(m,lab,c) in enumerate(R):
    v=g.loc[ks2,m].values*100; b=ax.bar(np.arange(3)+(i-1.5)*w*1.05,v,w,color=c,label=lab,edgecolor=SURF,linewidth=1.5)
    if m=='finetune_clip_weighted': hatch=None
    for xx,yy in zip(np.arange(3)+(i-1.5)*w*1.05,v): ax.text(xx,yy+1.5,f'{yy:.0f}',ha='center',fontsize=7,color=INK2)
ax.set_xticks(range(3)); ax.set_xticklabels(['1','5','20']); ax.set_xlabel('Labelled images of the rare defect (others have 50 each)'); ax.set_ylabel('Rare-defect recall (%)'); ax.set_ylim(0,112); clean(ax)
ax.legend(fontsize=7.5,ncol=4,loc='upper center',bbox_to_anchor=(0.5,-0.2)); plt.tight_layout(); plt.savefig(OUTD+'/fig2_rare_defect.png',dpi=220); plt.close()
# fig3 open set
e3=pd.read_csv('exp3_open_set.csv'); O=[('frozen_clip+LR_maxprob','Frozen LR\nmax-prob',C['frozen']),('frozen_clip_knn_distance','Frozen\nnearest-neighbour',C['tuned']),('partial_finetune_clip_maxprob','Partial FT\nmax-prob',C['part']),('finetune_clip_maxprob','Full FT\nmax-prob',C['ft'])]
fig,ax=plt.subplots(figsize=(5.6,3.0)); rng=np.random.RandomState(1)
for i,(m,lab,c) in enumerate(O):
    d=e3[e3.method==m].groupby('novel_class')['auroc'].mean(); ax.bar(i,d.mean(),.55,color=c,edgecolor=SURF,alpha=.9)
    ax.scatter(i+rng.uniform(-.14,.14,len(d)),d.values,s=14,color=INK,zorder=3,lw=0); ax.text(i,.5+0.005,f'{d.mean():.3f}',ha='center',fontsize=8,color='white' if c!=C['tuned'] else INK,va='bottom')
ax.set_xticks(range(4)); ax.set_xticklabels([o[1] for o in O],fontsize=8); ax.set_ylim(.5,1.0); ax.set_ylabel('AUROC, unseen vs known defect'); clean(ax)
ax.set_title('Dots: each of the 6 held-out defect types (mean of 2 seeds)',fontsize=8,color=INK2,loc='left'); plt.tight_layout(); plt.savefig(OUTD+'/fig3_open_set.png',dpi=220); plt.close()
# fig4 robustness
e4=pd.read_csv('exp4_robustness.csv'); e4=e4[e4.k==20]
Rb=[('finetune_clip','Full FT (CLIP)',C['ft'],'-'),('finetune_clip_distaug','Full FT + blur/noise aug.',C['aug'],'--'),('finetune_imagenet_vit','Full FT (ImageNet ViT)',C['imn'],'-'),('partial_finetune_clip','Partial FT (CLIP)',C['part'],'-'),('frozen_clip+LR','Frozen LR',C['frozen'],'-')]
fig,ax=plt.subplots(1,2,figsize=(7.2,3.0),sharey=True)
for a,fam,xl in [(ax[0],'blur','Gaussian blur σ (pixels)'),(ax[1],'noise','Gaussian noise σ (grey levels)')]:
    for m,lab,c,ls in Rb:
        d=e4[(e4.method==m)&((e4['family']==fam)|(e4['shift']=='clean'))].copy(); d['lvl']=np.where(d['shift']=='clean',0,d['level'])
        d=d.groupby('lvl')['acc'].mean()*100; a.plot(d.index,d.values,ls,color=c,marker='o',ms=3.5,lw=1.6,label=lab)
    a.set_xlabel(xl+' (0 = clean)'); clean(a)
ax[0].set_ylabel('Test accuracy (%)'); ax[0].set_ylim(30,101); ax[0].set_title('(a) Blur',fontsize=9,color=INK,loc='left'); ax[1].set_title('(b) Noise',fontsize=9,color=INK,loc='left')
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=3,fontsize=7.5,bbox_to_anchor=(0.5,-0.02)); plt.tight_layout(rect=(0,0.13,1,1)); plt.savefig(OUTD+'/fig4_robustness.png',dpi=220); plt.close()
print('ok')
