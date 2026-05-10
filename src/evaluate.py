"""
evaluate.py — Metric computation
Extracted from notebook cells 46-47: exact match, test set evaluation, dashboard plot.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
    recall_score, r2_score, confusion_matrix)
from src.preprocessing import clean_text, expand_to_options, make_verify_corpus
from src.model_a_train import soft_vote_predict

def exact_match(predictions, ground_truths):
    """Exact match: 1 if predicted == ground truth (after normalising)."""
    def norm(s): return clean_text(str(s), lowercase=True, remove_punct=True)
    correct = sum(norm(p)==norm(g) for p,g in zip(predictions, ground_truths))
    return correct / max(len(predictions), 1)

def evaluate_test_set(rf_clf, svm_clf, lr_clf, ohe_verify, test_df, n=2000):
    """Full Model A evaluation on test set."""
    exp_test = expand_to_options(test_df.sample(n=min(n,len(test_df)),random_state=42))
    Xv_test = ohe_verify.transform(make_verify_corpus(exp_test))
    yv_test = exp_test['is_correct'].values
    preds, proba = soft_vote_predict([rf_clf,svm_clf,lr_clf], Xv_test)
    acc = accuracy_score(yv_test,preds); f = f1_score(yv_test,preds,average='macro')
    pr = precision_score(yv_test,preds,average='macro')
    rc = recall_score(yv_test,preds,average='macro')
    r2 = r2_score(yv_test, proba[:,1])
    # Exact match per question
    n_q = len(exp_test)//4; em_p, em_t = [], []
    for i in range(n_q):
        best = np.argmax(proba[i*4:(i+1)*4, 1])
        em_p.append(str(exp_test.iloc[i*4+best].get('option_text','')))
        em_t.append(str(exp_test.iloc[i*4].get('answer_text','')))
    em = exact_match(em_p, em_t)
    print(f'Test Acc={acc:.4f} F1={f:.4f} Prec={pr:.4f} Rec={rc:.4f} R2={r2:.4f} EM={em:.4f}')
    return {'test_acc':acc,'test_f1':f,'test_prec':pr,'test_rec':rc,
            'test_r2':r2,'exact_match':em}, preds, proba, yv_test

def plot_dashboard(metrics_a, rf_acc, svm_acc, lr_acc, ens_acc,
                   rf_f1, svm_f1, lr_f1, ens_f1, ens_prec, ens_rec,
                   test_metrics, test_preds, test_proba, yv_test,
                   cluster_summary, train_df, save_dir='content'):
    """Plot the full 6-panel analytics dashboard."""
    fig = plt.figure(figsize=(16, 12))
    # 1 — Bar chart
    ax1 = fig.add_subplot(2,3,1)
    mdf = pd.DataFrame({'Model':['RF','SVM','LR','Ens'],'F1':[rf_f1,svm_f1,lr_f1,ens_f1],
                         'Accuracy':[rf_acc,svm_acc,lr_acc,ens_acc]})
    x = np.arange(4); w = 0.35
    ax1.bar(x-w/2,mdf['Accuracy'],w,label='Acc',color='#4C78A8')
    ax1.bar(x+w/2,mdf['F1'],w,label='F1',color='#F58518')
    ax1.set_xticks(x); ax1.set_xticklabels(mdf['Model']); ax1.set_ylim(0,1.05)
    ax1.set_title('Model A — Dev'); ax1.legend(fontsize=8)
    # 2 — CM
    ax2 = fig.add_subplot(2,3,2)
    cm = confusion_matrix(yv_test,test_preds)
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax2,
        xticklabels=['Inc','Cor'],yticklabels=['Inc','Cor'])
    ax2.set_title('Test CM')
    # 3 — Clusters
    ax3 = fig.add_subplot(2,3,3)
    ax3.bar(cluster_summary.index,cluster_summary['correct_rate'],color='#54A24B',edgecolor='white')
    ax3.axhline(0.25,color='red',ls='--'); ax3.set_title('K-Means Clusters')
    # 4 — Proba histogram
    ax4 = fig.add_subplot(2,3,4)
    ax4.hist(test_proba[yv_test==0,1],bins=30,alpha=0.6,color='#E45756',label='Inc')
    ax4.hist(test_proba[yv_test==1,1],bins=30,alpha=0.6,color='#4C78A8',label='Cor')
    ax4.set_title('P(Correct) Dist.'); ax4.legend(fontsize=8)
    # 5 — Table
    ax5 = fig.add_subplot(2,3,5); ax5.axis('off')
    td = [['Metric','Dev','Test'],
          ['Accuracy',f'{ens_acc:.4f}',f'{test_metrics["test_acc"]:.4f}'],
          ['Macro-F1',f'{ens_f1:.4f}',f'{test_metrics["test_f1"]:.4f}'],
          ['Precision',f'{ens_prec:.4f}',f'{test_metrics["test_prec"]:.4f}'],
          ['Recall',f'{ens_rec:.4f}',f'{test_metrics["test_rec"]:.4f}'],
          ['EM','—',f'{test_metrics["exact_match"]:.4f}']]
    tbl = ax5.table(cellText=td,loc='center',cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2,1.6)
    ax5.set_title('Summary',pad=20)
    # 6 — Answer len
    ax6 = fig.add_subplot(2,3,6)
    if 'answer_len' in train_df.columns:
        train_df['answer_len'].hist(bins=30,ax=ax6,color='#72B7B2',edgecolor='white')
    ax6.set_title('Answer Length')
    plt.suptitle('RACE ML Pipeline — Dashboard',fontsize=14,y=1.01)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/analytics_dashboard.png',dpi=130,bbox_inches='tight')
    plt.show()
