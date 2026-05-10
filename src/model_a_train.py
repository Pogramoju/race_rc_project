"""
model_a_train.py — Training script for Model A
(Question & Answer Generator / Verifier)
Trains supervised (RF, SVM, LR), unsupervised (K-Means),
semi-supervised (Label Propagation), ensemble, and question ranker.
"""
import time, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.semi_supervised import LabelSpreading
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report, silhouette_score)
from sklearn.model_selection import train_test_split
from src.preprocessing import clean_text, tokenize

def train_random_forest(Xv_train, yv_train, Xv_dev, yv_dev):
    print('Training Random Forest...')
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=20,
        class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(Xv_train, yv_train)
    p = rf.predict(Xv_dev); t = time.time()-t0
    a, f = accuracy_score(yv_dev,p), f1_score(yv_dev,p,average='macro')
    pr, rc = precision_score(yv_dev,p,average='macro'), recall_score(yv_dev,p,average='macro')
    print(f'RF Acc={a:.4f} F1={f:.4f} ({t:.1f}s)')
    print(classification_report(yv_dev,p,target_names=['Incorrect','Correct']))
    return rf, {'rf_acc':a,'rf_f1':f,'rf_precision':pr,'rf_recall':rc}

def train_svm(Xv_train, yv_train, Xv_dev, yv_dev):
    print('Training Linear SVM...')
    t0 = time.time()
    svm = CalibratedClassifierCV(LinearSVC(C=1.0,class_weight='balanced',max_iter=2000,random_state=42))
    svm.fit(Xv_train, yv_train)
    p = svm.predict(Xv_dev); t = time.time()-t0
    a, f = accuracy_score(yv_dev,p), f1_score(yv_dev,p,average='macro')
    pr, rc = precision_score(yv_dev,p,average='macro'), recall_score(yv_dev,p,average='macro')
    print(f'SVM Acc={a:.4f} F1={f:.4f} ({t:.1f}s)')
    print(classification_report(yv_dev,p,target_names=['Incorrect','Correct']))
    return svm, {'svm_acc':a,'svm_f1':f,'svm_precision':pr,'svm_recall':rc}

def train_logistic_regression(Xv_train, yv_train, Xv_dev, yv_dev):
    print('Training Logistic Regression...')
    lr = LogisticRegression(C=1.0,class_weight='balanced',max_iter=1000,
        solver='lbfgs',random_state=42,n_jobs=-1)
    lr.fit(Xv_train, yv_train)
    p = lr.predict(Xv_dev)
    a, f = accuracy_score(yv_dev,p), f1_score(yv_dev,p,average='macro')
    print(f'LR Acc={a:.4f} F1={f:.4f}')
    return lr, {'lr_acc':a,'lr_f1':f}

def train_kmeans(Xv_train, yv_train, n_clusters=20, save_dir='content'):
    print(f'Running K-Means (k={n_clusters})...')
    t0 = time.time()
    Xn = sk_normalize(Xv_train, norm='l2')
    km = MiniBatchKMeans(n_clusters=n_clusters,random_state=42,n_init=5,batch_size=2048,max_iter=100)
    km.fit(Xn); print(f'Done in {time.time()-t0:.1f}s')
    cdf = pd.DataFrame({'cluster':km.labels_,'is_correct':yv_train})
    cs = cdf.groupby('cluster')['is_correct'].agg(['mean','count'])
    cs.columns = ['correct_rate','size']
    fig,ax = plt.subplots(figsize=(10,4))
    ax.bar(cs.index,cs['correct_rate'],color='#4C78A8',edgecolor='white')
    ax.axhline(0.25,color='red',ls='--',lw=1.5,label='Baseline (25%)')
    ax.set_title('K-Means Cluster Correct Rates'); ax.legend()
    plt.tight_layout(); plt.savefig(f'{save_dir}/kmeans_clusters.png',dpi=120); plt.show()
    return km, Xn, cs

def train_label_propagation(Xv_train_norm, yv_train, label_frac=0.15, semi_sample=3000):
    idx = np.arange(min(semi_sample,len(yv_train)))
    idx_lab = np.random.choice(idx,size=int(label_frac*len(idx)),replace=False)
    mask = np.full(len(idx),-1); mask[idx_lab] = yv_train[idx_lab]
    X_semi = Xv_train_norm[:len(idx)].toarray()
    t0 = time.time()
    lp = LabelSpreading(kernel='knn',n_neighbors=7,alpha=0.2,max_iter=30)
    lp.fit(X_semi,mask); pred = lp.predict(X_semi)
    ul = mask == -1
    a = accuracy_score(yv_train[:len(idx)][ul],pred[ul])
    f = f1_score(yv_train[:len(idx)][ul],pred[ul],average='macro')
    print(f'LP Acc={a:.4f} F1={f:.4f} ({time.time()-t0:.1f}s)')
    return lp, {'lp_acc':a,'lp_f1':f}

WH_MAP = {'who':0,'what':1,'where':2,'when':3,'why':4,'how':5,'which':6}
def question_features(question, article):
    q_lower = question.lower(); wh_type = 7
    for wh,idx in WH_MAP.items():
        if q_lower.startswith(wh): wh_type=idx; break
    q_tok = set(tokenize(question)); a_tok = set(tokenize(article))
    return [len(question.split()), wh_type, len(q_tok&a_tok)/max(len(q_tok),1)]

def train_question_ranker(test_df, lr_clf, ohe_verify, gen_q_fn, n=200):
    feats, labels = [], []
    for _,row in test_df.head(n).iterrows():
        art,q,ans = str(row.get('article','')),str(row.get('question','')),str(row.get('answer_text',''))
        feats.append(question_features(q,art)); labels.append(1)
        gq = gen_q_fn(art,ans,lr_clf,ohe_verify)
        feats.append(question_features(gq,art)); labels.append(0)
    X,y = np.array(feats),np.array(labels)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=42)
    svm_r = SVC(kernel='rbf',probability=True,random_state=42); svm_r.fit(Xtr,ytr)
    rf_r = RandomForestClassifier(n_estimators=100,random_state=42); rf_r.fit(Xtr,ytr)
    print(f'Q-Ranker SVM={accuracy_score(yte,svm_r.predict(Xte)):.4f} RF={accuracy_score(yte,rf_r.predict(Xte)):.4f}')
    return svm_r, rf_r

def soft_vote_predict(classifiers, X):
    proba_sum = np.zeros((X.shape[0],2))
    for clf in classifiers: proba_sum += clf.predict_proba(X)
    avg = proba_sum/len(classifiers)
    return (avg[:,1]>=0.5).astype(int), avg

def evaluate_ensemble(rf,svm,lr,Xv,yv,save_dir='content'):
    preds,proba = soft_vote_predict([rf,svm,lr],Xv)
    a = accuracy_score(yv,preds); f = f1_score(yv,preds,average='macro')
    pr = precision_score(yv,preds,average='macro'); rc = recall_score(yv,preds,average='macro')
    cm = confusion_matrix(yv,preds)
    fig,ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax,
        xticklabels=['Incorrect','Correct'],yticklabels=['Incorrect','Correct'])
    ax.set_title('Model A — Ensemble CM'); plt.tight_layout()
    plt.savefig(f'{save_dir}/cm_model_a.png',dpi=120); plt.show()
    return {'ens_acc':a,'ens_f1':f,'ens_prec':pr,'ens_rec':rc,'confusion_matrix':cm.tolist()}, preds, proba

def build_comparison_table(metrics_a, Xv_train_norm, kmeans):
    s = min(5000,Xv_train_norm.shape[0])
    sil = silhouette_score(Xv_train_norm[:s],kmeans.predict(Xv_train_norm[:s]))
    df = pd.DataFrame({
        'Model':['RF','SVM','LR','Ensemble','K-Means','Label Prop.'],
        'Paradigm':['Supervised']*3+['Ensemble','Unsupervised','Semi-Supervised'],
        'Accuracy':[metrics_a.get(k,0) for k in ['rf_acc','svm_acc','lr_acc','ens_acc']]+['—',metrics_a.get('lp_acc',0)],
        'Macro-F1':[metrics_a.get(k,0) for k in ['rf_f1','svm_f1','lr_f1','ens_f1']]+['—',metrics_a.get('lp_f1',0)],
        'Silhouette':['—']*4+[round(sil,4),'—'],
    })
    print(df.to_string(index=False))
    return df, sil
