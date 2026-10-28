"""Part B: Customer segmentation and lapse prediction. Run from repository root."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
orders = pd.read_csv(ROOT / 'data/clean/orders.csv', parse_dates=['order_ts'])
customers = pd.read_csv(ROOT / 'data/clean/customers.csv', parse_dates=['signup_date'])
products = pd.read_csv(ROOT / 'data/clean/products.csv')
orders = orders.merge(products[['product_id','category']], on='product_id', how='left', validate='many_to_one')
assert orders['category'].notna().all()
completed = orders.loc[orders.status.eq('completed')].copy()

# 1. RFM, using the whole window only for descriptive segmentation.
snapshot = pd.Timestamp('2025-10-01')
rfm = completed.groupby('customer_id').agg(last_order=('order_ts','max'), frequency=('order_id','nunique'), monetary=('order_amount','sum')).reset_index()
rfm['recency'] = (snapshot-rfm.last_order).dt.days
rfm = rfm[['customer_id','recency','frequency','monetary']]
print('\nRFM CHECKPOINTS')
print('Customers:', len(rfm), 'recency median/min/max:', rfm.recency.median(), rfm.recency.min(), rfm.recency.max())
print('Frequency mean/max:', round(rfm.frequency.mean(),2), rfm.frequency.max())
print('Monetary mean/max:', round(rfm.monetary.mean(),2), rfm.monetary.max())
assert len(rfm)==1175

# 2. K-means, standardized features; evaluate K=2..8.
rfm_cols = ['recency','frequency','monetary']
X_rfm = StandardScaler().fit_transform(rfm[rfm_cols])
k_rows=[]
for k in range(2,9):
    km=KMeans(n_clusters=k,n_init=10,random_state=42)
    labels=km.fit_predict(X_rfm)
    k_rows.append({'K':k,'inertia':km.inertia_,'silhouette':silhouette_score(X_rfm,labels)})
print('\nK-MEANS (K=2..8)')
print(pd.DataFrame(k_rows).round(3).to_string(index=False))
final_k=4
km=KMeans(n_clusters=final_k,n_init=10,random_state=42)
rfm['segment_id']=km.fit_predict(X_rfm)
profile=rfm.groupby('segment_id').agg(size=('customer_id','size'),mean_recency=('recency','mean'),mean_frequency=('frequency','mean'),mean_monetary=('monetary','mean'))
premium_id=int(profile.mean_monetary.idxmax())
# Business names depend on cluster behavior, not the numeric cluster ID.
segment_names={premium_id:'Premium'}
remaining=[int(i) for i in profile.index if i != premium_id]
frequent_id=max(remaining,key=lambda i:profile.loc[i,'mean_frequency'])
segment_names[frequent_id]='Frequent'
remaining.remove(frequent_id)
at_risk_id=max(remaining,key=lambda i:profile.loc[i,'mean_recency'])
segment_names[at_risk_id]='At Risk'
remaining.remove(at_risk_id)
for i in remaining: segment_names[i]='Occasional'
rfm['segment_name']=rfm.segment_id.map(segment_names)
rfm['is_premium']=rfm.segment_id.eq(premium_id)
rfm.to_csv(OUT/'customer_segments.csv',index=False)
print('\nSEGMENT PROFILES')
print(profile.round(2).to_string())
print('Names:',segment_names,'Premium segment:',premium_id)
pca=PCA(n_components=2).fit(X_rfm)
print('PCA ratios:',pca.explained_variance_ratio_.round(4),'total:',round(pca.explained_variance_ratio_.sum(),4))

# 3. Pre-cutoff features; September orders are used ONLY to create labels.
cutoff=pd.Timestamp('2025-09-01')
end=pd.Timestamp('2025-10-01')
pre=orders.loc[orders.order_ts.lt(cutoff)].copy()
pre_done=pre.loc[pre.status.eq('completed')].copy()
cohort_ids=sorted(pre_done.customer_id.unique())
base=pre_done.groupby('customer_id').agg(
    last_order=('order_ts','max'),frequency=('order_id','nunique'),
    monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),
    avg_delivery=('delivery_minutes','mean'))
base['recency']=(cutoff-base.last_order).dt.days
base=base.drop(columns='last_order')
cancel_rate=pre.groupby('customer_id').status.apply(lambda s:s.eq('cancelled').mean()).rename('cancellation_rate')
base=base.join(cancel_rate)
# Most common category; tie-break alphabetically for deterministic output.
cat_counts=pre_done.groupby(['customer_id','category']).size().rename('n').reset_index()
cat_counts=cat_counts.sort_values(['customer_id','n','category'],ascending=[True,False,True])
fav=cat_counts.drop_duplicates('customer_id').set_index('customer_id')['category'].rename('favourite_category')
base=base.join(fav)
base=base.join(customers.set_index('customer_id')[['city','age','gender','signup_date']])
base['tenure']=(cutoff-base.signup_date).dt.days
base=base.drop(columns='signup_date').loc[cohort_ids]
sept_ids=set(completed.loc[completed.order_ts.ge(cutoff)&completed.order_ts.lt(end),'customer_id'])
base['lapsed']=(~base.index.isin(sept_ids)).astype(int)
base=base.reset_index()
print('\nLAPSE COHORT:',len(base),'lapsed:',int(base.lapsed.sum()),'rate:',round(100*base.lapsed.mean(),2),'%')
assert len(base)==1156 and int(base.lapsed.sum())==443
num_cols=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','age','tenure']
cat_cols=['favourite_category','city','gender']
feature_cols=num_cols+cat_cols
assert not base[feature_cols].isna().any().any(),base[feature_cols].isna().sum()
X=base[feature_cols]
y=base.lapsed
X_train,X_test,y_train,y_test,idx_train,idx_test=train_test_split(X,y,base.index,test_size=.2,stratify=y,random_state=42)
print('Split:',len(X_train),'train,',len(X_test),'test')
print('Baseline accuracy:',round(accuracy_score(y_test,np.zeros(len(y_test))),4),'lapsed F1:',f1_score(y_test,np.zeros(len(y_test)),zero_division=0))

def preprocess():
    return ColumnTransformer([
        ('numeric',StandardScaler(),num_cols),
        ('categorical',OneHotEncoder(drop='first',handle_unknown='ignore'),cat_cols)
    ])

# 4. Hyperparameter tuning. All transforms are fitted on training folds only.
log_pipe=Pipeline([('preprocess',preprocess()),('model',LogisticRegression(max_iter=1000,random_state=42))])
log_search=GridSearchCV(log_pipe,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=-1)
log_search.fit(X_train,y_train)
rf_pipe=Pipeline([('preprocess',preprocess()),('model',RandomForestClassifier(n_estimators=100,random_state=42))])
rf_search=GridSearchCV(rf_pipe,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=-1)
rf_search.fit(X_train,y_train)
print('\nTUNING')
print('Logistic:',log_search.best_params_,'CV F1:',round(log_search.best_score_,4))
print('Forest:',rf_search.best_params_,'CV F1:',round(rf_search.best_score_,4))

def evaluate(label,model):
    pred=model.predict(X_test)
    scores={'precision':precision_score(y_test,pred,zero_division=0),'recall':recall_score(y_test,pred,zero_division=0),'f1':f1_score(y_test,pred,zero_division=0)}
    print(label,':', {k:round(v,4) for k,v in scores.items()},'confusion matrix:\n',confusion_matrix(y_test,pred))
    return scores
log_scores=evaluate('Logistic',log_search.best_estimator_)
rf_scores=evaluate('Random forest',rf_search.best_estimator_)

# 5. Training-only random oversampling, to balance class counts.
train_lapsed=np.flatnonzero(y_train.to_numpy()==1)
train_not=np.flatnonzero(y_train.to_numpy()==0)
rng=np.random.RandomState(42)
extra=rng.choice(train_lapsed,size=len(train_not)-len(train_lapsed),replace=True)
positions=np.concatenate([np.arange(len(X_train)),extra])
X_bal=X_train.iloc[positions]
y_bal=y_train.iloc[positions]
assert y_bal.value_counts().nunique()==1
balanced_log=Pipeline([('preprocess',preprocess()),('model',LogisticRegression(max_iter=1000,C=log_search.best_params_['model__C'],random_state=42))])
balanced_log.fit(X_bal,y_bal)
print('\nBALANCING: original',y_train.value_counts().to_dict(),'balanced',y_bal.value_counts().to_dict())
balanced_scores=evaluate('Balanced logistic',balanced_log)
print('Fitted training-only frequency mean:',round(float(log_search.best_estimator_.named_steps['preprocess'].named_transformers_['numeric'].mean_[num_cols.index('frequency')]),4))
print('Full cohort frequency mean:',round(float(base.frequency.mean()),4))

# 6. Model interpretation.
rf_best=rf_search.best_estimator_
log_best=log_search.best_estimator_
feature_names=rf_best.named_steps['preprocess'].get_feature_names_out()
importance=pd.Series(rf_best.named_steps['model'].feature_importances_,index=feature_names).sort_values(ascending=False)
print('\nTOP FIVE RANDOM FOREST FEATURES')
print(importance.head(5).round(4).to_string())
log_names=log_best.named_steps['preprocess'].get_feature_names_out()
coeffs=pd.Series(log_best.named_steps['model'].coef_[0],index=log_names)
print('\nLOGISTIC COEFFICIENT SIGNS (positive = higher lapse odds)')
for col in ['numeric__recency','numeric__frequency','numeric__monetary']:
    print(col,round(float(coeffs[col]),4),'positive' if coeffs[col]>0 else 'negative')

# 7. Choose based on untouched test-set lapsed-class F1.
selected_name,selected=('Random Forest',rf_best) if rf_scores['f1']>=log_scores['f1'] else ('Logistic Regression',log_best)
assert max(rf_scores['f1'],log_scores['f1'])>=.55,'F1 below required threshold'
joblib.dump(selected,OUT/'lapse_model.joblib')
reloaded=joblib.load(OUT/'lapse_model.joblib')
assert np.array_equal(selected.predict(X_test),reloaded.predict(X_test))
print('\nSELECTED MODEL:',selected_name,'(reload predictions identical)')

# Premium is post-hoc and therefore not leakage-free for retention evaluation.
test_customers=base.loc[idx_test,['customer_id','lapsed']].copy()
test_customers['lapse_probability']=selected.predict_proba(X_test)[:,1]
test_customers=test_customers.merge(rfm[['customer_id','is_premium']],on='customer_id',how='left')
retention=(test_customers.loc[test_customers.is_premium.eq(True)]
           .sort_values(['lapse_probability','customer_id'],ascending=[False,True]).head(10)
           .rename(columns={'lapsed':'actually_lapsed'})
           [['customer_id','lapse_probability','actually_lapsed']])
retention.to_csv(OUT/'retention_list.csv',index=False)
print('Premium test-set retention list:',len(retention),'actually lapsed:',int(retention.actually_lapsed.sum()))
print('NOTE: Premium flag uses the full window including September; the retention list is illustrative, not leakage-free.')
print('\nPART B OUTPUT FILES CREATED')
for filename in ['customer_segments.csv','lapse_model.joblib','retention_list.csv']:
    print(OUT/filename)
