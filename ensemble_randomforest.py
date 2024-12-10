
import pandas as pd 
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
train_x=pd.read_csv('train_stack.csv').values
train_y=(pd.read_csv('Train_SyntheticAKI_MAIC2023.csv')[['O_AKI','O_Critical_AKI_90','O_Death_90','O_RRT_90']]).values
test_x=pd.read_csv('test_stack_1211.csv').values
submit=pd.read_csv("Submission_example.csv")

# forest = LogisticRegression()
# multi_target_forest = MultiOutputClassifier(forest,n_jobs=4)
# multi_target_forest.fit(train_x, train_y)
# preds=multi_target_forest.predict_proba(test_x)
# sub=pd.DataFrame(preds,columns=submit.columns)
# sub.to_csv(f"StackingEnsemble_rf.csv",index=False)

# clf = xgb.XGBClassifier(n_estimators=20,tree_method="hist", multi_strategy="multi_output_tree")
# preds=clf.fit(train_x, train_y).predict_proba(test_x)
# sub=pd.DataFrame(preds,columns=submit.columns)
# sub.to_csv(f"StackingEnsemble_rf.csv",index=False)