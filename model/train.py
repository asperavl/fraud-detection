import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("Loading data...")
trans = pd.read_csv('data/train_transaction.csv')
identity = pd.read_csv('data/train_identity.csv')
df = trans.merge(identity, on='TransactionID', how='left')

print("Preprocessing...")
target = df['isFraud']
df = df.drop(columns=['isFraud', 'TransactionID'])

drop_cols = ['P_emaildomain', 'R_emaildomain', 'DeviceInfo']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].astype('category').cat.codes

df = df.fillna(-999)

X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42, stratify=target)

print("Training...")
fraud_ratio = (y_train==0).sum() / (y_train==1).sum()
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    scale_pos_weight=fraud_ratio,
    eval_metric='aucpr',
    random_state=42
)
model.fit(X_train,y_train)

print("\nEvaluation on test set:")
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))

joblib.dump(model, 'api/model.joblib')
joblib.dump(list(df.columns), 'api/features.joblib')
print("\nModel saved to api/model.joblib")