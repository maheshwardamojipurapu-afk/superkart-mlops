
import os, pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from datasets import load_dataset
from huggingface_hub import HfApi, login, create_repo
HF_TOKEN     = os.environ["HF_TOKEN"]
HF_USERNAME  = os.environ["HF_USERNAME"]
DATASET_REPO = f"{HF_USERNAME}/superkart-sales-forecast"
MODEL_REPO   = f"{HF_USERNAME}/superkart-model"
login(token=HF_TOKEN); api = HfApi()
train_df = load_dataset(DATASET_REPO, data_files="data/train.csv", split="train").to_pandas()
test_df  = load_dataset(DATASET_REPO, data_files="data/test.csv",  split="train").to_pandas()
X_train = train_df.drop(columns=['Product_Store_Sales_Total'])
y_train = train_df['Product_Store_Sales_Total']
X_test  = test_df.drop(columns=['Product_Store_Sales_Total'])
y_test  = test_df['Product_Store_Sales_Total']
model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                     subsample=0.8, random_state=42, verbosity=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test,y_pred)):.2f}")
print(f"R2: {r2_score(y_test,y_pred):.4f}")
os.makedirs("superkart/model", exist_ok=True)
with open("superkart/model/best_model.pkl","wb") as f: pickle.dump(model, f)
try: create_repo(MODEL_REPO, repo_type="model", exist_ok=True, token=HF_TOKEN)
except: pass
api.upload_file(path_or_fileobj="superkart/model/best_model.pkl",
                path_in_repo="best_model.pkl",
                repo_id=MODEL_REPO, repo_type="model", token=HF_TOKEN)
print("Model registered.")
