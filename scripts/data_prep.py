
import os, pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from huggingface_hub import HfApi, login
HF_TOKEN     = os.environ["HF_TOKEN"]
HF_USERNAME  = os.environ["HF_USERNAME"]
DATASET_REPO = f"{HF_USERNAME}/superkart-sales-forecast"
login(token=HF_TOKEN)
api = HfApi()
df = load_dataset(DATASET_REPO, data_files="data/superkart_raw.csv", split="train").to_pandas()
df['Product_Sugar_Content'] = df['Product_Sugar_Content'].str.strip()
df['Product_Sugar_Content'] = df['Product_Sugar_Content'].replace({'reg':'Regular','REG':'Regular'})
df.drop_duplicates(inplace=True)
df['Product_Weight'] = df.groupby('Product_Type')['Product_Weight'].transform(lambda x: x.fillna(x.median()))
df['Store_Size'].fillna(df['Store_Size'].mode()[0], inplace=True)
df['Store_Age'] = 2024 - df['Store_Establishment_Year']
df.drop(columns=['Product_Id','Store_Id','Store_Establishment_Year'], inplace=True)
cat_cols = ['Product_Sugar_Content','Product_Type','Store_Size','Store_Location_City_Type','Store_Type']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
X, y = df.drop(columns=['Product_Store_Sales_Total']), df['Product_Store_Sales_Total']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
os.makedirs("superkart/data", exist_ok=True)
os.makedirs("superkart/model", exist_ok=True)
train_df = X_train.copy(); train_df['Product_Store_Sales_Total'] = y_train.values
test_df  = X_test.copy();  test_df['Product_Store_Sales_Total']  = y_test.values
train_df.to_csv("superkart/data/train.csv", index=False)
test_df.to_csv("superkart/data/test.csv", index=False)
with open("superkart/model/encoders.pkl","wb") as f: pickle.dump(le_dict, f)
for fname in ["train.csv","test.csv"]:
    api.upload_file(path_or_fileobj=f"superkart/data/{fname}",
                    path_in_repo=f"data/{fname}",
                    repo_id=DATASET_REPO, repo_type="dataset", token=HF_TOKEN)
api.upload_file(path_or_fileobj="superkart/model/encoders.pkl",
                path_in_repo="encoders.pkl",
                repo_id=f"{HF_USERNAME}/superkart-model", repo_type="model", token=HF_TOKEN)
print("Data prep done.")
