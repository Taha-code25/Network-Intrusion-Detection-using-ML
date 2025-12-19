import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def Load_Data():
    df_train = pd.read_csv("kdd_train.csv")
    df_test = pd.read_csv("kdd_test.csv")
    return pd.concat([df_train,df_test])

def preprocessing(df):
    df.replace([float('inf'), float('inf')], pd.NA, inplace = True)
    df.dropna(inplace = True)
    return df

def Encoding(df):
    categorical_cols = ['protocol_type', 'service','flag']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def normalize_Labels(df):
    df['labels'] = df['labels'].apply(lambda x: 0 if x == 'normal' else 1)
    return df
def scalingAndBalance(df):
    df.drop_duplicates(inplace = True)
    X = df.drop('labels',axis = 1)
    y = df['labels']

    scaler = StandardScaler()
    X_scaled= scaler.fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    s = SMOTE(random_state=42)
    X_train_resampled,y_train_resampled = s.fit_resample(X_train,y_train)

    return X_train_resampled,y_train_resampled, X_test,y_test
