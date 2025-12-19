from sklearn.feature_selection import SelectKBest,f_classif
import numpy as np
import pandas as pd
from data_preprocessing import (
    Load_Data,
    preprocessing,
    Encoding,
    normalize_Labels,
    scalingAndBalance
)
def feature_selection():
    df = Load_Data()
    columns = df.columns.tolist()
    df = preprocessing(df)
    df = Encoding(df)
    df = normalize_Labels(df)
    X_train, y_train, X_test, y_test = scalingAndBalance(df)

    corr_matrix = pd.DataFrame(X_train).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X_train = np.delete(X_train,to_drop,axis = 1)
    X_test = np.delete(X_test,to_drop, axis = 1)

    selector = SelectKBest(f_classif, k = 20)
    X_train_selected = selector.fit_transform(X_train,y_train)
    X_test_selected = selector.transform(X_test)

    selected_features = [columns[i] for i in selector.get_support(indices = True)]
    print(f"Selected features: {selected_features}")
    return X_train_selected,X_test_selected, y_train,y_test
