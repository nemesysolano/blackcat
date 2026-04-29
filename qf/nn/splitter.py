# đ1, đ2, đ3, and đ4
# đ1_1, đ1_2, đ1_3, đ1_4, ...
from datetime import timedelta

import numpy as np

def create_lstm_features(X, lookback_periods, length, col_prefix):
    features = []
    index = X.index

    for t in index:
        feature = []
        for i in range(1, lookback_periods + 1):          
            item = X.loc[t:t, f"{col_prefix}{i}_{length}":f"{col_prefix}{i}_1"]
            feature.append(item.values)
        features.append(np.vstack(feature))

    return features

def create_lstm_targets(Y):
    rows = []
    index = Y.index

    for t in index:
        rows.append(Y.loc[t].values)
    return rows

def create_lstm_datasets(args):
    X_train, X_val, X_test, Y_train, Y_val, Y_test, test_data = create_local_lstm_datasets(args) # (X, lookback_periods, length, col_prefix)
    (_, _, _, lookback_periods, length, col_prefix) = args # 
    features = lambda X: np.array(create_lstm_features(X, lookback_periods, length, col_prefix))
    targets = lambda Y: np.array(create_lstm_targets(Y))
    return features(X_train), features(X_val), features(X_test), Y_train.values, Y_val.values, Y_test.values, test_data

def create_local_lstm_datasets(args):
    (dataset, features, targets, _, _, _) = args
    n = len(dataset)

    # Define split points for a 65/20/15 split
    # We use cumulative indices to slice the list correctly
    train_end = int(n * 0.65)
    val_end = train_end + int(n * 0.20)
    index = dataset.index

    train_data = dataset[:train_end]
    val_data = dataset[train_end:val_end]
    test_data = dataset[val_end:]

    train_data, val_data, test_data
    X_train = train_data[features]
    X_val = val_data[features]
    X_test = test_data[features]

    Y_train = train_data[targets]
    Y_val = val_data[targets]
    Y_test = test_data[targets]
    
    assert len(X_train) == len(Y_train)
    assert len(X_val) == len(Y_val)
    assert len(X_test) == len(Y_test)
    assert len(Y_train) + len(Y_val) + len(Y_test) == n

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, test_data

def create_datasets(args):     
    X_train, X_val, X_test, Y_train, Y_val, Y_test, test_data = create_local_datasets(args)
    return X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy(), Y_train.to_numpy(), Y_val.to_numpy(), Y_test.to_numpy(), test_data
 

def create_local_datasets(args): 
    (dataset, features, target) = args
    n = len(dataset)

    # Define split points for a 65/20/15 split
    # We use cumulative indices to slice the list correctly
    train_end = int(n * 0.65)
    val_end = train_end + int(n * 0.20)
    
    train_data = dataset[:train_end]
    val_data = dataset[train_end:val_end]
    test_data = dataset[val_end:]
    
    train_data, val_data, test_data
    X_train = train_data[features]
    X_val = val_data[features]
    X_test = test_data[features]
    Y_train = train_data[target]
    Y_val = val_data[target]
    Y_test = test_data[target]

    assert len(X_train) == len(Y_train)
    assert len(X_val) == len(Y_val)
    assert len(X_test) == len(Y_test)
    assert len(Y_train) + len(Y_val) + len(Y_test) == n

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, test_data
 