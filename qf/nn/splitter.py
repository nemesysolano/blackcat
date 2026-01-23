def create_datasets(args): 
    (dataset, features, target) = args
    # print(dataset)
    # print(features)
    # print(target)
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

    return X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy(), Y_train.to_numpy(), Y_val.to_numpy(), Y_test.to_numpy()
 

