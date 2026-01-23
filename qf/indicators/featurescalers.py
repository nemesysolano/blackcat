
def scale_with_multiplier(args, multiplier):
    (dataset, features, target) = args

    dataset[target] = dataset[target] * multiplier
    for feature in features:
        dataset[feature] = dataset[feature] * multiplier
    return (dataset, features, target)

