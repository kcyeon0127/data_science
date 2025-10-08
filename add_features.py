import pandas as pd

def add_features(train, test):
    for col in ['new_feature']:
        train[col] = 0
        test[col] = 0

if __name__ == '__main__':
    train = pd.DataFrame({'a':[1,2],'b':[3,4]})
    test = train.copy()
    add_features(train, test)
    print(train)
    print(test)
