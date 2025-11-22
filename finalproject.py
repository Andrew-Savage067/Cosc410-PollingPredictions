#import pytorch
import pandas as pd
import random
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
import sklearn.tree


def get_data(filename):
    poll = pd.read_csv(filename)
    
    poll = poll.drop(index=poll.index[0])
    poll = poll.drop(columns=["Unnamed: 0","Sample", "Date", "Spread","Days_Out"], axis=1)
    results = poll.head(1).drop(columns=["Poll"])
    #poll = pd.get_dummies(poll, prefix = "Poll:", columns=["Poll"], drop_first=True, dtype=int)
    poll = poll.drop(columns=["Poll"])
    poll.replace("--",0.0,inplace=True)
    results.replace("--",0.0,inplace=True)
    poll = poll.astype(float)
    results = results.astype(float)
    print(np.argmax(np.array(results)))
    return poll.loc[0:41],np.argmax(np.array(results))

def train():
    x = []
    y = []
    for filename in glob.glob(os.path.join("Data", "*csv")):
        input,output = get_data(filename)
        x.append(np.array(input)[random.randint(0,40)])
        y.append(output)

    max_len = max(len(row) for row in x)
    newX = np.zeros((len(x), max_len))
    for i, row in enumerate(x):
        newX[i, :len(row)] = row

    print(np.array(newX))
    X_train, X_test, y_train, y_test = train_test_split(newX,y, random_state=10, test_size=.2)

    print(X_train)
    dt = sklearn.tree.DecisionTreeClassifier(max_depth=2,random_state=1)

    clf_dt=dt.fit(X_train,y_train)
    print(clf_dt.score(X_test,y_test))
    return
def main():
    train()
    #test()


main()