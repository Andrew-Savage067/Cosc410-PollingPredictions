#import pytorch
import pandas as pd
import random
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
import sklearn.tree
import keras as ks
import torch
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import tensorflow as tf
import math

#gets input and output data from the files
def get_data(filename):
    poll = pd.read_csv(filename)
    
    poll = poll.drop(index=poll.index[0])
    poll = poll.drop(columns=["Unnamed: 0","Sample", "Date", "Spread"], axis=1)
    poll = poll.drop(columns=["Poll"])
    poll.replace("--",0.0,inplace=True)
    poll = poll.astype(float)


    #print(filename)
    if filename == "Data\d_iowa_20.csv":
        poll["Days_Out"] = poll["Days_Out"]+7

    results = poll.head(1)
    poll = poll.iloc[1:]
    lastpoll = poll.head(1)
    poll = poll.iloc[1:]
    poll = poll.groupby(["Days_Out"]).mean()
    #print(poll.head(10))


    poll = poll.iloc[:,0:5]
    results = results.iloc[:,0:5]
    lastpoll = lastpoll.iloc[:,0:5]

    poll.columns = range(len(poll.columns))
    results.columns = range(len(results.columns))
    lastpoll.columns = range(len(lastpoll.columns))

    final = pd.DataFrame(0,index=poll.index,columns = [0,1,2,3,4])
    fResults = pd.DataFrame(0,index=results.index,columns = [0,1,2,3,4])
    fLastpoll = pd.DataFrame(0,index=lastpoll.index,columns = [0,1,2,3,4])
    final = poll.combine_first(final)
    results = results.combine_first(fResults)
    lastpoll = lastpoll.combine_first(fLastpoll)
    #print(results)
    #print(lastpoll)
    #print(final.head(5))
    return final/100, results/100, lastpoll/100


#initial simplified model that does not take time into account
def decisiontrain():
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

    #print(np.array(newX))
    X_train, X_test, y_train, y_test = train_test_split(newX,y, random_state=1, test_size=.4)

    #   print(X_train)
    dt = sklearn.tree.DecisionTreeClassifier(max_depth=2,random_state=1)

    clf_dt=dt.fit(X_train,y_train)
    score = clf_dt.score(X_test,y_test)
    print(f'Accuracy: {score}')
    return

#creates and returns RNN model given number of recurrent layers and number of features
def create_model(num_timesteps, features):
    
    inputs = ks.layers.Input(shape=(num_timesteps, features))
    #model.add(ks.layers.Embedding(vocab_size, embedding_dim))
    dense = ks.layers.Dense(15, activation="sigmoid")(inputs)
    lstm = ks.layers.LSTM(10)(dense)
    output = ks.layers.Dense(5, activation="sigmoid")(lstm)
    model = ks.models.Model(inputs = inputs, outputs = output)
    optimizer = Adam(learning_rate=.0001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    model.summary()
    return model


#formats and splits the data for use in the RNN
def RNNData(sample_number, finalResults):
    x = []
    yl = []
    yr = []
    for filename in glob.glob(os.path.join("Data", "*csv")):
        input, output, lastpoll = get_data(filename)
        input = np.array(input)
        #print(input.shape)
        for row in range(len(input)-sample_number-1):
            #print(input[row,:])
            addon = input[row:row+sample_number,:]
            #print(addon)
            #print(np.fliplr(addon))
            #print(indivTest.shape)
            yl.append(np.array(lastpoll)[0])
            yr.append(np.array(output)[0])
            x.append(np.array(addon))
    x = np.array(x)
    yl = np.array(yl)
    yr = np.array(yr)
    if finalResults:
        y = yr #results output
    else:
        y = yl #last poll output
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10, shuffle = True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=20, shuffle = True)
    return x_train, x_test, x_val, y_train, y_test, y_val
    

#tests the model using the test data
def test(model, x, y, finalResults):
    print("test results:")
    results = model.evaluate(x,y,20)
    print(results)

    #print(x[0])
    print(tf.nn.softmax(model(np.array([x[0]]))))
    print(y[0])

    #accuracy
    numCorrect = 0
    total = 0
    for test in range(len(x)):
        if(np.argmax(tf.nn.softmax(model(np.array([x[test]])))) == np.argmax(y[test])):
            numCorrect+=1
        total+=1
    print(f'Accuracy is {100*(numCorrect/total)}%')

    #percentage error
    error = 0
    for test in range(len(x)):
        error += abs(np.max(tf.nn.softmax(model(np.array([x[test]])))) - np.max(y[test]))
        total+=1
    print(f'Percentage Error is {100*(error/total)}%')
    return



#graphs the loss of valiudation data and trainintg data
def graph_fit(history):
    training = history.history["loss"]
    validation = history.history["val_loss"]
    epochs = range(len(training))
    plt.figure()
    plt.plot(epochs, training, "b", label="Training loss")
    plt.plot(epochs, validation, "r", label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    



def main():

    finalResults = True #change this to switch between looking at results and lastpoll(true is results)
    features = 5
    sampleNumber = 10#number of polls looked at (number of recurrent layers)
    numEpochs = 100
    batchSize = 20
    model = create_model(sampleNumber, features)
    #decisiontrain()
    x_train, x_test, x_val, y_train, y_test, y_val = RNNData(sampleNumber, finalResults)
    print(x_train.shape)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    history = model.fit(x_train, y_train, batch_size=batchSize, epochs=numEpochs, validation_data=(x_val, y_val))
    graph_fit(history)
    test(model, x_test, y_test, finalResults)


main()
