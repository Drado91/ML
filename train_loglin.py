import loglinear as ll
import random
import numpy as np
from utils import read_data,text_to_bigrams,vocabu
from loglinear import loss_and_gradients,create_classifier
import utils
from collections import Counter

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def feats_to_vec(features,F2I):
    # YOUR CODE HERE.
  features_vec=np.zeros([len(features),600])
  for i in range(len(features)):
     for j in range(len(features[i])):
        if features[i][j] in F2I.keys():
            features_vec[i,F2I.get(features[i][j])] = features_vec[i,F2I.get(features[i][j])] + 1
  return features_vec

def labels_to_y(TRAIN,L2I):
    y=np.zeros([len(TRAIN),1])
    for i in range(len(TRAIN)):
        if TRAIN[i][0] in L2I.keys():
          #  y[i][L2I.get(TRAIN[i][0])]=1 retrun y 6X2334
          y[i] = L2I.get(TRAIN[i][0])
    return y

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        pass
    return good / (good + bad)

def train_classifier(train_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.
    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data);
        for l in range(train_data.shape[0]):
            x = train_data[l,1:]             # convert features to a vector.
            y = train_data[l,0]          # convert the label to number if needed.
            loss, grads = loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            grad_W, grad_b = grads
            W, b = params
            params[0] = grad_W*learning_rate + params[0]
            params[1] =


        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        #dev_accuracy = accuracy_on_dataset(dev_data, params)
        #print (I, train_loss, train_accuracy, dev_accuracy)
    return params

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    TRAIN = [(l, text_to_bigrams(t)) for l, t in read_data(open('train', 'r'))]
    train_vocab = vocabu(TRAIN)
    L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}
    F2I = {f: i for i, f in enumerate(list(sorted(vocabu(TRAIN))))}


    features=[]
    for idx in range(len(TRAIN)): #take the features out of TRAIN
        features.append(TRAIN[idx][-1])
    ##print(F2I[features[1][3]])
    features_vec=feats_to_vec(features,F2I)
    y=labels_to_y(TRAIN,L2I)
    in_dim = len(F2I)
    out_dim = len(L2I)
    params = create_classifier(in_dim, out_dim)
   # y=y.reshape(-1)
    features_vec=np.array(features_vec)
    train_data=np.concatenate((y,features_vec),axis=1)
    #p=random.shuffle(train_data);
    # train_classifier
    # ...
    params = create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, 5, 0.01, params)

