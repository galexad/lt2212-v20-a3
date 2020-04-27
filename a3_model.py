import warnings
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import random
from torch.utils import data
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()


    def load_data(data):
        print("------------Loading Data-----------")
        data = pd.read_csv(data)
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        # splitting train/test data
        for i,row in data.iterrows():
            if data['Train/test'][i] == 0:
                X_train.append(row)
                y_train.append(data['Authors'][i])
            else:
                X_test.append(row)
                y_test.append(data['Authors'][i])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # convert train/test data to tensors
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()
        return X_train, y_train, X_test, y_test


    class DocPredict(nn.Module): 
        """
        Instantiating the nn.Linear module
        """

        def __init__(self, input_size=3832, num_classes=1):
            super().__init__()      
            self.linear = nn.Linear(input_size, num_classes)


        def forward(self, x):
            x = x.unsqueeze(0)
            # using a sigmoid as an activation function
            y_pred = torch.sigmoid(self.linear(x)) 

            return y_pred


    class DocFFNN():
        """ Instantiating the feed forward NN)"""

        def get_samples(self, X_train, y_train):
            samples = []
            self.documents = X_train
            self.authors = y_train
            a1 = []
            a0 = []

            # randomly select a document[i]
            i = np.random.randint(0, len(X_train))

            for i in range(i, len(X_train)-1):

                # flipping a coin so that k is either 1 or 0
                k = np.random.randint(0, 2)

                for j in range(0, len(X_train)-1):

                    self.doc1 = self.documents[i]
                    self.doc2 = self.documents[j]

                    # check if documents have same or different authors and split them in 2 groups
                    if self.authors[i] == self.authors[j]:
                        a1.append(self.doc2)   

                    else:
                        a0.append(self.doc2)

                # if k is 1 and d1 has the same author with d2, select the d2, elif k is 0 choose a d2 with a different author        
                if k == 1:
                    x = np.random.randint(0, len(a1))
                    self.doc2 = a1[x]
                    samples.append([self.doc1, self.doc2, k])  

                else:
                    x = np.random.randint(0, len(a0))
                    self.doc2 = a0[x]
                    samples.append([self.doc1, self.doc2, k])
            return samples        

        def process_samples(self, s):
            self.instance = Variable(torch.cat((s[i][0], s[i][1])))

        def make_model(self, inputfeatures):
            self.model = DocPredict(inputfeatures, 1)
            
            
        def __init__(self, X_train, y_train, epochs=20, lr=0.01):
            super(DocFFNN, self).__init__()
            self.epochs = epochs
            self.lr = lr

            self.features = []
            self.samples = self.get_samples(X_train, y_train)

            for i in range(len(self.samples)):

                instance = Variable(torch.cat((self.samples[i][0], self.samples[i][1])))
                self.features = len(instance)

            self.make_model(self.features)

        def train(self, inputs):
            """
            The training loop.
            """

            criterion = nn.BCELoss()
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

            loss_arr = []

            for j in range(self.epochs):
                for i in range(len(self.samples)):
                    # create instance
                    instance = Variable(torch.cat((self.samples[i][0], self.samples[i][1])))

                    # get instance label
                    label = Variable(torch.Tensor([self.samples[i][2]]))
                    train_outputs = self.model.forward(instance)

                    loss = criterion(train_outputs, label.unsqueeze(1))
                    loss_arr.append(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        def predict(self, X, author_labels):
            preds = []
            y_true = []
            test_samples = self.get_samples(X, author_labels)
            with torch.no_grad():
                for i in range(len(test_samples)):
                    instance = Variable(torch.cat((test_samples[i][0], test_samples[i][1])))
                    y = Variable(torch.Tensor([test_samples[i][2]]))
                    y_pred = self.model.forward(instance)
                    preds.append(y_pred.argmax().item())
                    y_true.append(y)
                y_true = np.array(y_true)
                y_pred = np.array(preds)
            return y_true, y_pred

        def eval_model(self, y, y_pred):
            print("------------Model Evaluation Score-----------")
            results = precision_recall_fscore_support(y, y_pred, average='macro')
            accuracy = accuracy_score(y, y_pred)
            print("Precision: ", results[0],"\nRecall: ", results[1], "\nF1: ", results[2], '\nAccuracy: ', accuracy)
    
    # load data
    X_train, y_train, X_test, y_test  = load_data(args.featurefile)

    # Creating the model
    ffnn = DocFFNN(X_train, y_train)
    
    # Training the model
    print("Training the model..")
    ffnn.train(X_train)
    
    # Testing the model...
    print("Predicting outputs..")
    y_true, y_pred = ffnn.predict(X_test,y_test)

    # Evaluating the model
    ffnn.eval_model(y_true, y_pred)
    
