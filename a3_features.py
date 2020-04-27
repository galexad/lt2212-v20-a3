import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.utils import shuffle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.
    basepath = Path(args.inputdir)

    def get_files(samples):
        dirs_in_basepath = (entry for entry in samples.iterdir() if entry.is_dir())
        authors = []
        sample_list=[]
        my_dict={}
        for directory in dirs_in_basepath:
            subdirs =  (e for e in directory.iterdir() if directory.is_dir())
            stop_words = set(stopwords.words('english'))
            for filename in subdirs:
                authors.append(directory)
                with open(filename, "r") as thefile:
                    thefile = list(thefile)
                    thefile = ' '.join(thefile)
                    thefile = word_tokenize(thefile)
                    thefile = [''.join(c.lower() for c in s if c!="") for s in thefile if s not in string.punctuation if s.isalpha() is True]
                    f_sample = [w for w in thefile if not w in stop_words]
                    thefile = ' '.join(f_sample)
                    sample_list.append(thefile)
                
        return sample_list, authors
    
    def extract_features(samples):
        sample_list,authors = get_files(samples)
        tfidf = TfidfVectorizer()
        feature_matrix = tfidf.fit_transform(sample_list)

        return feature_matrix.toarray()
    

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    
    def reduce_dim(X, n=10):
        svd = TruncatedSVD(n)
        transformed = svd.fit_transform(X)
        return transformed

    def shuffle_split(X):
        df = pd.DataFrame(X)
        le = preprocessing.LabelEncoder()
        df.insert(0,'Authors', pd.Series(get_files(basepath)[1]).values)
        df['Authors'] = le.fit_transform(df['Authors'])
        data_labels = []
        df= shuffle(df)
        df.reset_index(inplace=True, drop=True)
        
        n = 100 - args.testsize
        m = args.testsize
        
        x = df.head(int(len(df)*(n/100)))
        y = df.tail(int(len(df)*(m/100)))
        for i in range(len(x)):
            data_labels.append(0)
        for j in range(len(y)):
            data_labels.append(1)
    
        df.insert(0,'Train/test', pd.Series(data_labels))
        return df

    X1 = extract_features(basepath)
    X2 = reduce_dim(X1, args.dims)
    
    print("Writing to {}...".format(args.outputfile))
    with open(args.outputfile, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
        df = shuffle_split(reduce_dim((extract_features(basepath)), args.dims))
        df.to_csv("outputfile.csv", index=False)
    
    
    print("Done!")

