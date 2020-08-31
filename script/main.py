# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 18:05:46 2020

@author: Jayola Gbenga

"""
from data import data_loader, resample_data, split
from model import classifiers_cv, evaluation, Logisitic

def main():
    
    print("load csv into a pandas dataframe")
    dt= data_loader()
    data = dt.load_data()
    print(f"data has {data.shape}")
    data = dt.encode_target(data)
    print("preprocess data by removing outliers and encoding feature variables")
    data = dt.preprocess(data)
    #print(data.columns)
    print("scale data using standardscaler and encoding using pandas get_dummies")
    data = dt.scale_columns(data)
    print(f"data contains {data.columns}")
    
    sam = resample_data()
    data = sam.under_sample(data)
    
    print(data['y'].value_counts())

    s = split()
    data = s.train_test(data)
    print(data[0].shape)
    
    classifiers_cv(data[0], data[1], data[2], data[3])

if __name__ == "__main__":
    main()
    
    

