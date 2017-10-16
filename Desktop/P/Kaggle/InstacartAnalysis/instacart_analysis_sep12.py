# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:13:29 2017

@author: a613274

<Result>
order_id,products
17,39276 29259
34,39276 29259
137,39276 29259
182,39276 29259
257,39276 29259
313,39276 29259

<Goal> 
customer orders over time to predict which previously purchased products will be in a user’s next order. 
You are predicting reordered items only for the test set orders.

Q. Where this order id comes? in test or ?? 

Q. What is the difference between order_product_prior and order_product_train

"""

#---------------------
# Import libraries 
#---------------------
import os
import pandas as pd
import numpy as np


#-----------------------------------------
# readCSV
# Parameter : directory(path), filename
#-----------------------------------------
def readCSV(filePath, fileName): 
    csv_path = os.path.join(filePath,fileName)
    return pd.read_csv(csv_path)

#---------------
# Load Data 
#---------------
DATA_PATH = "C:/Users/a613274/Kaggle/InstacartAnalysis/data"
AISLES_FILE = "aisles.csv"
DEPARTMENT_FILE = "departments.csv"
ORDER_PRIOR_FILE = "order_products__prior.csv"
ORDER_TRAIN_FILE = "order_products__train.csv"
ORDERS_FILE = "orders.csv"
PRODUCT_FILE = "products.csv"
EXAMPLE_FILE = "sample_submission.csv"


"""
aisles_path = os.path.join(DATA_PATH, AISLES_FILE)
print(aisles_path)
aisles_data = open(aisles_path, 'rb')
aisles = pd.read_csv(aisles_data)
print(aisles.head())
aisles.info()
"""

#---------------------
# print plain data
#---------------------
aisles_data = readCSV(DATA_PATH, AISLES_FILE)
print(aisles_data)

dept_data = readCSV(DATA_PATH, DEPARTMENT_FILE)
print(dept_data)

product_data = readCSV(DATA_PATH, PRODUCT_FILE)
print(product_data)

#--------------------------
# print sample submission
# 'order_id' 'products' 
sample_data = readCSV(DATA_PATH, EXAMPLE_FILE)
print(sample_data) #75000

#-----------------------------------------------------------------------------
# order_products__prior.csv contains previous order contents for all customers. 
#'reordered' indicates that the customer has a previous order that contains the product.
orders_product_prior = readCSV(DATA_PATH, ORDER_PRIOR_FILE) #order_id, product_id, add_to_cart_order, reordered
print(orders_product_prior) #32434488

orders_product_train = readCSV(DATA_PATH, ORDER_TRAIN_FILE) #order_id, product_id, add_to_cart_order, reordered
print(orders_product_train) #1384617

#-------------------------------------------------------------------
# This file tells to which set (prior, train, test) an order belongs. 
# You are predicting reordered items only for the test set orders. 'order_dow' is the day of week.
orders_data = readCSV(DATA_PATH, ORDERS_FILE) #order_id, user_id, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order
print(orders_data.info()) #3421083
print(orders_data.head())
print(type(orders_data))
#ToDo : divide orders_data using eval_set as 'prior', 'train', 'test' 
#print(len(orders_data))

orders_test_data = orders_data.loc[orders_data['eval_set'] == 'test'] #75000
print(type(orders_test_data))

orders_train_data = orders_data.loc[orders_data['eval_set'] == 'train'] #131209
print(orders_train_data)

orders_prior_data = orders_data.loc[orders_data['eval_set'] == 'prior'] #3214874
print(orders_prior_data)

"""
#---------------------------------------------------------
# order test data to check the 'order_id' is same as sample 
from pandas import DataFrame
print(type(orders_test_data))
df = DataFrame(orders_test_data)
ordered = df.sort_values(['order_id'], ascending=True)

print(ordered)
"""

def orderedDataFrameWithOrderId(df):
    ordered_df = df.sort_values(['order_id'], ascending=True)
    return ordered_df 
    

#----------------------------------------------------------
# join data frame for order data and order & product data
#----------------------------------------------------------
order_prior_merged = pd.merge(orders_prior_data, orders_product_prior, on='order_id', how='inner')
order_prior_merged.head #32434489
orderedDataFrameWithOrderId(order_prior_merged) #out of memory error

order_train_merged = pd.merge(orders_train_data, orders_product_train, on='order_id', how='inner')
order_train_merged #1384617
orderedDataFrameWithOrderId(order_train_merged)

order_test_merged = pd.merge(orders_prior_data, orders_test_data, on='order_id', how='inner')
order_test_merged #here - it is empty data

"""
#create dataframe of order_test 
raw_order_test = {'order_id': [np.nan], 'product_id': [np.nan], 'add_to_cart_order': [np.nan], 'reordered': [np.nan] }
print(type(raw_order_test))

order_test = pd.DataFrame(raw_order_test, columns = ['order_id', 'product_id', 'add_to_cart_order', 'reordered' ])
print(type(order_test))
pd.concat([order_test, orders_test_data], axis=1) #horizontally concatenate
pd.merge(order_test, orders_test_data, on='order_id', how='outer')
"""

#---------------------------------------------
# list as value (e.g. 'order_id' 'products')
ordered_order_train_merged = orderedDataFrameWithOrderId(order_train_merged)
ordered_order_train_merged
ordered_order_train_merged.groupby('order_id')['product_id'].agg({'list':(lambda x: list(x))})
