#!/usr/bin/env python
# coding: utf-8

##################################
# about this code:
# data exploration, cleaning and processing stage of a real estate price\
# dateset to build a prediction Neural network.
##################################


# load housing dataset
import kagglehub
import os, shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download("kento731/housing-price-dubai-uae")
kaggle_filename = "Housing Price Dubai UAE.csv"
if not(os.path.exists("Datasets")):
    os.mkdir("Datasets")
Dataset_path = shutil.copy(os.path.join(path,kaggle_filename),"Datasets")
print("Loaded dataset:", Dataset_path)




pd.set_option('display.max_columns', None) # to display the large dataset.


# ## 1. Data Exploration/Visualization



#load dataset from storage
Dataframe = pd.read_csv(Dataset_path)
print(Dataframe.shape)
Dataframe.info()




# finding missing values
pd.set_option('display.max_rows', None)
Dataframe.isnull().sum()
Dataframe.isna().sum()
pd.set_option('display.max_rows', 10)
# No missing values found.

# inspecting Numerics
Dataframe.hist(bins=50, figsize=(16,8))




# use Seaborn boxplot to find outliers
for i in Dataframe.select_dtypes(include="number").columns:
    sns.boxplot(data=Dataframe, x=i)
    # plt.show()
    # # continious features show a similar behaviour thus no treatment is done to perserve accuracy.
    # # ID is not used in training thus no treatment is need.

# # pair plot to show how features scatter in relation to price
# sns.pairplot(Dataframe) # HEAVY ON PROCESSOR, USE ALTERNATIVE:
Slice = Dataframe.iloc[:, 4:9]
sns.pairplot(Slice)

plt.show()


# # 2. Data value cleaning and preparation



# remove duplicates
Dataframe = Dataframe.drop_duplicates()
Dataframe.shape
# no duplicates found.

# we could remove "price_per_sqft" column to avoid leaks in machine learning training, but let's keep it for now.
# Dataframe = Dataframe.drop(columns = "price_per_sqft")

# remove impactless data
Dataframe = Dataframe.drop(columns = ["latitude","longitude"])

# # remove rows with invalid values (but earlier inspection shows no null)
# Dataframe.dropna(axis=0)




Dataframe




# inspect strings
pd.set_option('display.max_rows', None)

pd.Series.value_counts(Dataframe["neighborhood"]) 
Dataframe["neighborhood"].nunique()
Dataframe["neighborhood"].sort_values()
# seems fine, 54 categories, no mismatch in naming.

pd.Series.value_counts(Dataframe["quality"]) 
# seems fine, 4 categories.

# No need for stripping.
pd.set_option('display.max_rows', 10)


# ## 2.1 Data Encoding



# replacing Strings with Numerics giving apporoximate advantage by area/class.

# take the neighborhoods slice from the dataframe sorted by prices and covert it to list
neighborhoods = Dataframe[["neighborhood", "price"]].sort_values("price", ascending=True)
neighborhoods = neighborhoods["neighborhood"].unique().tolist()
# generate neighborhood_values and replace in dataframe
neighborhood_values = np.arange(len(neighborhoods))
Dataframe = Dataframe.replace(neighborhoods,neighborhood_values)  ##downcasting warning expected##
print(Dataframe["neighborhood"])
qualities = ["Low", "Medium", "High", "Ultra"]
Dataframe = Dataframe.replace(qualities,[0, 1, 2, 3])




# replacing True and False
Dataframe = Dataframe.replace([False, True],[0, 1]) ##downcasting warning expected##
Dataframe


# ## 2.2 Final Inspection



Dataframe.hist(column=["price", "size_in_sqft", "price_per_sqft", "price_per_sqft", "no_of_bedrooms", "quality"],  bins=50, figsize=(20,10), layout=(2,3))
# The distribution sounds about right.




# seaborn heatmap to analize correlation between features
data_heat = Dataframe.corr()
plt.figure(figsize=(20,20))
sns.heatmap(data_heat,annot=True)
# Upon inspection, correlation seems to be logical.


# # 3. Feature Scaling/Normalization