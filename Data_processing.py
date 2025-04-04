import kagglehub
import os, shutil
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None) # to display the large dataset

# Download latest version
path = kagglehub.dataset_download("kento731/housing-price-dubai-uae")
kaggle_filename = "Housing Price Dubai UAE.csv"

if not(os.path.exists("Datasets")):
    os.mkdir("Datasets")

Dataset_path = shutil.copy(os.path.join(path,kaggle_filename),"Datasets")
print("Loaded dataset:", Dataset_path)



Dataframe = pd.read_csv(Dataset_path)
Dataframe.info()

# remove duplicates
Dataframe = Dataframe.drop_duplicates()
# Dataframe.shape
# no duplicates found

# remove data that might cause leaks in training
Dataframe = Dataframe.drop(columns = "price_per_sqft")

# remove impact-less data
Dataframe = Dataframe.drop(columns = ["latitude","longitude"])

# remove rows with invalid values (info() shows no null, but let's run dropna anyways)
Dataframe.dropna(axis=0)

# inspect strings
pd.set_option('display.max_rows', None) # temporarily display all rows

pd.Series.value_counts(Dataframe["neighborhood"]) 
Dataframe["neighborhood"].nunique()
Dataframe["neighborhood"].sort_values()
# seems fine, 54 categories, no mismatch in naming.

pd.Series.value_counts(Dataframe["quality"]) 
# seems fine, 4 categories.

# No need for stripping.
pd.set_option('display.max_rows', 10) # default rows display


# replacing Strings with Numerics giving apporoximate advantage by area/class.
# take the neighborhoods slice from the dataframe sorted by prices and covert it to list
neighborhoods = Dataframe[["neighborhood", "price"]].sort_values("price", ascending=True)
neighborhoods = neighborhoods["neighborhood"].unique().tolist()

# generate neighborhood_values and replace in dataframe
neighborhood_values = np.arange(len(neighborhoods))
Dataframe = Dataframe.replace(neighborhoods,neighborhood_values)  ##downcasting warning expected##

# replacing 'quality'
qualities = ["Low", "Medium", "High", "Ultra"]
Dataframe = Dataframe.replace(qualities,[0, 1, 2, 3])

# replacing True and False
Dataframe = Dataframe.replace([False, True],[0, 1]) ##downcasting warning expected##
Dataframe

