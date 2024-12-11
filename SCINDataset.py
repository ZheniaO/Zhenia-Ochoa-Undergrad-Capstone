import collections
import copy
import hashlib
import io
import os
import subprocess
import textwrap
import time
import packaging 

from typing import List, Text

from PIL import Image

import numpy as np
import pandas as pd

#import tensorflow as tf

import matplotlib.pyplot as plt


#@title Global params

dataloc = 'D:/dataset'



class Globals:
  # GCP project with GCS bucket of interest
  gcp_project = 'dx-scin-public' #@param

  # GCS bucket with data to read
  gcs_bucket_name = 'dx-scin-public-data' #@param

  # CSV of case metadata to read
  cases_csv = dataloc + '/scin_cases.csv' #@param

  # CSV of label metadata to read
  labels_csv = dataloc + '/scin_labels.csv' #@param

  # Images directory
  gcs_images_dir = dataloc + '/images/' #@param

  ### Key column names
  image_path_columns = ['image_1_path', 'image_2_path', 'image_3_path']
  weighted_skin_condition_label = "weighted_skin_condition_label"
  skin_condition_label = "dermatologist_skin_condition_on_label_name"

  ###### Formed during execution:

  # Client for querying GCS
  gcs_storage_client = None

  # Bucket object for loading files
  gcs_bucket = None

  # pd.DataFrame for the loaded metadata_csv
  cases_df = None

  # pd.DataFrame for the loaded labels_csv
  cases_and_labels_df = None

print(f'GCS bucket name: {Globals.gcs_bucket_name}')
print(f'cases_csv: {Globals.cases_csv}')
print(f'labels_csv: {Globals.labels_csv}')
print(f'images dir: {Globals.gcs_images_dir}')

''''
def list_blobs(storage_client, bucket_name):
  """Helper to list blobs in a bucket (useful for debugging)."""
  blobs = storage_client.list_blobs(bucket_name)
  for blob in blobs:
    print(blob)'''

def initialize_df_with_metadata(csv_path):
  """Loads the given CSV into a pd.DataFrame."""
  df = pd.read_csv(csv_path, dtype={'case_id': str})
  df['case_id'] = df['case_id'].astype(str)
  return df

def augment_metadata_with_labels(df, csv_path):
  """Loads the given CSV into a pd.DataFrame."""
  labels_df = pd.read_csv(csv_path, dtype={'case_id': str})
  labels_df['case_id'] = labels_df['case_id'].astype(str)
  merged_df = pd.merge(df, labels_df, on='case_id')
  return merged_df
'''
Globals.gcs_storage_client = storage.Client(Globals.gcp_project)
Globals.gcs_bucket = Globals.gcs_storage_client.bucket(
    Globals.gcs_bucket_name
)'''
Globals.cases_df = initialize_df_with_metadata(Globals.cases_csv)
Globals.cases_and_labels_df = augment_metadata_with_labels(Globals.cases_df, Globals.labels_csv)
print(len(Globals.cases_and_labels_df))

Globals.cases_and_labels_df.columns

Globals.cases_and_labels_df.sample(1)

#@title Display the images for a case (and condition labels, optionally)
import random

import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

'''def display_image(image_path):
  image = Image.open(image_path)
  figure_size=4
  f, axarr = plt.subplots(1, 1, figsize = (figure_size, figure_size))
  axarr.imshow(image, cmap='gray')
  axarr.axis('off')
  plt.show()

def display_images_for_case(df, case_id="", print_condition_labels=True):
  # Use a random case if none is provided:
  if case_id:
    matched_df = df[df['case_id'] == case_id]
  else:
    matched_df = df.sample(1)

  image_paths = matched_df[Globals.image_path_columns].values.tolist()[0]
  for path in image_paths:
    
    if isinstance(path, str):
      path = "D:/" + path
      display_image(path)
  if print_condition_labels:
    condition_labels = matched_df[[Globals.weighted_skin_condition_label]].values.tolist()[0]
    for label in condition_labels:
      if isinstance(label, str):
        print(label)'''

# display_images_for_case(Globals.cases_and_labels_df, "-1000600354148496558")
# display_images_for_case(Globals.cases_and_labels_df)


all_data = Globals.cases_and_labels_df[[Globals.image_path_columns[0], Globals.skin_condition_label]]
all_data = all_data.rename(columns={Globals.image_path_columns[0]:"filename"})

mask = all_data[Globals.skin_condition_label].str.contains("Eczema")
all_data["has_eczema"] = 0
all_data.loc[mask, "has_eczema"] = 1
print(all_data)


#print(all_data[all_data["has_eczema"] == 1])