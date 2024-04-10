#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import cartopy 
import geopandas
import os
import re


# In[553]:


url = 'https://www.spc.noaa.gov/products/outlook/archive/2024/KWNSPTSDY1_202404101300.txt'
# Give the URL for the ACO file to read from the SPC.


# In[554]:


df = pd.read_csv(url, sep='\t', header=None)


# In[555]:


# Define start and end phrases
start_phrase = "... TORNADO ..."
end_phrase = "&&"

# Find rows containing phrases
start_mask = df[0].str.contains(start_phrase).idxmax()
# Find the first occurrence of rows containing the end phrase
end_mask = df[0].str.contains(end_phrase)

# Filter for the first occurrence of "End Marker" using cumulative sum
end_mask = (end_mask & (~end_mask.cumsum().gt(1))).idxmax()  # Ensures only first True


# Extract the DataFrame slice (excluding the rows with phrases)
tor_df = df[start_mask+1:end_mask]


# In[557]:


def convert_to_coordinates(encoded):
    lat_str = encoded[:4]
    lon_str = encoded[4:]
    latitude = float(lat_str)/100
    longitude = -float(lon_str)/100
    return latitude, longitude


# In[558]:


### Create 2% dataframe, if a 0.02 exists.
phrase = "0.02"
end = "0.05"
sign = "SIGN"

if tor_df[0].str.contains(phrase).any():
    start_mask_2_perc = tor_df[0].str.contains(phrase, regex=False).idxmax()

    # Find rows containing "end_phrase" (if it exists)
    end_mask_2_perc = tor_df[0].str.contains(end, regex=False).idxmax()

    # Handle cases where "end_phrase" is not found
    if end_mask_2_perc <= start_mask_2_perc:
        end_mask_2_perc = tor_df[0].str.contains("SIGN", refex=False).idxmax()
      # If no "end_phrase" found, use the last row index
        if end_mask_2_perc <= start_mask_2_perc:
            end_mask_2_perc = tor_df[0].index.stop


    df_2_perc_pre = df[start_mask_2_perc:end_mask_2_perc]
    all_elements_2_perc = pd.Series([item for row in df_2_perc_pre[0] for item in row.split(' ')])
    values_to_delete_2_perc = ['0.02']

    # Create a boolean mask for elements to keep
    mask = ~all_elements_2_perc.isin(values_to_delete_2_perc)

    # Delete elements using boolean indexing and .drop()
    filtered_series = all_elements_2_perc.drop(all_elements_2_perc.index[~mask], axis=0) # Optional: reset index
    series = filtered_series.tolist()
    str_list = list(filter(None, series))
    coords_2_perc = [convert_to_coordinates(coords) for coords in str_list]
    
    


# In[559]:


### Create 5% dataframe, if a 0.05 exists.
phrase = "0.05"
end = "0.10"
sign = "SIGN"

if tor_df[0].str.contains(phrase).any():
    start_mask_5_perc = tor_df[0].str.contains(phrase, regex=False).idxmax()

    # Find rows containing "end_phrase" (if it exists)
    end_mask_5_perc = tor_df[0].str.contains(end, regex=False).idxmax()

    # Handle cases where "end_phrase" is not found
    if end_mask_5_perc <= start_mask_5_perc:
        end_mask_5_perc = tor_df[0].str.contains("SIGN", regex=False).idxmax()
      # If no "end_phrase" found, use the last row index
        if end_mask_5_perc <= start_mask_5_perc:
            end_mask_5_perc = tor_df[0].index.stop


    df_5_perc_pre = df[start_mask_5_perc:end_mask_5_perc]
    all_elements_5_perc = pd.Series([item for row in df_5_perc_pre[0] for item in row.split(' ')])
    values_to_delete_5_perc = ['0.05']

    # Create a boolean mask for elements to keep
    mask = ~all_elements_5_perc.isin(values_to_delete_5_perc)

    # Delete elements using boolean indexing and .drop()
    filtered_series = all_elements_5_perc.drop(all_elements_5_perc.index[~mask], axis=0) # Optional: reset index
    series = filtered_series.tolist()
    str_list = list(filter(None, series))
    coords_5_perc = [convert_to_coordinates(coords) for coords in str_list]
    
    


# In[560]:


### Create 10% dataframe, if a 0.10 exists.
phrase = "0.10"
end = "0.15"
sign = "SIGN"

if tor_df[0].str.contains(phrase).any():
    start_mask_10_perc = tor_df[0].str.contains(phrase, regex=False).idxmax()

    # Find rows containing "end_phrase" (if it exists)
    end_mask_10_perc = tor_df[0].str.contains(end, regex=False).idxmax()

    # Handle cases where "end_phrase" is not found
    if end_mask_10_perc <= start_mask_10_perc:
        end_mask_10_perc = tor_df[0].str.contains("SIGN", regex=False).idxmax()
      # If no "end_phrase" found, use the last row index
        if end_mask_10_perc <= start_mask_10_perc:
            end_mask_10_perc = tor_df[0].index.stop


    df_10_perc_pre = df[start_mask_10_perc:end_mask_10_perc]
    all_elements_10_perc = pd.Series([item for row in df_10_perc_pre[0] for item in row.split(' ')])
    values_to_delete_10_perc = ['0.10']

    # Create a boolean mask for elements to keep
    mask = ~all_elements_10_perc.isin(values_to_delete_10_perc)

    # Delete elements using boolean indexing and .drop()
    filtered_series = all_elements_10_perc.drop(all_elements_10_perc.index[~mask], axis=0) # Optional: reset index
    series = filtered_series.tolist()
    str_list = list(filter(None, series))
    coords_10_perc = [convert_to_coordinates(coords) for coords in str_list]
    
    


# In[561]:


### Create 15% dataframe, if a 0.15 exists.
phrase = "0.15"
end = "0.30"
sign = "SIGN"

if tor_df[0].str.contains(phrase).any():
    start_mask_15_perc = tor_df[0].str.contains(phrase, regex=False).idxmax()

    # Find rows containing "end_phrase" (if it exists)
    end_mask_15_perc = tor_df[0].str.contains(end, regex=False).idxmax()

    # Handle cases where "end_phrase" is not found
    if end_mask_15_perc <= start_mask_15_perc:
        end_mask_15_perc = tor_df[0].str.contains("SIGN").idxmax()
      # If no "end_phrase" found, use the last row index
        if end_mask_15_perc <= start_mask_15_perc:
            end_mask_15_perc = tor_df[0].index.stop


    df_15_perc_pre = df[start_mask_15_perc:end_mask_15_perc]
    all_elements_15_perc = pd.Series([item for row in df_15_perc_pre[0] for item in row.split(' ')])
    values_to_delete_15_perc = ['0.15']

    # Create a boolean mask for elements to keep
    mask = ~all_elements_15_perc.isin(values_to_delete_15_perc)

    # Delete elements using boolean indexing and .drop()
    filtered_series = all_elements_15_perc.drop(all_elements_15_perc.index[~mask], axis=0) # Optional: reset index
    series = filtered_series.tolist()
    str_list = list(filter(None, series))
    coords_15_perc = [convert_to_coordinates(coords) for coords in str_list]
    
    


# In[562]:


### Create 30% dataframe, if a 0.30 exists.
phrase = "0.30"
end = "0.45"
sign = "SIGN"

if tor_df[0].str.contains(phrase).any():
    start_mask_30_perc = tor_df[0].str.contains(phrase, regex=False).idxmax()

    # Find rows containing "end_phrase" (if it exists)
    end_mask_30_perc = tor_df[0].str.contains(end, regex=False).idxmax()

    # Handle cases where "end_phrase" is not found
    if end_mask_30_perc <= start_mask_30_perc:
        end_mask_30_perc = tor_df[0].str.contains("SIGN",regex=False).idxmax()
      # If no "end_phrase" found, use the last row index
        if end_mask_30_perc <= start_mask_30_perc:
            end_mask_30_perc = tor_df[0].index.stop


    df_30_perc_pre = df[start_mask_30_perc:end_mask_30_perc]
    all_elements_30_perc = pd.Series([item for row in df_30_perc_pre[0] for item in row.split(' ')])
    values_to_delete_30_perc = ['0.30']

    # Create a boolean mask for elements to keep
    mask = ~all_elements_30_perc.isin(values_to_delete_30_perc)

    # Delete elements using boolean indexing and .drop()
    filtered_series = all_elements_30_perc.drop(all_elements_30_perc.index[~mask], axis=0) # Optional: reset index
    series = filtered_series.tolist()
    str_list = list(filter(None, series))
    coords_30_perc = [convert_to_coordinates(coords) for coords in str_list]
    
    


# In[563]:


### Create 45% dataframe, if a 0.45 exists.
phrase = "0.45"
end = "0.60"
sign = "SIGN"

if tor_df[0].str.contains(phrase).any():
    start_mask_45_perc = tor_df[0].str.contains(phrase, regex=False).idxmax()

    # Find rows containing "end_phrase" (if it exists)
    end_mask_45_perc = tor_df[0].str.contains(end, regex=False).idxmax()

    # Handle cases where "end_phrase" is not found
    if end_mask_45_perc <= start_mask_45_perc:
        end_mask_45_perc = tor_df[0].str.contains("SIGN", regex=False).idxmax()
      # If no "end_phrase" found, use the last row index
        if end_mask_45_perc <= start_mask_45_perc:
            end_mask_45_perc = tor_df[0].index.stop


    df_45_perc_pre = df[start_mask_45_perc:end_mask_45_perc]
    all_elements_45_perc = pd.Series([item for row in df_45_perc_pre[0] for item in row.split(' ')])
    values_to_delete_45_perc = ['0.45']

    # Create a boolean mask for elements to keep
    mask = ~all_elements_45_perc.isin(values_to_delete_45_perc)

    # Delete elements using boolean indexing and .drop()
    filtered_series = all_elements_45_perc.drop(all_elements_45_perc.index[~mask], axis=0) # Optional: reset index
    series = filtered_series.tolist()
    str_list = list(filter(None, series))
    coords_45_perc = [convert_to_coordinates(coords) for coords in str_list]
    
    


# In[564]:


### Create 60% dataframe, if a 0.60 exists.
phrase = "0.60"
end = "0.60"
sign = "SIGN"

if tor_df[0].str.contains(phrase).any():
    start_mask_60_perc = tor_df[0].str.contains(phrase, regex=False).idxmax()

    # Find rows containing "end_phrase" (if it exists)
    end_mask_60_perc = tor_df[0].str.contains(end, regex=False).idxmax()

    # Handle cases where "end_phrase" is not found
    if end_mask_60_perc <= start_mask_60_perc:
        end_mask_60_perc = tor_df[0].str.contains("SIGN", regex=False).idxmax()
      # If no "end_phrase" found, use the last row index
        if end_mask_60_perc <= start_mask_60_perc:
            end_mask_60_perc = tor_df[0].index.stop


    df_60_perc_pre = df[start_mask_60_perc:end_mask_60_perc]
    all_elements_60_perc = pd.Series([item for row in df_60_perc_pre[0] for item in row.split(' ')])
    values_to_delete_60_perc = ['0.60']

    # Create a boolean mask for elements to keep
    mask = ~all_elements_60_perc.isin(values_to_delete_60_perc)

    # Delete elements using boolean indexing and .drop()
    filtered_series = all_elements_60_perc.drop(all_elements_60_perc.index[~mask], axis=0) # Optional: reset index
    series = filtered_series.tolist()
    str_list = list(filter(None, series))
    coords_60_perc = [convert_to_coordinates(coords) for coords in str_list]
    
    


# In[565]:


### Create SIG dataframe, if a SIG exists.
phrase = "SIGN"
end = "&&"
sign = "SIGN"

if tor_df[0].str.contains(phrase).any():
    start_mask_SIGN_perc = tor_df[0].str.contains(phrase, regex=False).idxmax()

    # Find rows containing "end_phrase" (if it exists)
    end_mask_SIGN_perc = tor_df[0].str.contains(end, regex=False).idxmax()

    # Handle cases where "end_phrase" is not found
    if end_mask_SIGN_perc != start_mask_SIGN_perc:
        end_mask_SIGN_perc = tor_df[0].str.contains("&&", regex=False).idxmax()
      # If no "end_phrase" found, use the last row index
        if end_mask_SIGN_perc != start_mask_SIGN_perc:
            end_mask_SIGN_perc = tor_df[0].index.stop


    df_SIGN_perc_pre = df[start_mask_SIGN_perc:end_mask_SIGN_perc]
    all_elements_SIGN_perc = pd.Series([item for row in df_SIGN_perc_pre[0] for item in row.split(' ')])
    values_to_delete_SIGN_perc = ['SIGN','&&']

    # Create a boolean mask for elements to keep
    mask = ~all_elements_SIGN_perc.isin(values_to_delete_SIGN_perc)

    # Delete elements using boolean indexing and .drop()
    filtered_series = all_elements_SIGN_perc.drop(all_elements_SIGN_perc.index[~mask], axis=0) # Optional: reset index
    series = filtered_series.tolist()
    str_list = list(filter(None, series))
    coords_SIGN_perc = [convert_to_coordinates(coords) for coords in str_list]
    
    


# In[568]:


import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import cartopy.crs as ccrs

def create_and_plot_polygon(coordinates, projection=ccrs.PlateCarree()):
    crs_proj = projection
    crs_proj4 = crs_proj.proj4_init
    df_plots = geodf.to_crs(crs_proj4)
    return df_plots

coord_df_2 = pd.DataFrame(coords_2_perc)
geodf_2 = geopandas.GeoDataFrame(coords_2_perc,geometry=geopandas.points_from_xy(coord_df_2[1], coord_df_2[0]), crs="EPSG:4326")
perc_2 = create_and_plot_polygon(geodf_2)

coord_df_5 = pd.DataFrame(coords_5_perc)
geodf_5 = geopandas.GeoDataFrame(coords_5_perc,geometry=geopandas.points_from_xy(coord_df_5[1], coord_df_5[0]), crs="EPSG:4326")
perc_5 = create_and_plot_polygon(geodf_5)



coord_df_10 = pd.DataFrame(coords_10_perc)
geodf_10 = geopandas.GeoDataFrame(coords_10_perc,geometry=geopandas.points_from_xy(coord_df_10[1], coord_df_10[0]), crs="EPSG:4326")
perc_10 = create_and_plot_polygon(geodf_10)



coord_df_15 = pd.DataFrame(coords_15_perc)
geodf_15 = geopandas.GeoDataFrame(coords_15_perc,geometry=geopandas.points_from_xy(coord_df_15[1], coord_df_15[0]), crs="EPSG:4326")
perc_15 = create_and_plot_polygon(geodf_15)

import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
import matplotlib.patches as mpatches
# Assuming your GeoPandas DataFrame is called 'gdf'

# Define the desired map projection (optional, defaults to PlateCarree)
projection = ccrs.PlateCarree()

# Create a figure and axes
fig = plt.figure()
ax = fig.add_subplot(projection=projection)

swapped = pd.concat([pd.DataFrame(coords_2_perc)[1],pd.DataFrame(coords_2_perc)[0]],axis=1)
swapped_5 = pd.concat([pd.DataFrame(coords_5_perc)[1],pd.DataFrame(coords_5_perc)[0]],axis=1)
swapped_10 = pd.concat([pd.DataFrame(coords_10_perc)[1],pd.DataFrame(coords_10_perc)[0]],axis=1)
swapped_15 = pd.concat([pd.DataFrame(coords_15_perc)[1],pd.DataFrame(coords_15_perc)[0]],axis=1)
swapped_SIGN = pd.concat([pd.DataFrame(coords_SIGN_perc)[1],pd.DataFrame(coords_SIGN_perc)[0]],axis=1)




# Plot the geometry column from the GeoDataFrame on the axes with the projection
perc_2.plot(ax=ax, column='geometry', facecolor='lightgreen', linewidth=0, alpha=0)
#perc_5.plot(ax=ax, column='geometry', facecolor='lightgreen', linewidth=0)

# Add features like coastlines, background, and title (optional)
ax.add_feature(cfeature.COASTLINE,zorder=10)
ax.add_feature(cfeature.STATES,zorder=10)
ax.add_feature(cfeature.OCEAN,zorder=10)
poly = mpatches.Polygon(swapped, closed=True, fill=True, lw=1, fc="#7bbe7b", transform=projection,zorder=3)
ax.add_patch(poly)
poly_5 = mpatches.Polygon(swapped_5, closed=True, fill=True, lw=1, fc="#c9a388", transform=projection,zorder=4)
ax.add_patch(poly_5)
poly_10 = mpatches.Polygon(swapped_10, closed=True, fill=True, lw=1, fc="#f5e17a", transform=projection,zorder=5)
ax.add_patch(poly_10)
poly_15 = mpatches.Polygon(swapped_15, closed=True, fill=True, lw=1, fc="#f05f5f", transform=projection,zorder=6)
ax.add_patch(poly_15)
ax.gridlines(draw_labels=True, zorder=10)
ax.grid(color='gray', linestyle='--', linewidth=0.5)  # Adjust color, style, and width


#######################################
########### Plot Tornadoes #############



import random
from shapely.geometry import Polygon, Point

def random_point_in_polygon_with_probability(polygon_coords, probability):

  # Convert polygon coordinates to Shapely Polygon object
  polygon = Polygon(polygon_coords)
  polygon_SIG = Polygon(list(zip(swapped_SIGN[1], swapped_SIGN[0])))

  max_attempts = int(round(50*np.random.exponential(1),0))
  points = []
  for _ in range(max_attempts):
    # Generate random point within bounding box of the polygon
    x_min, y_min, x_max, y_max = polygon.bounds
    random_x = random.uniform(x_min, x_max)
    random_y = random.uniform(y_min, y_max)
    random_point = Point(random_x, random_y)

    # Check if the point is within the polygon
    if polygon.contains(random_point):
      # Generate random float between 0 and 1
      plot_chance = random.random()
      
      # Plot the point based on probability
      if plot_chance <= probability:
            if polygon_SIG.contains(random_point) and plot_chance <= probability**2:
                points.append([random_x,random_y,"white"])
            else:
                points.append([random_x,random_y,"orange"])
          # Return the random point with probability
        #print(points)
  return pd.DataFrame(points)

# Example usage (replace with your polygon coordinates and desired probability)
#polygon_coords = [(0, 0), (10, 0), (10, 5), (0, 5)]
#probability = 0.7  # Adjust probability (0.0 to 1.0)

random_point_2 = random_point_in_polygon_with_probability(list(zip(swapped[1], swapped[0])), 0.02)
if len(random_point_2) >= 1:
     ax.scatter(random_point_2[0],random_point_2[1], zorder=11, color=random_point_2[2])
random_point_5 = random_point_in_polygon_with_probability(list(zip(swapped_5[1], swapped_5[0])), 0.05)
if len(random_point_5) >= 1:
     ax.scatter(random_point_5[0],random_point_5[1], zorder=11, color=random_point_5[2])
random_point_10 = random_point_in_polygon_with_probability(list(zip(swapped_10[1], swapped_10[0])), 0.10)
if len(random_point_10) >= 1:
    ax.scatter(random_point_10[0],random_point_10[1], zorder=11, color=random_point_10[2])
random_point_15 = random_point_in_polygon_with_probability(list(zip(swapped_15[1], swapped_15[0])), 0.15)
if len(random_point_15) >= 1:
     ax.scatter(random_point_15[0],random_point_15[1], zorder=11, color=random_point_15[2])


##########################
##########################


# Show the plot"
plt.show()


# In[ ]:




