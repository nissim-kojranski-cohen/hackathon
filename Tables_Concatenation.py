#!/usr/bin/env python
# coding: utf-8

# # - Tables Concatenation

# In[1]:


import pandas as pd


# In[2]:


demographics = pd.read_csv('demographics.csv')
electricity = pd.read_csv('electricity.csv')
water = pd.read_csv('water.csv')
arnona = pd.read_csv('arnona.csv')

dataframes = [demographics, electricity, water, arnona]


# In[3]:

#
# for df in dataframes:
#     df = df.drop('Unnamed: 0', axis=1)


# In[4]:


# def add_prefix(df, )
months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'Novemeber','December']
prefixed_months_elec = ['Elec_'+month for month in months]
prefixed_months_water = ['Water_'+month for month in months]

dict_elec = {month: prefixed_month for month,prefixed_month in zip(months, prefixed_months_elec)}
dict_water = {month: prefixed_month for month,prefixed_month in zip(months, prefixed_months_water)}


# In[5]:


electricity.rename(columns = dict_elec, inplace=True)
water.rename(columns = dict_water, inplace=True)
arnona.rename(columns = {'price': 'Arnona'}, inplace=True)


# In[6]:


df_merged = demographics.merge(electricity, how='outer')
df_merged = df_merged.merge(water, how='left')
df_merged = df_merged.merge(arnona, how='left')

if 'Unnamed: 0' in df_merged.columns:
    df_merged.drop('Unnamed: 0', axis=1, inplace=True)


# In[7]:


df_merged.to_csv('merged.csv', index=False)

