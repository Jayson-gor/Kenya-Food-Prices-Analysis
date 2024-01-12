#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imporing and reading the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DF = pd.read_csv(r"C:\Users\gorja\Downloads\wfp_food_prices_ken.csv")
DF


# In[4]:


#columns and their data structure
DF.dtypes


# In[6]:


#cleaning data and copying it to new csv file
DF = DF.dropna()
Kenya_DF = DF[['date', 'admin2', 'market', 'category', 'commodity', 'pricetype','price']]
Kenya_DF.to_csv('Cleaned_Kenya_DF.csv', index = False)


# In[8]:


#reading the cleaned data
New_df = pd.read_csv(r"C:\Users\gorja\OneDrive\Documents\Desktop\Test\Cleaned_Kenya_DF.csv")
New_df


# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
from datetime import datetime

# Assuming 'New_df' is your DataFrame
New_df = New_df.iloc[1:]

# Changing the date column to date
New_df['date'] = pd.to_datetime(New_df['date'])
New_df['price'] = pd.to_numeric(New_df['price'], errors='coerce')

# confirming data types
New_df.dtypes



# In[24]:


#confirming import matplotlib.pyplot as plt

# Assuming 'New_df' is your DataFrame
# Make sure 'date' column is of datetime type
New_df['year'] = New_df['date'].dt.year
New_df['price'] = pd.to_numeric(New_df['price'], errors='coerce')


# Calculate average price per year
average_price_per_year = New_df.groupby('year')['price'].mean()

# Plotting as a line chart
plt.figure(figsize=(10, 6))
plt.plot(average_price_per_year.index, average_price_per_year, marker='o', linestyle='-')
plt.title('Average Price Per Year in KSH')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.grid(True)
plt.show()


# In[114]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'New_df' is your DataFrame
# Make sure 'date' column is of datetime type
New_df['year'] = New_df['date'].dt.year

# Filter data up to the year 2022
filtered_df = New_df[New_df['year'] <= 2022]

# Calculate average price per year
average_price_per_year = filtered_df.groupby('year')['price'].mean().reset_index()

# Display the table
print(average_price_per_year)

# Plotting as a line chart
plt.figure(figsize=(10, 6))
plt.plot(average_price_per_year['year'], average_price_per_year['price'], marker='o', linestyle='-')
plt.title('Average Price Per Year')
plt.xlabel('Year')
plt.ylabel('Average Price in Ksh')
plt.grid(True)
plt.show()


# In[117]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'New_df' is your DataFrame
# Make sure 'date' column is of datetime type
New_df['year'] = New_df['date'].dt.year

# Filter data up to the year 2022
filtered_df = New_df[New_df['year'] == 2022]

# Calculate average price per year
average_price_per_year = filtered_df.groupby(['year','admin2'])['price'].mean().reset_index()

# Display the table
print(average_price_per_year)




# In[118]:


# now top 10 counties by average price
# Sort the DataFrame by 'price' in descending order and print the top 20
top_20_max_prices = average_price_per_year.sort_values(by='price', ascending=False).head(10)
print(top_20_max_prices)


#plotting a bar graph
x = top_20_max_prices['admin2']
y = top_20_max_prices['price']

plt.figure(figsize=(10, 6))
plt.title('Top 10 Counties on Average Price in 2023')
plt.xlabel('County')
plt.ylabel('Average Price in 2022')
plt.legend
plt.bar(x, y)
plt.xticks(rotation='vertical')
plt.show()



# In[66]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'New_df'
# Ensure 'date' column is in datetime format
New_df['date'] = pd.to_datetime(New_df['date'])

# Extract the year and create a new column 'year'
New_df['year'] = New_df['date'].dt.year

# Drop rows with missing values
New_df = New_df.dropna(subset=['admin2', 'year', 'price'])

# Filter data for the years 2021 and 2022
df_2021_2022 = New_df[New_df['year'].isin([2021, 2022])]


# Group data by county and calculate the average price for each year
avg_prices = df_2021_2022.groupby(['admin2', 'year'])['price'].mean().reset_index()

# Pivot the data for better visualization
pivot_prices = avg_prices.pivot(index='admin2', columns='year', values='price')
pivot_prices = pivot_prices.dropna()

# Plotting the comparative bar graph
plt.figure(figsize=(10, 6))
pivot_prices.plot.bar(rot=0, width=0.6)
plt.title('Average Food Prices Comparison (2021-2022) by County')
plt.xlabel('County')
plt.ylabel('Average Food Price (Ksh)')
plt.legend(title='Year', bbox_to_anchor=(1, 1))

plt.xticks(rotation='vertical')
#plt.tight_layout()
plt.show()




#printing table
pivot_prices['price_difference'] = pivot_prices[2022] - pivot_prices[2021]
print(pivot_prices)


# In[70]:


#Uncover and analyze any noticeable seasonal patterns in food prices in the leading county Kisumu
import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'New_df'
# Ensure 'date' column is in datetime format
New_df['date'] = pd.to_datetime(New_df['date'])

# Extract the year, month, and create new columns 'year' and 'month'
New_df['year'] = New_df['date'].dt.year
New_df['month'] = New_df['date'].dt.month_name()

# Drop rows with missing values
New_df = New_df.dropna(subset=['admin2', 'year', 'price'])

# Filter data for Kisumu and the years 2021 and 2022
df_kisumu = New_df[(New_df['admin2'] == 'Kisumu') & New_df['year'].isin([2021, 2022])]

# Group data by month and calculate the average price for each year
avg_prices_kisumu = df_kisumu.groupby(['month', 'year'])['price'].mean().reset_index()

# Pivot the data for better visualization
pivot_prices_kisumu = avg_prices_kisumu.pivot(index='month', columns='year', values='price')

# Print the DataFrame with 2021 and 2022 data for Kisumu
print(pivot_prices_kisumu)

# Plotting the comparative bar graph for Kisumu with price difference
plt.figure(figsize=(12, 6))
pivot_prices_kisumu.plot.bar(rot=0, width=0.8)
plt.title('Average Food Prices Comparison (2021-2022) in Kisumu by Month')
plt.xlabel('Month')
plt.ylabel('Average Food Price (Ksh)')
plt.legend(title='Year', bbox_to_anchor=(1, 1))
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.grid(axis = 'y')
plt.show()



# In[71]:


#Uncover and analyze any noticeable seasonal patterns in food prices in the capital county Nairobi
import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'New_df'
# Ensure 'date' column is in datetime format
New_df['date'] = pd.to_datetime(New_df['date'])

# Extract the year, month, and create new columns 'year' and 'month'
New_df['year'] = New_df['date'].dt.year
New_df['month'] = New_df['date'].dt.month_name()

# Drop rows with missing values
New_df = New_df.dropna(subset=['admin2', 'year', 'price'])

# Filter data for Kisumu and the years 2021 and 2022
df_nairobi = New_df[(New_df['admin2'] == 'Nairobi') & New_df['year'].isin([2021, 2022])]

# Group data by month and calculate the average price for each year
avg_prices_nairobi = df_nairobi.groupby(['month', 'year'])['price'].mean().reset_index()

# Pivot the data for better visualization
pivot_prices_nairobi = avg_prices_nairobi.pivot(index='month', columns='year', values='price')

# Print the DataFrame with 2021 and 2022 data for Nairobi
print(pivot_prices_kisumu)

# Plotting the comparative bar graph for Nairobi with price difference
plt.figure(figsize=(12, 6))
pivot_prices_kisumu.plot.bar(rot=0, width=0.8)
plt.title('Average Food Prices Comparison (2021-2022) in Nairobi by Month')
plt.xlabel('Month')
plt.ylabel('Average Food Price (Ksh)')
plt.legend(title='Year', bbox_to_anchor=(1, 1))
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.grid(axis = 'y')
plt.show()



# In[128]:


import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'New_df'
# Ensure 'date' column is in datetime format
New_df['date'] = pd.to_datetime(New_df['date'])

# Extract the year, month, and create new columns 'year' and 'month'
New_df['year'] = New_df['date'].dt.year 

# Filter data up to the year 2022
filtered_df = New_df[New_df['year'] == 2022]

# Drop rows with missing values
filtered_df = filtered_df.dropna(subset=['admin2', 'year', 'price', 'category'])

# Group data by category, year, and calculate the average price
avg_prices_category = filtered_df.groupby(['category', 'year'])['price'].mean().reset_index()

# Pivot the data for better visualization
pivot_prices_category = avg_prices_category.pivot(index='category', columns='year', values='price')

# Sort values within each category by price in descending order
pivot_prices_category = pivot_prices_category.sort_values(by=2022, ascending=False)

# Print the DataFrame with average prices per year by category
print(pivot_prices_category)

# Plotting the bar graph for average prices per year by category
plt.figure(figsize=(12, 6))
pivot_prices_category.plot.bar(rot=0, width=0.8)
plt.title('Average Food Prices Comparison by Category')
plt.xlabel('Category')
plt.ylabel('Average Food Price (Ksh)')
plt.legend(title=2023, bbox_to_anchor=(1, 1))
plt.grid(axis='y')
plt.tight_layout()
plt.xticks(rotation='vertical')  # Corrected parameter
plt.show()


# In[101]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'New_df'
# Ensure 'date' column is in datetime format
New_df['date'] = pd.to_datetime(New_df['date'])

# Extract the year, month, and create new columns 'year' and 'month'
New_df['year'] = New_df['date'].dt.year 

# Drop rows with missing values
New_df = New_df.dropna(subset=['admin2', 'year', 'price', 'category'])

# Convert 'price' column to numeric, handling errors by setting non-convertible values to NaN
New_df['price'] = pd.to_numeric(New_df['price'], errors='coerce')

# Filter data for the category 'pulses and nuts' and the year 2022
pulses_nuts = New_df[(New_df['category'] == 'pulses and nuts') & (New_df['year'] == 2022)]

# Group data by county and calculate the average price
avg_prices_county = pulses_nuts.groupby(['admin2'])['price'].mean().reset_index()

# Get the top 5 counties by average price spent
top_5_counties = avg_prices_county.nlargest(5, 'price')

# Print the DataFrame with average prices for 'pulses and nuts' in top 5 counties
print(top_5_counties)

# Plotting the bar graph for average prices for 'pulses and nuts' in top 5 counties
plt.figure(figsize=(12, 6))
plt.bar(top_5_counties['admin2'], top_5_counties['price'], color='brown')
plt.title('Average Price Spent on pulses and nuts in Top 5 Counties')
plt.xlabel('County')
plt.ylabel('Average Food Price (Ksh)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[102]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'New_df'
# Ensure 'date' column is in datetime format
New_df['date'] = pd.to_datetime(New_df['date'])

# Extract the year, month, and create new columns 'year' and 'month'
New_df['year'] = New_df['date'].dt.year 

# Drop rows with missing values
New_df = New_df.dropna(subset=['admin2', 'year', 'price', 'category'])

# Convert 'price' column to numeric, handling errors by setting non-convertible values to NaN
New_df['price'] = pd.to_numeric(New_df['price'], errors='coerce')

# Filter data for the category 'miscellaneous food' and the year 2022
miscellaneous_food_2022 = New_df[(New_df['category'] == 'miscellaneous food') & (New_df['year'] == 2022)]

# Group data by county and calculate the average price
avg_prices_county = miscellaneous_food_2022.groupby(['admin2'])['price'].mean().reset_index()

# Get the top 5 counties by average price spent
top_5_counties = avg_prices_county.nlargest(5, 'price')

# Print the DataFrame with average prices for 'miscellaneous food' in top 5 counties
print(top_5_counties)

# Plotting the bar graph for average prices for 'miscellaneous food' in top 5 counties
plt.figure(figsize=(12, 6))
plt.bar(top_5_counties['admin2'], top_5_counties['price'], color='red')
plt.title('Average Price Spent on Miscellaneous Food in Top 5 Counties')
plt.xlabel('County')
plt.ylabel('Average Food Price (Ksh)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[100]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'New_df'
# Ensure 'date' column is in datetime format
New_df['date'] = pd.to_datetime(New_df['date'])

# Extract the year, month, and create new columns 'year' and 'month'
New_df['year'] = New_df['date'].dt.year 

# Drop rows with missing values
New_df = New_df.dropna(subset=['admin2', 'year', 'price', 'category'])

# Convert 'price' column to numeric, handling errors by setting non-convertible values to NaN
New_df['price'] = pd.to_numeric(New_df['price'], errors='coerce')

# Filter data for the category 'cereals_and_tubers' and the year 2022
cereals_and_tubers = New_df[(New_df['category'] == 'cereals and tubers') & (New_df['year'] == 2022)]

# Group data by county and calculate the average price
avg_prices_county = cereals_and_tubers.groupby(['admin2'])['price'].mean().reset_index()

# Get the top 5 counties by average price spent
top_5_counties = avg_prices_county.nlargest(5, 'price')

# Print the DataFrame with average prices for 'cereals_and_tubers' in top 5 counties
print(top_5_counties)

# Plotting the bar graph for average prices for 'cereals_and_tubers' in top 5 counties
plt.figure(figsize=(12, 6))
plt.bar(top_5_counties['admin2'], top_5_counties['price'], color='skyblue')
plt.title('Average Price Spent on cereals_and_tubers in Top 5 Counties')
plt.xlabel('County')
plt.ylabel('Average Food Price (Ksh)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[104]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'New_df'
# Ensure 'date' column is in datetime format
New_df['date'] = pd.to_datetime(New_df['date'])

# Extract the year, month, and create new columns 'year' and 'month'
New_df['year'] = New_df['date'].dt.year 

# Drop rows with missing values
New_df = New_df.dropna(subset=['admin2', 'year', 'price', 'category'])

# Convert 'price' column to numeric, handling errors by setting non-convertible values to NaN
New_df['price'] = pd.to_numeric(New_df['price'], errors='coerce')

# Filter data for the category 'vegetables_and_fruits' and the year 2022
vegetables_and_fruits = New_df[(New_df['category'] == 'vegetables and fruits') & (New_df['year'] == 2022)]

# Group data by county and calculate the average price
avg_prices_county = vegetables_and_fruits.groupby(['admin2'])['price'].mean().reset_index()

# Get the top 5 counties by average price spent
top_5_counties = avg_prices_county.nlargest(5, 'price')

# Print the DataFrame with average prices for 'vegetables_and_fruits' in top 5 counties
print(top_5_counties)

# Plotting the bar graph for average prices for 'vegetables_and_fruits' in top 5 counties
plt.figure(figsize=(12, 6))
plt.bar(top_5_counties['admin2'], top_5_counties['price'], color='green')
plt.title('Average Price Spent on vegetables_and_fruits in Top 5 Counties')
plt.xlabel('County')
plt.ylabel('Average Food Price (Ksh)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[105]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'New_df'
# Ensure 'date' column is in datetime format
New_df['date'] = pd.to_datetime(New_df['date'])

# Extract the year, month, and create new columns 'year' and 'month'
New_df['year'] = New_df['date'].dt.year 

# Drop rows with missing values
New_df = New_df.dropna(subset=['admin2', 'year', 'price', 'category'])

# Convert 'price' column to numeric, handling errors by setting non-convertible values to NaN
New_df['price'] = pd.to_numeric(New_df['price'], errors='coerce')

# Filter data for the category 'meat_fish_and_eggs' and the year 2022
meat_fish_and_eggs = New_df[(New_df['category'] == 'meat, fish and eggs') & (New_df['year'] == 2022)]

# Group data by county and calculate the average price
avg_prices_county = meat_fish_and_eggs.groupby(['admin2'])['price'].mean().reset_index()

# Get the top 5 counties by average price spent
top_5_counties = avg_prices_county.nlargest(5, 'price')

# Print the DataFrame with average prices for 'meat_fish_and_eggs' in top 5 counties
print(top_5_counties)

# Plotting the bar graph for average prices for 'meat_fish_and_eggs' in top 5 counties
plt.figure(figsize=(12, 6))
plt.bar(top_5_counties['admin2'], top_5_counties['price'], color='yellow')
plt.title('Average Price Spent on meat_fish_and_eggs in Top 5 Counties')
plt.xlabel('County')
plt.ylabel('Average Food Price (Ksh)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[106]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'New_df'
# Ensure 'date' column is in datetime format
New_df['date'] = pd.to_datetime(New_df['date'])

# Extract the year, month, and create new columns 'year' and 'month'
New_df['year'] = New_df['date'].dt.year 

# Drop rows with missing values
New_df = New_df.dropna(subset=['admin2', 'year', 'price', 'category'])

# Convert 'price' column to numeric, handling errors by setting non-convertible values to NaN
New_df['price'] = pd.to_numeric(New_df['price'], errors='coerce')

# Filter data for the category 'oil and fats' and the year 2022
oil_and_fats = New_df[(New_df['category'] == 'oil and fats') & (New_df['year'] == 2022)]

# Group data by county and calculate the average price
avg_prices_county = oil_and_fats.groupby(['admin2'])['price'].mean().reset_index()

# Get the top 5 counties by average price spent
top_5_counties = avg_prices_county.nlargest(5, 'price')

# Print the DataFrame with average prices for 'oil_and_fats' in top 5 counties
print(top_5_counties)

# Plotting the bar graph for average prices for 'oil_and_fats' in top 5 counties
plt.figure(figsize=(12, 6))
plt.bar(top_5_counties['admin2'], top_5_counties['price'], color='grey')
plt.title('Average Price Spent on oil_and_fats in Top 5 Counties')
plt.xlabel('County')
plt.ylabel('Average Food Price (Ksh)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[108]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'New_df'
# Ensure 'date' column is in datetime format
New_df['date'] = pd.to_datetime(New_df['date'])

# Extract the year, month, and create new columns 'year' and 'month'
New_df['year'] = New_df['date'].dt.year 

# Drop rows with missing values
New_df = New_df.dropna(subset=['admin2', 'year', 'price', 'category'])

# Convert 'price' column to numeric, handling errors by setting non-convertible values to NaN
New_df['price'] = pd.to_numeric(New_df['price'], errors='coerce')

# Filter data for the category 'milk and dairy' and the year 2022
milk_and_dairy = New_df[(New_df['category'] == 'milk and dairy') & (New_df['year'] == 2022)]

# Group data by county and calculate the average price
avg_prices_county = milk_and_dairy.groupby(['admin2'])['price'].mean().reset_index()

# Get the top 5 counties by average price spent
top_5_counties = avg_prices_county.nlargest(5, 'price')

# Print the DataFrame with average prices for 'milk_and_dairy' in top 5 counties
print(top_5_counties)

# Plotting the bar graph for average prices for 'milk_and_dairy' in top 5 counties
plt.figure(figsize=(12, 6))
plt.bar(top_5_counties['admin2'], top_5_counties['price'], color='blue')
plt.title('Average Price Spent on milk_and_dairy in Top 5 Counties')
plt.xlabel('County')
plt.ylabel('Average Food Price (Ksh)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()



# In[110]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'New_df'
# Ensure 'date' column is in datetime format
New_df['date'] = pd.to_datetime(New_df['date'])

# Extract the year and create a new column 'year'
New_df['year'] = New_df['date'].dt.year 

# Drop rows with missing values
New_df = New_df.dropna(subset=['year', 'price'])

# Convert 'price' column to numeric, handling errors by setting non-convertible values to NaN
New_df['price'] = pd.to_numeric(New_df['price'], errors='coerce')

# Group data by year, category, and calculate the average price
avg_prices_yearly = New_df.groupby(['year'])['price'].mean().reset_index()

# Train-test split for model evaluation
train_data, test_data = train_test_split(avg_prices_yearly, test_size=0.2, random_state=42)

# Prepare features (X) and target variable (y) for training
X_train = train_data[['year']]
y_train = train_data['price']

# Prepare features (X) and target variable (y) for testing
X_test = test_data[['year']]
y_test = test_data['price']

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions for 2023 and 2024
years_to_predict = [2023, 2024]
predictions = model.predict(pd.DataFrame(years_to_predict, columns=['year']))

# Print the predictions for 2023 and 2024
for year, prediction in zip(years_to_predict, predictions):
    print(f"Predicted average price for all food categories in {year}: {prediction:.2f} Ksh")

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error on Test Set: {mse:.2f}")

# Plotting the predictions
plt.figure(figsize=(10, 6))

plt.plot(years_to_predict, predictions, 'ro-', label='Predictions')
plt.title('Average Prices Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Price (Ksh)')
plt.legend()
plt.show()


# In[119]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'New_df'
# Ensure 'date' column is in datetime format
New_df['date'] = pd.to_datetime(New_df['date'])

# Extract the month and create a new column 'month'
New_df['month'] = New_df['date'].dt.month_name()

# Drop rows with missing values
New_df = New_df.dropna(subset=['month', 'price', 'category'])

# Convert 'price' column to numeric, handling errors by setting non-convertible values to NaN
New_df['price'] = pd.to_numeric(New_df['price'], errors='coerce')

# Group data by month and calculate the average price
avg_prices_monthly = New_df.groupby(['month'])['price'].mean().reset_index()

# Sort months in chronological order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
avg_prices_monthly['month'] = pd.Categorical(avg_prices_monthly['month'], categories=month_order, ordered=True)
avg_prices_monthly = avg_prices_monthly.sort_values('month')

# Plotting the average prices by month
plt.figure(figsize=(12, 6))
plt.bar(avg_prices_monthly['month'], avg_prices_monthly['price'], color='skyblue')
plt.title('Average Prices by Month (Up to 2022)')
plt.xlabel('Month')
plt.ylabel('Average Price (Ksh)')
plt.show()


# In[122]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'New_df'
# Ensure 'date' column is in datetime format
New_df['date'] = pd.to_datetime(New_df['date'])

# Extract the month and create a new column 'month'
New_df['month'] = New_df['date'].dt.month_name()

# Drop rows with missing values
New_df = New_df.dropna(subset=['month', 'price', 'category'])

# Convert 'price' column to numeric, handling errors by setting non-convertible values to NaN
New_df['price'] = pd.to_numeric(New_df['price'], errors='coerce')

# Group data by month and calculate the average price
avg_prices_monthly = New_df.groupby(['month'])['price'].mean().reset_index()

# Sort months by average price in descending order
avg_prices_monthly = avg_prices_monthly.sort_values('price', ascending=False)

#plotting  a table
print(avg_prices_monthly)

# Plotting the average prices by month
plt.figure(figsize=(12, 6))
plt.bar(avg_prices_monthly['month'], avg_prices_monthly['price'], color='skyblue')
plt.title('Average Prices by Month')
plt.xlabel('Month')
plt.ylabel('Average Price (Ksh)')
plt.show()


# In[ ]:




