#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Task 1

#a) Import the required libraries.
import pandas as pd
import numpy as np
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
import statsmodels.api as sm
from sklearn import datasets,linear_model
from sklearn.linear_model import SGDRegressor,LinearRegression
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Store the data into pandas dataframe and call it df_marketing
df_marketing=pd.read_csv('C:\\Users\\L HIMAJA REDDY\\Desktop\\Dataset\\SalesBasedOnAdvertising.csv',
                         names=["Sno","TV","radio","newspaper","sales"],
                         skiprows=1,
                         engine='python')

#b) Analyze the shape of the Data using df_makreby checking the no. of rows and columns available
print("\n----Shape of dataset----\n")
print(df_marketing.shape)
# There are 204 rows x 5 columns

#c) Check the basic statistics of the given parameters and also look into the type of the data.
#Describe the Dataset in your words including the columns, their type, ranges and other stats.
print("\n----Basic Statistics of dataset----\n")
print(df_marketing.describe())
print("\n----Information of dataset----\n")
print(df_marketing.info())

#d) Next check if there is any Null/character type data in the given data.
#If any Sales data is Null or contain value 0, drop the complete row.

print("\n----df_marketing data contains any null value or not ?----\n",df_marketing.isnull().values.any())
df_marketing=df_marketing[df_marketing.sales != 0]
df_marketing= df_marketing[pd.notnull(df_marketing['sales'])]

#For any other column, fill those cell with the average value for those columns.
df_marketing['radio'] = df_marketing['radio'].fillna((df_marketing['radio'].mean()))
df_marketing['TV'] = df_marketing['TV'].fillna((df_marketing['TV'].mean()))
df_marketing['newspaper'] = df_marketing['newspaper'].fillna((df_marketing['newspaper'].mean()))


#e)  Once again perform the  Task b) and c) to check the shape and statistics of the data. Explain the changes.
print("\n----Shape of dataset after modifications in the dataset----\n")
print(df_marketing.shape)
print("\n----Basic Statistics after modifications in the dataset----\n")
print(df_marketing.describe())
print("\n----Information of the modified dataset----\n")
print(df_marketing.info())


# ->What are the features?
# 
# TV: advertising money spent on TV for a single product in a given market (in Thousand  Rupees)
# 
# Radio: advertising money spent on Radio (in Thousand  Rupees)
# 
# Newspaper: advertising money spent on Newspaper (in Thousand  Rupees)
# 
# ->What is the response?
# 
# Sales: sales of a single product in a given market (in Lac Rupees)
# 
# ->What else do we know?
# 
# As the response variable is continuous, this is a regression problem.

# In[195]:


#Task 2
# a) Create the Box-plot of all the numeric columns and explain the result in your words.

color = {'boxes': 'DarkGreen', 'whiskers': 'DarkOrange','medians': 'DarkBlue', 'caps': 'Gray'}
df_marketing.plot(kind='box',color=color, sym='r*',title="Box-plot of all the columns")

#b) Next you need to create a figure and create two subplots(contained in the same row) on it with the following details -
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(15,3))

#First subplot will show the Box plot of Sales, tv, radio and newpapers columns
boxplot = df_marketing.boxplot(column=["sales","TV","radio","newspaper"],sym='r*',ax=axes[0],notch=True)

#Create Line plot of all the 4 columns Sales, tv, radio and newpapers
exclude = ['Sno','TotalAdvt']
df_marketing.ix[:, df_marketing.columns.difference(exclude)].plot(ax=axes[1], kind='line',title="Line Plot for  4 columns Sales, tv, radio and newpapers ");



# OBSERVATION FROM THE BOX-PLOT OF ALL THE COLUMNS
# By analyzing the whisker of TV column we can say that the range of values of TV is very high when compared with other columns.
# The range of the data points of radio are between 0 and 50.
# The box plot of newspaper is centred at the lower portion of the graph.This shows that most of the data points are centred lower down on the y-axis but there are two outliers as specified by marker * in red color.
# There is an outlier for Sales column which is not consistent and far above the other values and therefore it is represented by the marker * in red.

# In[196]:


#Task 3
#a) In order to predict the Sales based on Marketing expenses,
#find out the possible independent and dependent variables from the given Dataset. 

plt.figure(figsize=(10, 5))
plt.scatter(
    df_marketing['TV'],
    df_marketing['sales'],
    c='black'
)
plt.xlabel("Money spent on TV ads")
plt.ylabel("Sales")
plt.title("Scatter plot showing Money spent on TV ads vs Sales")
plt.show()

#Generating a linear approximation of this data.
X = df_marketing['TV'].values.reshape(-1,1)
y = df_marketing['sales'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(X, y)

#Let’s visualize how the line fits the data.
predictions = reg.predict(X)
plt.figure(figsize=(10, 5))
plt.scatter(
    df_marketing['TV'],
    df_marketing['sales'],
    c='black'
)
plt.plot(
    df_marketing['TV'],
    predictions,
    c='blue',
    linewidth=2
)
plt.xlabel("Money spent on TV ads")
plt.ylabel("Sales")
plt.title("Plot showing fitting of data")
plt.show()

#Assessing the relevancy of the model(we need to look at the R² value and the p-value from each coefficient.)
X = df_marketing['TV']
y = df_marketing['sales']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# The simple linear regression equation has the form Y=a+bX, where Y is the dependent variable (that's the variable that goes on the Y axis), X is the independent variable (i.e. it is plotted on the X axis), b is the slope of the line and a is the y-intercept.
# 
# By using visualisation, we can judge which variables have a linear relationship with y.
# 
# In this case, we have used “Sales” as our response/y.we can see, there is a clear relationship between the amount spent on TV ads and sales.
# 
# From the second graph , it seems that a simple linear regression can explain the general impact of amount spent on TV ads and sales.
# 
# Therefore,Sales is the dependent variable.The independent variables are advertising money spent on TV for a single product in a given market (in Thousand Rupees),advertising money spent on Radio (in Thousand Rupees),advertising money spent on Newspaper (in Thousand Rupees)
# 
# Observations from the OLS regression results:
# 
# Looking at both coefficients, we have a p-value that is very low (although it is probably not exactly 0). This means that there is a strong correlation between these coefficients and the target (Sales).Then, looking at the R² value, we have 0.522 which is pretty low.

# In[197]:


#b) Now, you would have already figured out that Sales is your dependent variable for which you are going to create your model.
#Lets Create the distribution plot along with kde plot for the Sales data and explain it in your words.
sns.distplot(df_marketing['sales'],bins=20)


# The above distribution plot contains the sales on the x-axis and it's frequency on the y-axis.The area of the bars in the histogram is the measure of the frequency of sales in the dataset.From the highest peak in the histogram,we see that most of the sales of the product are in between 12 and 14 lacs.

# In[198]:


#Task 4

#a) Create the Scatter plot using the dataframe plot function keeping the Sales on Y-axis and expenditure on tv on x-axis.

df_marketing.plot.scatter(x='TV',y='sales',c='red') 

#b) Create 3 subplots (in a single row) in a figure for the following -
#The points should be of seagreen, darkgreen and black colours respectively
#Provide the appropriate labels for x and y axis and name of the graphs.
fig = plt.figure(figsize = (16,5))

ax1 = fig.add_subplot(131)
ax1.set_title('Sales vs TV Advertising')
ax1.scatter(df_marketing['sales'],
         df_marketing['TV'], 
         color = 'seagreen')#Sales vs TV Advertising
plt.xlabel('Sales') 
plt.ylabel('TV') 

ax2 = fig.add_subplot(132)
ax2.set_title('Sales vs Radio Advertising')
ax2.scatter(df_marketing['sales'],
         df_marketing['radio'], 
         color = 'darkgreen')#Sales vs Radio Advertising
plt.xlabel('Sales') 
plt.ylabel('Radio')

ax3 = fig.add_subplot(133)
ax3.set_title('Sales vs NewsPaper Advertising')
ax3.scatter(df_marketing['sales'],
         df_marketing['newspaper'], 
         color = 'black')#Sales vs NewsPaper Advertising
plt.xlabel('Sales') 
plt.ylabel('Newspaper')

#c) Create the pairplot for Sales with different marketing expenses using seaborn

sns.pairplot(df_marketing, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=7, aspect=0.7, kind='reg')


# OBSERVATIONS:
# 
# Strong relationship between TV ads and sales.
# Weak relationship between Radio ads and sales.
# Very weak to no relationship between Newspaper ads and sales.

# In[199]:


#Task 5

#a) Add a new column “TotalAdvt” to the dataframe df_marketing, which will have the total of all the three marketing expenditures
#(tv, radio and news-paper)

df_marketing['TotalAdvt'] = df_marketing['TV'] + df_marketing['radio'] + df_marketing['newspaper']

#b) Fit a Simple Linear Regression for Sales and  TotalAdvt and find out the values of intercept slope
#and the R-square value. Describe the model in your terms. The model should be named simple_model.

X_train, X_test, y_train, y_test = train_test_split(df_marketing.TotalAdvt.values.reshape(-1,1),
                                                    df_marketing.sales)

# Instantiating
simple_model = LinearRegression()

#Fitting the model to the training data 
simple_model.fit(X_train, y_train)


# print the intercept and coefficients and  R-square
print(simple_model.intercept_)
print(simple_model.coef_) 
print(simple_model.score(X_train, y_train))

#d) Perform part c) using the predict method for the model created in part b)
y_pred = simple_model.predict(X_test)
predictors = ['TotalAdvt']
X = df_marketing[predictors]
simple_model.predict(X)
new_X = [[50000]]
print("\n----Sales using the predicate Model if total expenditure on advertisements is Rupees 50,000.----\n",
      simple_model.predict(new_X))


# #c) What will be the Sales using the above Model if total expenditure on advertisements is Rupees 50,000.
# #Show the steps done to find out the Sales value. (write the mathematical equation used )
# 
# Equation used is sales=4.289+0.048*TotalAdvt.
# If total expenditure on advertisements is Rupees 50,000 
# then 
# sales=4.289+0.048*50000
# sales=4.289+2400
# sales=2404.289

# In[200]:


#Task 6 – In this section, we will create a different model to see the impact of different marketing expenditures on Sales.
#For this we will use advertising on tv, radio and news-paper as separate variables or independent variables.

#a) Fit a Simple Linear Regression for Sales while keeping tv, radio and newspaper as three different x-parameters.
#Find out the values of intercept slopes and the R-square value. 
multi_X_train, multi_X_test, multi_y_train, multi_y_test = train_test_split(df_marketing[["TV", "radio", "newspaper"]],
                                                    df_marketing.sales)

#Instantiating
multi_model = LinearRegression()#The model should be named multi_model

# Fitting the model to the training data.
multi_model.fit(multi_X_train, multi_y_train)


# print the intercept and coefficients and  R-square
print(multi_model.intercept_)
print(list( zip( ["TV", "Radio", "Newspaper"], list( multi_model.coef_ ) ) ) )
print(multi_model.score(multi_X_train, multi_y_train))


#c) Perform part b) using  the predict method for the model.
multi_y_pred = multi_model.predict(multi_X_test)
predictors = ["TV", "radio", "newspaper"]
X = df_marketing[predictors]
multi_model.predict(X)
new_X = [[90000,3000,45000]]
print("\n----Sales using the predicate Model if advertising money spent on Newspaper is 90000, advertising money spent on Radio is 3000 and advertising money spent on Newspaper is 45000.----\n",
      multi_model.predict(new_X))

new_X1 = [[290000,0,80000]]
print("\n----Sales using the predicate Model if advertising money spent on Newspaper is 290000, advertising money spent on Radio is 0 and advertising money spent on Newspaper is 80000.----\n",
      multi_model.predict(new_X1))


# Here we used a technique that estimates a single regression model with more than one outcome variable. When there is more than one predictor variable in a multivariate regression model, the model is a multivariate multiple regression.
# 
# #b) Write the Mathematical equation for Sales for this Model. Predict Sales for the below mentioned Expenditures
# 
# Equation is:
# sales=2.786+0.046*TV+0.189*Radio+0.001*Newspaper
# 
# #i) tv 90000, radio 3000 and newspaper 45000
# 
# sales=2.786+0.046*90000+0.189*3000+0.001*45000
# sales=4754.786
# 
# #ii) tv 290000 and newspaper 80000
# sales=2.786+0.046*290000+0.189*0+0.001*80000
# sales=13422.786
# 
# #d) Compare the result in part b) and c)
# The predicted values of multi_model are close to the actual ones.The difference between these values is pretty low.This indicates that our model fitted the data well.

# In[201]:


#Task 7 – As part of task 5) and 6) we created two different models. 
#Compare the two models (simple_model and  multi_model) by calculating the Error in the Models.
#Explain which model is better and why?

print('Mean Absolute Error of simple_model:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error of simple_model:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error of simple_model:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  

test_pred_df = pd.DataFrame( { 'actual': y_test,
                            'predicted': np.round( y_pred, 2),
                            'residuals': y_test - y_pred } )
print("\n----Errors in simple_model----")
print(test_pred_df[0:10])


print('\nMean Absolute Error of multi_model:', metrics.mean_absolute_error(multi_y_test, multi_y_pred))  
print('Mean Squared Error of multi_model:', metrics.mean_squared_error(multi_y_test, multi_y_pred))  
print('Root Mean Squared Error of multi_model:', np.sqrt(metrics.mean_squared_error(multi_y_test, multi_y_pred)))  

multi_test_pred_df = pd.DataFrame( { 'actual': multi_y_test,
                            'predicted': np.round( multi_y_pred, 2),
                            'residuals': multi_y_test - multi_y_pred } )
print("\n----Errors in multi_model----")
print(multi_test_pred_df[0:10])

#a) For both the simple regression model (stored in simple_model) and Multi-variate model (stored in multi_model) 
#find out the predicted value of Y, for the given  X-values and store it in simple_y and multi_y.
predictors = ['TotalAdvt']
X = df_marketing[predictors]
simple_model.predict(X)
new_X = [[50000]]
simple_y=simple_model.predict(new_X)

predictors = ["TV", "radio", "newspaper"]
X = df_marketing[predictors]
multi_model.predict(X)
new_X = [[90000,3000,45000]]
multi_y=multi_model.predict(new_X)

new_X1 = [[290000,0,80000]]
multi_y1=multi_model.predict(new_X1)
multi_y=np.concatenate((multi_y , multi_y1))

print("\npredicted value of Y, for the given  X-values of simple regression model")
print(simple_y)
print("\npredicted value of Y, for the given  X-values of multi variate model")
print(multi_y)

#b) Find out the Mean Squared Error, using the metrics module of sklearn, for both the models 
#(using actual and predicted values of Sales). 
#Store the result in simple_mse and multi_mse respectively.
simple_mse=metrics.mean_squared_error(y_test, y_pred)
multi_mse= metrics.mean_squared_error(multi_y_test, multi_y_pred)
print('\nMean Squared Error of simple_model:',simple_mse)
print('Mean Squared Error of multi_model:',multi_mse)


# #c) Explain the result in step b) in your words describing which model is better and why?
# 
# After calculating the errors for both the models,we can conclude that  multi variate model is better because in this case the errors are less when compared to simple regression model.Also by observing residuals in both the cases we can say that multi variate model is better than simple regression model.
