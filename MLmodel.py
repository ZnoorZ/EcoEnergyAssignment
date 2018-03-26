#MPI values are real numbers so I am simply using linear regression 
from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#fetching data
filename = "processed_data.csv"

data = pd.read_csv(filename)

model_data =  data.loc[:,['Continent' ,'Year_of_survey_population', 'Intensity_of_Deprivation_Rural',
                   'Intensity_of_Deprivation_Urban', 'MPI_National']]

model_data.columns = ['Continent', 'Population', 'IoD_Rural', 'IoD_Urban', 'MPI' ] 

#converting all features to numerical type
#model_data.info()
model_data['Continent'].unique()
num_labels = {"Continent":{'Asia':1, 'Africa':2, 'Americas':3}}
model_data.replace(num_labels, inplace=True)
#print(model_data['Population'])
model_data.info()

#there are 984 entries in total
#600 datapoints for training data
train_set_x = model_data.loc[0:599,['Continent', 'Population', 'IoD_Rural', 'IoD_Urban']] 

train_set_y = model_data.loc[0:599,['MPI']] 

#384 data points for testing data
test_set_x = model_data.loc[600:983,['Continent', 'Population', 'IoD_Rural', 'IoD_Urban']] 

test_set_y = model_data.loc[600:983,['MPI']] 
        
#converting dataframes to matrices

train_set_x = train_set_x.as_matrix().astype(np.float)
train_set_y = train_set_y.as_matrix().astype(np.float)
test_set_x = test_set_x.as_matrix().astype(np.float)
test_set_y = test_set_y.as_matrix().astype(np.float)

#print(test_set_x)

# main flow of training
lm = linear_model.LinearRegression()
lm.fit(train_set_x, train_set_y) 
print(lm.coef_) 

#predicting MPI
y_hat = lm.predict(test_set_x)
plt.scatter(test_set_y, y_hat)
plt.title("MPIs vs Predicted MPIs")
plt.xlabel('Original MPIs')
plt.ylabel('Predicted MPIs')

#calculating residual errors
r_errors = test_set_y - y_hat

#plotting residual errors for both training and testing data
plt.scatter(lm.predict(train_set_x), lm.predict(train_set_x)-train_set_y, c='b', s=40, alpha=0.5)
plt.scatter(y_hat, r_errors, c='g', s=40, alpha=0.5) 
plt.hlines(y=0, xmin=-0.2, xmax=0.8)
plt.title('Residual Plot using training(blue) and test (green) data')
plt.ylabel('Residuals')
plt.xlabel('Predicted values')
