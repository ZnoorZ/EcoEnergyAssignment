#MPI values are real numbers so I am simply using linear regression 
import numpy as np
import pandas as pd

#procedures

def feature_Normalize(x):
    x_norm = x
    mu = np.zeros(len(x.columns)-1)
    sigma = np.zeros(len(x.columns)-1)
    mu = x_norm.loc[:, x_norm.columns != 'Continent'].mean(axis=0)
    sigma = x_norm.loc[:, x_norm.columns != 'Continent'].std(axis=0)    
    x_norm.loc[:, x_norm.columns != 'Continent']= (x_norm.loc[:, x_norm.columns != 'Continent']-mu)/sigma
    return x_norm, mu, sigma


def compute_cost(X, y, theta):
    J = 0
    m = len(y)
    J = np.sum(np.square(np.subtract(np.dot(X,theta),y)))/(2*m)
    return J

def gradient_descent(X, y, theta2, alpha, iterations):
    m = len(y)
    J_history = np.zeros(iterations)
    temp = theta2
    for i in range(1,iterations): 
        s=pd.DataFrame(np.zeros(5))
        for j in range(0, 4):
            x_column= X.iloc[:,j]
            s.iloc[j] = np.sum((np.subtract(np.dot(X,theta2),y)).mul(x_column, axis=0))
        temp = theta2 - alpha*s/m        
        theta2=temp            
        J_history[i] = compute_cost(X, y, theta2) 
    return theta2, J_history



def num_missing(x):
  return sum(x==0)


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

model_data['Population'] = pd.to_numeric(model_data['Population'], errors='coerce')


#there are 984 entries in total
#600 datapoints for training data
train_set_x = model_data.loc[0:599,['Continent', 'Population', 'IoD_Rural', 'IoD_Urban']] 

train_set_y = model_data.loc[0:599,['MPI']] 

#384 data points for testing data
test_set_x = model_data.loc[600:983,['Continent', 'Population', 'IoD_Rural', 'IoD_Urban']] 

test_set_y = model_data.loc[600:983,['MPI']] 
        

# main flow of training
m = len(train_set_x.columns)

#normalizing features

X, mu, sigma = feature_Normalize(train_set_x)

#adding intercept term
X['intercept']=1

# choosing alpha values and number of iterations
alpha = 0.01
num_iterations = 100

#initialising theta for gradient descent
theta = pd.DataFrame(np.zeros(m+1))

compute_cost(X, train_set_y, theta)

theta2 = pd.DataFrame(np.zeros(m+1))

theta2, J_history = gradient_descent(X, train_set_y, theta2, alpha, num_iterations)

#badly got stuck in dot product with pandas in gradient descent procedure, 
#can't move ahead with this code
