import pandas
import numpy

# procedures

def compute_iqr(data):
    data.boxplot('MPI_National')
    Q1 = data['MPI_National'].quantile(0.25)
    Q3 = data['MPI_National'].quantile(0.75)
    mask = data['MPI_National'].between(Q1, Q3, inclusive=True)
    iqr = data.loc[mask, 'ISO':'MPI_National']
    print(iqr)
   

def find_correlation(x):
    pandas.plotting.scatter_matrix(x)
    
def find_disparities(x, a, b):
    sd = x.groupby(a).agg(numpy.std, ddof=1)
    print(sd.sort_values(b, ascending=False).head(10))
    
def num_missing(x):
  return sum(x == 0)

def show_missing(x):
       if (num_missing(x) > 0):       
           print(x)


#fetching data 
# Load CSV using Pandas
           
filenameN = 'data\MPI_national.csv'
filenameSn = 'data\MPI_subnational.csv'
fileMr = 'data\Main_MPI_results.csv'           #contains population of countries data
fileCountry = 'data\Countries_info.csv'        #contains per capita income of countries
fileContinent = 'data\Continents.csv'          #contaisn continent info of countries

dataN = pandas.read_csv(filenameN)
dataSn = pandas.read_csv(filenameSn)
dataMr = pandas.read_csv(fileMr)
dataCountry = pandas.read_csv(fileCountry)
dataContinents = pandas.read_csv(fileContinent)

dataContinents.info()

#getting only required columns
dataContinents = dataContinents.loc[:,['ISO', 'Continent']] 


#setting new column names
dataN.columns = ['ISO', 'Country', 'MPI_Urban', 'Headcount_Ratio_Urban',
                 'Intensity_of_Deprivation_Urban', 'MPI_Rural', 'Headcount_Ratio_Rural',
                 'Intensity_of_Deprivation_Rural']

dataSn.columns = ['ISO', 'Country', 'Sub-national_region', 'World_region', 
         'MPI_National', 'MPI_Regional', 'Headcount_Ratio_Regional', 
         'Intensity_of_Deprivation_Regional']

dataMr.columns = ['ISO', 'Country', 'World_region', 'MPI_datasource', 'Year','MPI',
                  'Headcount_ratio', 'Intensity_of_Deprivation', 'Population_vulnerable_to_poverty',
                  'Population_in_severe_poverty', 'Destitute_population', 'MPI_proportion_of_destitutes',
                  'Inequality_among_poor', 'Year_of_survey_population', 'Population13',
                  'Population14', 'Year_of_survey_MPI', 'MPI_population13', 'Indicators_num', 'missing_indicators']

#removing repeting columns
columnsToRemove = ['Country', 'World_region']
dataSn.drop(columnsToRemove, axis=1, inplace=True)
dataMr.drop(['Country'], axis=1, inplace=True)


dataN.head()
dataSn.head()
dataMr.head()

dataN.info()
dataSn.info()
dataMr.info()

dataCountry.info()
dataCountry.drop(['Country Name'], axis=1, inplace=True)

#filling missing values in country dataset
dataCountry[['2016']].isna().any()    #just checking
dataCountryT = dataCountry.T
dataCountryT.fillna(method='ffill', inplace=True)
dataCountry = dataCountryT.T

#checking for missing values if any 
dataCountry[['2017']].isna().any()
dataCountry.head()

#only 2017 column will be used (for income info) therefore converting only its datatype to numeric
dataCountry['2017'] = pandas.to_numeric(dataCountry['2017'], errors='coerce').fillna(0)

print (dataN.apply(num_missing, axis=0)) 
print (dataSn.apply(num_missing, axis=0))
#axis=0 defines that function is to be applied on each column

#for displaying complete datapoints, which contain missing values
dataN.apply(show_missing, axis=1)

#merging both datasets into one
raw_data = dataSn.merge(dataN, on='ISO', how='left')
raw_data.info()
raw_data.ix[40:60]
raw_data = raw_data.merge(dataContinents, on='ISO', how='left')
raw_data = raw_data.merge(dataMr, on='ISO', how='left')

print (raw_data.apply(num_missing, axis=0))
raw_data.apply(show_missing, axis=1)

#replacing zeros with NaN values
raw_data[['MPI_Regional']]=raw_data[['MPI_Regional']].replace(0, numpy.NaN)
raw_data[['Headcount_Ratio_Regional']]=raw_data[['Headcount_Ratio_Regional']].replace(0, numpy.NaN)
raw_data.MPI_Regional = raw_data.groupby('World_region')['MPI_Regional'].apply(lambda x: x.fillna(x.mean()))

#replacing na values with the mean withinn same region
raw_data.Headcount_Ratio_Regional = raw_data.groupby('World_region')['Headcount_Ratio_Regional'].apply(lambda x: x.fillna(x.mean()))
raw_data.Intensity_of_Deprivation_Regional = raw_data.groupby('World_region')['Intensity_of_Deprivation_Regional'].apply(lambda x: x.fillna(x.mean()))

#merging country dataset
raw_data = raw_data.merge(dataCountry, on='ISO', how='left')

raw_data.info()

#processed data
pdata= raw_data.copy()
pdata.info()

#cleaning completed

#task 1
pdata.head()
# Computing IQR
data = pdata[['ISO', 'MPI_National']].sort_values('MPI_National').drop_duplicates()
compute_iqr(data)

# finding correlation between the Intensity of Deprivation Rural and the Continent the country is on   

#replacing continent labels with numbers
pdata['Continent'].unique()
num_labels = {"Continent":{'Asia':1, 'Africa':2, 'Americas':3}}
data = pdata.replace(num_labels)
find_correlation(data[['Continent', 'Intensity_of_Deprivation_Rural']])

# finding correlation between the Intensity of Deprivation Urban and the Population of a country
find_correlation(pdata[['Intensity_of_Deprivation_Urban', 'Year_of_survey_population']])

#finding correlation between the Intensity of Deprivation Urban and the Intensity of Deprivation Rural
find_correlation(pdata[['Intensity_of_Deprivation_Urban', 'Intensity_of_Deprivation_Rural']])

#finding countries that exhibit largest subnational disparities in MPI
data = pdata[['Country','MPI_Regional']]
find_disparities(data, 'Country', 'MPI_Regional')    

# finding countries which have high per-capita incomes still rank highly in MPI 
data = pdata[['ISO', 'MPI_National', '2017']].drop_duplicates()
data = data.sort_values('2017', ascending=False)
data = data.head(10).sort_values('MPI_National', ascending=False)
print(data)

#saving processed data to csv
pdata.to_csv('processed_data.csv', encoding='utf-8', index=False) 