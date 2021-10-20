# Logistic regression with statsmodels

`a) importing libraries`
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

`b) dummy variables (yes/no example)`
data['var_1'] = raw_data['var_1'].map({'Yes':1, 'No':0})

`c) declaring dependent and independent variables`
y = data['']
x1 = data[['', '']]

`c) visualizing on scatter plot`
plt.scatter(x1,y)
plt.xlabel('', fontsize = 15)
plt.ylabel('', fontsize = 15)
plt.show()

`d) regression model`
x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()

results_log.summary()

`Summary table explanation:`
MLE (maximum likelihood estimation) - a function which estimates how likely it is that the model at hand describes the real unredlying relationship of the variables. The bigger the likelyhood function, the higher the probability that model is correct.
MLE tries to maximize the likelihood function. When it can no longer be improved it will stop the optimization.

Log-likelihood - it is almost always negative and the bigger value the better 

LL-Null (log likelihood-null) - the log-likelihood of a model which has no independent variables

LLR p-value (log likelihood ratio) - it measures if our model is statistically different from LL-Null (which is a useless model). Just like with p-value in linear regression closer to 0.00 the better. 

Pseudo R-squ (pseudo R-squared) - this is McFadden's R-squared. Good value is between 0.2 and 0.4. This can be useful when comparing variations of the same model. Do not use it to compare different models

`e) coefficient explanation:`
Coefficient for one of the variables is 1.9449

np.exp(1.9449) 
6.992932526814459

Coefficient explanation:

The coefficient of Gender is 1.9449, and it's exp. value is 6.99. Given the same SAT score, female is 7 times more likely to get admitted.
This is according to odds ratio interpretation for binary logistic regression.
The equation for logistic regression is:
`log(odds) = -68.3489 + 1.9449 * Gender`

If we get 2 equations for two Genders we get:
`log(odds2) = -68.3489 + 1.9449 * Gender2`
`log(odds1) = -68.3489 + 1.9449 * Gender1`

As we know: `log(x) - log(y) = log(x/y)`, following that:
`log(odds2/odds1) = 1.9449*(Gender2 - Gender1)`

Knowing that Gender2 is 1 and Gender1 is 0 we will get:
`log(odds2/odds1) = 1.9449`
`odds2/odds1 = exp(1.9449)`
`odds2 = 6.99 * odds1`

so in this case:
`odds(female) = 6.99 * odds(male)`

so as I wrote at the beginning female is 7 times more likely to get admitted having the same SAT score (of course not in general, just in this case).

`f) confusion matrix`
cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0:'Actual 0', 1:'Actual 1'})
cm_df

.. or another way:

cm = np.array(cm_df)
accuracy = (cm[0,0]+cm[1,1])/cm.sum()
accuracy

`g) testing the model:`
Test data need to have the same shape as the x so we need to make some changes

test_actual = test['dependent_variable / y']
test_data = test.drop(['dependent_variable / y'], axis = 1)
test_data = sm.add_constant(test_data)

#If columns would be in different order we would have to rearrange them:
#test_data = test_data[x.columns.values]

test_data

#Now we check the test accuracy using (for example) prepared function:

def confusion_matrix(data, actual_values, model):
    
    pred_values = model.predict(data)
    bins = np.array([0,0.5,1])
    cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
    accuracy = (cm[0,0] + cm[1,1])/cm.sum()
    return cm, accuracy


# Logistic regression with sklearn

`a) importing libraries`
import pandas as pd
import numpy as np



`b) Data preparation`

`c) Reading data`
raw_data = pd.read_csv('data/data_file.csv')

`d) Descriptive statistics`
raw_data.describe(include='all')

`e) Basic info`
raw_data.info()

`f) Making a copy of original data before proceeding`
df = raw_data.copy()

`g) Showing all rows and column in notebook`
pd.options.display.max_columns = None
pd.options.display.max_rows = None

`h) Pandas dummies`
columns_dummies = pd.get_dummies(df['column_name'], drop_first = True)

To avoid multicolinearity we have to drop one of the columns (in this case column 0). 
Leaving all the columns except this one makes no harm to the model 
because this column is fully explainable using all of the columns left. 

`i) Grouping columns using .loc`
group_1 = columns_dummies.loc[:,'a':'c'].max(axis=1) # all rows, columns from 'a' to 'c'
group_2 = columns_dummies.loc[:,'d':'f'].max(axis=1)
group_3 = columns_dummies.loc[:,'g':'j'].max(axis=1)
group_4 = columns_dummies.loc[:,'k':].max(axis=1) # all rows, columns from 'k' to the end

`j) Dropping columns`
#We can drop columns for example after changing them to dummies (we drop original column)
df = df.drop('column_name', axis=1)

`k) Concatenating data`
#When for example groupin columns we are creating new df's. We can concatenate all of that information using pandas method
df = pd.concat([df, group_1, group_2, group_3, group_4], axis = 1)

`l) Renaming columns`
#We can also rename those column as we want. We can list all columns first to save some time 
df.columns.values
#Than we have to make a new list with column names (using list we just printed out)
column_names = []   #copy list from above and change those column which You want
df.columns = column_names

`m) Reordering columns`
columns_reordered = []   #get columns names from df.columns.values and put them in the order You want
df = df[columns_reordered]

`n) Object to datetime`
#We can change date (if stored as an object type) to datetime using pandas
df['Date'] = pd.to_datetime(df['Date'], format = '%d/%m/%Y')

`o) Extracting month/day/year from date`
#We can extract month/day/year from datetime object using for loop (here we have month as an example)
months_list = []
for i in range(df.shape[0]):
    months_list.append(df['Date'][i].month)

#Then we create new column for months
df['Month'] = months_list

#The same we can do for day (day of the week not the number stored in the datetime object):
df['Day of the week'] = df['Date'].dt.weekday

`p) Using .map to change column values (it can be string or number but in DS we need numbers)`
#In this particular case we are also grouping values by assigning the same values to different numbers
df['column_name'] = df['column_name'].map({1:0, 2:1, 3:1, 4:1})

`r) Saving to .csv`
#After doing all the preparation we can save our dataframe to .csv
df_preprocessed.to_csv('data/df_preprocessed.csv', index = False)

`s) Preparing data for Machine Learning`
#First we have to prepare targets and inputs for machine learning
#Sometimes targets are already in the data (in this case we just have to choose the appriopriate column)
#If not we can we can for example divide column into groups using .median() for binary classification

data_preprocessed = pd.read_csv('data/df_preprocessed.csv')
targets = np.where(df_preprocessed['column_name'] > 
                   dfpreprocessed['column_name'].median(), 1, 0)
df_preprocessed['Targets'] = targets
data_targets = df_preprocessed.drop('column_name', axis = 1)

#Getting inputs (in here we choose all the rest of columns)
unscaled_inputs = data_targets.iloc[:,:-1]


`Splitting data`
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, test_size = 0.2, random_state = 20)

`Modeling'
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

model = LogisticRegression()

model.fit(x_train, y_train)

model.get_params()

model.score(x_train, y_train)

#Intercept = bias
model.intercept_

#Coefficients = weights - the closer they are to 0 the smaller the weight (for the models with the same scale like this one)
model.coef_