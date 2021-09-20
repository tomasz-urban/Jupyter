# Logistic regression

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