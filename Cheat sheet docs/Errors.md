### ERROR METRICS

## SUM OF SQUARED ERROR (SSE) and MEAN SQUARED ERROR (MSE)

The downside of `SSE` is that it depends on number of datapoints. If You have number of n=100 predictions and n=1000 predictions, the error would be 10 times bigger just because of making more predictions.
The answer for this problem is `Mean Squared Error` which is `SSE` divided by the number of predictions. So `MSE` is invariant to the number of samples.
The `MSE` is also an estimate of the expected squared error.

Both `SSE` and `MSE` don't have intuitive units. It provides squared values, so for examples if we predict temperature we have squared Celsius degrees, which has no explanatory straight forward power. The answer for this problem is `Root mean squared error`.

## ROOT MEAN SQUARED ERROR (RMSE)

`RMSE` has intuitive units. If we predict price of a house we can say that `RMSE` is 100 dollars instead of `MSE` is 10 000 squared dollars.
There is one problem with `RMSE`. If we are predicting number bigger than 1, the square root of that number is lower than the number. In the other case, when the number is less than 1, the square root of that number is the bigger number.

## MEAN ABSOLUTE ERROR (MAE)

When we use `MAE` we get positive values and the same units as the data. The `MAE` corresponds to optimizing Laplace-distributed error.
If optimizing this loss function the model will be less influenced by outliers.

## SCALE INVARIANCE

Both `MSE` and `MAE` depend on the scale of data. 

## MEAN ABSOLUT PERCENTAGE ERROR (MAPE)

It shows how accurate the model is (in %). The reason to use that metric is to have scale invariant error. 
There is one problem with this error. In the denominator we have 'y' and if the value is 0 it gets to infinity. This can lead us to wrong assumptions because this error should only go to infinity if the predictions are wrong.





