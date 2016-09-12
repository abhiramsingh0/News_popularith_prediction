This file gives description of code done for Assignment 1

main.m is the starting file for program.

1. data points are featched from the given files.
2. data points are separated in attributes X and target y variables.
3. second order and third order polynomial terms are added into features X.
4. features X are normalized as (xi - mean)/std.
5. intercept term is added to features.
5. data set is "randomly divided" into training(X_train,y_train)(80%), validation(X_cv,y_cv)(10%) and test set(X_test,y_test)(10%).
	due to random division it may select different regularization parameter.
6. Ridge regression is used for error computation and its implementation is given in file computeCostMulti.m
7. for updating weights, gradient descent is used and its implementation is in gradientDescentMulti.m
8. model is trained using training data set (X_train, y_train) and then 
9. lambda is determined by calling gradient descent for diffent values of lambda and finding minimum RMSE for cross validation set.
10.for selected lambda, test error is calculated.
11. model is trained over all the data set for selected lambda and then test result is generated for given test file.

-------------------------------------------------------------------------------------------------------------------------------------
Command to run the program
--------------------------------------------------------------------------------------------------------------------------------------
1. from matlab command window
move to the destination folder and type script file name (model.m is the file)and press enter:
model

2. from linux shell:
matlab -nodisplay -nodesktop -r model
after program execution, type exit and hit enter
