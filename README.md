*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Your Project Title Here

*TODO:* Write a short introduction to your project.

In this project I explore both AutoML and Scikit-learn random forest model (that has its hyperparameters tuned with HyperDrive) in a task of predicting
stock market. More specifically, I'm trying to predict whether the Dow Jones Industrial Average goes up or down based on several fundamental and technical
features. 


## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.



## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.
This dataset is part of a data from UCI Machine Learning repository: https://archive.ics.uci.edu/ml/datasets/CNNpred%3A+CNN-based+stock+market+prediction+using+a+diverse+set+of+variables

In the original study they used not just DOW but also Nasdaq and other indexes in attempt to create a convolutional neural network for cross market prediction.
In this study I took a slightly less ambitious goal, and tried to just predict the movement of DOW.

I pre-processed the data a little bit by creating a new binary indicator feature that measured whether the stock index would go up or down the following trading day.
This indicator was then used as the label I tried to predict.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

The task is to predict the overall movement of the stock market, as measured by the Dow. The dataset contains 82 pre-calculated features, full list of
which can be found in the appendix of this paper https://arxiv.org/pdf/1810.08923.pdf. In summary, these features include things like  
* Relative change of volume
* 10 days Exponential Moving Average
* Relative change of oil price(Brent) 
* Relative change in US dollar to Japanese yen exchange rate 

and so on, for each of the days.  in total the dataset contained data for 1984 days.

One important thing to keep in mind when doing prediction is not to accidentally "look into future". To avoid using future
data, I made sure that even the pre-processing steps only used data from the past. That is, when doing the normal
z-score standardization, I first calculated the mean and std from the first month of data (20 trading days), and
then used them in centering and normalizing the data:

``
normalized_df=(df3-df3.iloc[0:20].min())/(df3.iloc[0:20].max()-df3.iloc[0:20].min())
``

Similarly, when imputting missing features, I did not use things like median of all values for the feature (including 
the future), but a padding strategy where datapoints were filled based on last previously seen value. That way the
same approach could be used in real-life scenario where only past is known.

### Access
*TODO*: Explain how you are accessing the data in your workspace.
I uploaded the raw .csv file to the Azure ML Studio, then did some pre-processing as described earlier, and saved
pre-processed version to separate .csv file. Then I created a new dataset out of it that I registered to the 
workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment


I chose AUC_weighted as the primary metric as it is suitable for a classification task.
Some of the settings like experiment time out were chosen as something that felt sensible.
The task was classification because I try to classify between situations where the DOW goes
up and down, and this label was stored in feature 'y' which was set as the label_column_name.
![AutoML configurations](screenshots/automl_confs.jpg)

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

The AutoML performed reasonably well (above 58%) considering that the task is very difficult (predicting stock market). 

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

For some reason the widget did not really update in my notebook even though I was in the Azure ML environment
![Run details widget](screenshots/automl_widget.jpg)

So here are also the details from the Azure ML Studio:
![Details from ML Studio](screenshots/run_details_in_studio.jpg)

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

I chose random forest model, as I have had positive experience with it in the past. 
I explored hypertuning the following parameters:
![Random forest hyperparameters](screenshots/hyperdrive_parameters.jpg)

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
The random forest performed worse than AutoML, but still managed to edge slighty over chance level with 0.523 AUC score.
Performing feature selection before training the model could be useful to improve the performance. 

![Random forest run details](screenshots/hyper_best_model.jpg)

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

![Hyperparameter rundetails](screenshots/hyperdrive_run_complete.jpg)

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

I deployed the best AutoML as an AciWebservice that can be accessed via REST API:

![Consuming model endpoint](screenshots/consuming_model_endpoint.jpg)

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
