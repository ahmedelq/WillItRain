# Will It Rain? 
## About 
This project goal is to predict whether it will rain on a certain day given some measurable variables (n=22) such as temperature,  evaporation, humidity and wind-speed. The data was outsourced from Kaggle ([Rain in Australia](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)). After data engineering, standardization, splitting, and balancing. I designed a neural network model that achieved 84% accuracy and F1 of 0.6689. 

## Data Description
There are 22 features and 1 target, 15 of these features are numerical, 6 are categorical and 1 is date-time, with 145,460 observations. I filled empty numerical values with the column's mean, after that I standardized the values, and removed outliers whose values exceeded [-3, 3]. I filled the empty categorical values with the last value used near that empty observation and converted them into numerical labels. Lastly for the date-time values I converted them into cyclical values using trigonometric functions.

The data was split into 90% Training data, 5% validation data and 5% testing data. After that I balanced the training data using SMOTE, resulting in (n = 189,680) trainable observations.  

## Model

I designed a neural network model with 6 layers sperated by ReLU activations, batch normalization and dropouts, except for the output layer which I used a sigmoid function.  I also included embeddings to encode categorical data into trainable vectors. Finally I used Adam optimizer with cross entropy loss. 

ANN Results: 

|          | Training | Validation | Testing  |
| -------- | -------- | ---------- | -------- |
| Loss     | 0.339279 | 0.365065   | 0.362375 |
| Accuracy | 84.71%   | 84.23%     | 84.02%   |
| F1       | 0.8428   | 0.6689     | 0.6517   |

Random Forest Tree classifier results:
|          | Training | Validation | Testing  |
| -------- | -------- | ---------- | -------- |
| Accuracy | 100%     | 84.89%     | 85.58%   |
| F1       | 1.0      | 0.6473     | 0.6448   |


![img](https://i.imgur.com/5wLuA2z.png)

![img](https://i.imgur.com/bo0OCte.png)![img](https://i.imgur.com/6fTXLHl.png)![img](https://i.imgur.com/kSwNiBR.png)

## Tools

- Numpy
- Pandas
- Sklearn
- PyTorch
- imbalanced-learn
- seaborn & matplotlib
