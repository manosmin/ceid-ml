
# Data Mining & Machine Learning Project @CEID

This project combines data analysis, machine learning, and natural language processing techniques to address questions related to energy demand and sentiment analysis on product reviews. 



## Features

- Perform basic **statistical analysis** on energy demand and source data. Calculate key statistics such as mean, median, standard deviation, and quartiles
- Create **graphical representations** to visualize the temporal trends in energy demand and source availability
- Utilize **K-Means** clustering algorithm to group similar days based on energy demand and source data
- Train a regressor using **LSTM neural networks** to predict the amount of energy required from non-renewable sources at each moment of the day
- Predict product ratings using a **RandomForest** model based on text reviews


## Tech Stack

**Back End:** Python

**ML Libraries:** scikit-learn, Tensorflow, Keras, NLTK

**Data Analysis Libraries:** pandas, NumPy, statsmodels

**Data Visualization Libraries:** Matplotlib


## Deployment

To create the needed csv files for the project run

```bash
  python MergeCSV.py
```


## Screenshots
*Data Analysis*</br>
![App Screenshot](https://github.com/manosmin/ceid-ml/blob/master/screenshots/ss1.png)</br>
![App Screenshot](https://github.com/manosmin/ceid-ml/blob/master/screenshots/ss2.png)</br>
*Outliers Detection using K-Means*</br>
![App Screenshot](https://github.com/manosmin/ceid-ml/blob/master/screenshots/ss4.png)</br>
![App Screenshot](https://github.com/manosmin/ceid-ml/blob/master/screenshots/ss3.PNG)</br>
*Evaluation*</br>
![App Screenshot](https://github.com/manosmin/ceid-ml/blob/master/screenshots/ss5.png)

