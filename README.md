# NN Regression with TensorFlow on Boston Housing Data

## Project Overview

This project focuses on building a neural network regression model using TensorFlow to predict house prices based on the Boston Housing dataset. The goal is to create a predictive model that accurately estimates home prices, providing insights into the key factors influencing real estate value.

## Dataset

- **Source:** Boston Housing dataset, containing features like the number of rooms, crime rate, property tax rate, and proximity to employment centers.
- **Features:** Key features include average number of rooms per dwelling, property tax rate, and pupil-teacher ratio, among others.

## Tools & Libraries Used

- **Data Analysis:**
  - `Pandas` for data manipulation.
  - `Matplotlib` and `Seaborn` for data visualization.
- **Neural Network Development:**
  - `TensorFlow` and `Keras` for building and training the regression model.
- **Model Evaluation:**
  - Metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE) to evaluate model performance.

## Methodology

### Data Exploration:

- Conducted exploratory data analysis (EDA) to understand feature distributions and their relationship with house prices.

### Data Preprocessing:

- Normalized numerical features and encoded categorical variables.
- Split the data into training and testing sets.

### Model Development:

- Built a neural network regression model using TensorFlow and Keras, consisting of multiple layers with ReLU activation functions.
- Applied techniques like dropout to prevent overfitting.

### Model Training:

- Trained the neural network on the preprocessed data using MSE as the loss function and the Adam optimizer.
- Monitored performance using validation data.

### Model Evaluation:

- Evaluated the model's performance on the test set using MSE and MAE.
- Visualized the model's predictions against actual house prices.

## Results

The neural network model provided accurate predictions for house prices in the Boston area, identifying key factors like the number of rooms and property tax rate as significant predictors.

## Conclusion

This project successfully demonstrated the use of a neural network regression model to predict house prices. The results show that deep learning can effectively capture complex relationships in real estate data.

## Future Work

- Experiment with different neural network architectures and hyperparameter tuning.
- Incorporate additional data, such as recent sales trends or neighborhood amenities, to enhance predictions.
- Deploy the model as a web service for real-time price predictions.
