# Multivariate Linear Regression Using C++

This project was quite a leap in scale compared to my original Basic Linear Regression Using C++:
https://github.com/hl0228057-cmd/Basic-Linear-Regression-Using-Cpp

Besides the addition of multiple input features instead of a single feature, I also started implementing object-oriented programming into my machine learning projects. The model is now structured as a reusable class with functions such as fit() and predict(), inspired by the design style of libraries like Scikit-Learn.

This project was implemented entirely from scratch in C++ without using any machine learning libraries. The main goal was to better understand how multivariable machine learning models work internally rather than relying on prebuilt abstractions. I coded everything with some documentation and some Google searches, however, I did use ChatGPT for generating the data for inputs and expectedOutputs.

Some of the largest improvements compared to my original project include:

- Dot product for scalable vector operations
- Feature normalization for more stable training
- Dynamic feature support instead of hardcoded equations
- Reusable fit() and predict() abstractions

The features of the model are sleep, study time, outdoor activity, and time spent gaming (all are measured in hours)

How it works:

The model first normalizes the input features so that all values are scaled between 0 and 1. It then initializes weights and bias as 0 and calculates the initial loss.

After that, the model enters the training loop, which repeatedly:
- Makes predictions using dot product
- Calculates prediction error
- Computes total loss using mean squared error (MSE)
- Adjusts weights and bias using gradient descent

Once training is complete, the model can make predictions using new input feature vectors through the predict() function.

Example:
- inputs: {8, 3, 1, 3}
- output: ~84

Working on this project helped me better understand how larger machine learning systems scale beyond single-variable equations while also improving my software engineering and abstraction skills.
