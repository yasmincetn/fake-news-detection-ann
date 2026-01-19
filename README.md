# fake-news-detection-ann
Fake News Detection using Artificial Neural Networks (ANN)

This repository contains a deep learning–based fake news detection project.

The project focuses on detecting fake news using an Artificial Neural Network (ANN) and systematically evaluates different architectural and optimization strategies.

📌 Project Overview

Problem Type: Binary Classification

Task: Fake News Detection

Course: CMPE430 – Artificial Neural Networks

Dataset Size: 23,196 samples

Frameworks: TensorFlow / Keras, NumPy, Pandas, Scikit-learn

Hyperparameter Optimization: Optuna

📊 Dataset

Name: Fake News Detection Dataset

Source: Kaggle
https://www.kaggle.com/datasets/algord/fake-news

Dataset Structure
Column Name	Description
title	News headline text
news_url	URL of the news article
source_domain	News source domain
tweet_num	Number of tweets
real	Target label (1 = Real, 0 = Fake)

Total Rows: 23,196

Target Variable: real

⚙️ Data Preprocessing

The following preprocessing steps were applied:

Tokenization of the title column using a vocabulary size of 5,000

Multi-hot encoding for textual features

Numerical feature (tweet_num) concatenated with text vectors

Unused columns (news_url, source_domain) removed

Dataset split:

70% Training

15% Validation

15% Test

Normalization and scaling were not applied since multi-hot encoding was used for text representation.

🧠 Model Architecture & Experiments
Baseline Model

Architecture: 1 hidden layer (16 neurons, ReLU)

Optimizer: Adam

Epochs: 10

Metric	Value
Test Loss	0.4196
Test Accuracy	0.8460
Experiment 1 – Increased Hidden Layers

Architecture: 2 hidden layers (32 + 16)

Result: Higher accuracy but increased overfitting

| Test Accuracy | 0.8635 |

Experiment 2 – Dropout Regularization

Dropout: 0.5

Result: Reduced overfitting, slight accuracy drop

| Test Accuracy | 0.8397 |

Experiment 3 – Increased Epochs

Epochs: 20

Result: Worse generalization, higher validation loss

| Test Accuracy | 0.8417 |

Experiment 4 – Optimizer Change

Optimizer: RMSprop

Epochs: 10

Result: Best overall performance

| Test Accuracy | 0.8509 |

Experiment 5 – Hyperparameter Optimization (Optuna)

Optimized parameters:

Hidden units

Dropout rate

Learning rate

Optimizer

Batch size

| Test Accuracy | 0.8463 |

Although Optuna achieved competitive results, Experiment 4 remained the best-performing model on the test set.

📈 Final Results Summary
Model	Test Accuracy	Overfitting Status
Baseline	0.8460	Underfitting
Exp 1	0.8635	Overfitting
Exp 2	0.8397	Overfitting
Exp 3	0.8417	Overfitting
Exp 4	0.8509	Best balance
Exp 5 (Optuna)	0.8463	Overfitting
🧾 Conclusion

This project demonstrates that model complexity does not always lead to better generalization. While deeper architectures and automated hyperparameter tuning (Optuna) were explored, the best-performing model was achieved using a simpler architecture combined with an appropriate optimizer (RMSprop).

The experiments highlight the importance of:

Step-by-step experimentation

Monitoring validation loss

Avoiding blind reliance on automated optimization tools

🚀 How to Run
pip install numpy pandas matplotlib scikit-learn tensorflow optuna kagglehub


Open and run:

fakenews.ipynb

👩‍💻 Author

Yasmin Çetin
