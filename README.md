# Neural-Network-from-Scratch
A basic Artificial Neural Network(ANN) model made from scratch, predicting if a person will get heart diseases

This project implements a **fully manual feedforward neural network** (1 hidden layer) from scratch using NumPy, pandas, and no deep learning frameworks. It trains on a heart disease classification dataset.

---

## Features

- Manual forward & backward propagation
- ReLU and Sigmoid activation functions
- Binary cross-entropy loss
- One-hot encoding for categorical features
- Z-score normalization
- Simple manual evaluation and prediction

---

## Dataset

This project uses the `heart.csv` dataset, which is in the project directory.

Columns:
- `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina` ,`Oldpeak`, `ST_Slope`
- `HeartDisease` (binary target)

---

## Output Files
weights.csv: Hidden layer weights

bias.csv: Hidden layer biases

Each row corresponds to one neuron.
