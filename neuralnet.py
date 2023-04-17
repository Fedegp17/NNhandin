import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#import the data set ,evaluate it and preprocess the data
df=pd.read_csv("GYM.csv")
cols_to_drop = ['playerId', 'Name','Equipment']
# Drop columns from dataframe
df = df.drop(cols_to_drop, axis=1)
df[['Age', 'BodyweightKg', 'BestSquatKg','BestDeadliftKg']] = df[['Age', 'BodyweightKg','BestSquatKg','BestDeadliftKg']].apply(lambda x: x.astype(float))
df['Sex'] = df['Sex'].map({'F': 0, 'M': 1})
print(df.head())

x=df[['Age','BestDeadliftKg','BodyweightKg','BestSquatKg']] #Indexing all numerical values in the dataframe
y=df['Sex'] #target value, in this case the Best squat
y = df.Sex.values  # target value, in this case the Best squat
# split the data in training and testing sets
x = x.to_numpy()
y = y.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
# Define the number of neurons in each layer
input_neurons = 4
hidden_neurons = 5
output_neurons = 2

# Initialize the weights and biases
W1 = np.random.randn(input_neurons, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))
W2 = np.random.randn(hidden_neurons, output_neurons)
b2 = np.zeros((1, output_neurons))

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the softmax activation function
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Define the forward pass
def forward(X, W1, b1, W2, b2):
    hidden_layer_activation = sigmoid(np.dot(X, W1) + b1)
    output_activation = softmax(np.dot(hidden_layer_activation, W2) + b2)
    return hidden_layer_activation, output_activation


# Define the cross-entropy loss function
def loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred))



# Define the learning rate and number of epochs
learning_rate = 0.01
num_epochs = 10000

# Train the neural network using gradient descent
for epoch in range(num_epochs):

    # Forward pass
    hidden_layer_activation, y_pred = forward(x, W1, b1, W2, b2)

    # Calculate the loss and print it every 100 epochs
    ce_loss = loss(y, y_pred)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Cross-entropy loss = {ce_loss:.9f}")

    # Backward pass
    grad_y_pred = (y_pred - y) / len(y)
    grad_W2 = np.dot(hidden_layer_activation.T, grad_y_pred)
    grad_b2 = np.sum(grad_y_pred, axis=0, keepdims=True)
    grad_hidden = np.dot(grad_y_pred, W2.T) * (hidden_layer_activation * (1 - hidden_layer_activation))
    grad_W1 = np.dot(x.T, grad_hidden)
    grad_b1 = np.sum(grad_hidden, axis=0, keepdims=True)

    # Update the weights and biases
    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1
    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2

# Print the final weights and biases
print("Final weights:")
print(W1)
print(W2)
print("Final biases:")
print(b1)
print(b2)



# Make predictions on new data


while True:
    print("type in just numbers  ")
    age = float(input("How old are you? "))
    bmi = float(input("What is your Body weight?(Kg) "))
    Max_dl = float(input("Whats is your max deadlift?(Kg)  ")) 
    Max_squat = float(input("Whats is your max squat?(Kg)  "))

    X_user=[age,Max_dl,bmi,Max_squat]
    X_user=np.array(X_user)
    X_user=X_user.reshape(1,-1)
    
    X_new = X_user
    _, y_new = forward(X_new, W1, b1, W2, b2)

    print("Predictions:")
    print(y_new)

    if y_new[0][0]>y_new[0][1]:
        print("You are a Male")
    else:
        print("You are a female")
    
    user =input("Wanna make another prediction hommie?")
    

    if user=="n":
        break
