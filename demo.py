import torch
import matplotlib.pyplot as plt

# Generating synthetic data
temperatures = torch.tensor(
    [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]).float()
ice_creams_sold = torch.tensor(
    [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]).float()

plt.scatter(temperatures, ice_creams_sold)
plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Creams Sold')
plt.title('Ice Cream Sales vs Temperature')
plt.show()

# Simple Linear Regression Model
model = torch.nn.Linear(1, 1)  # One input feature, one output

# Optimizer and Loss Function
criterion = torch.nn.MSELoss()  # Mean Squared Error loss
# Optimizer: Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)


# Training the model
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(temperatures.view(-1, 1))
    loss = criterion(outputs, ice_creams_sold.view(-1, 1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


# Predicting ice creams sold for new temperatures
predicted_ice_creams = model(temperatures.view(-1, 1)).detach()

plt.scatter(temperatures, ice_creams_sold)
plt.plot(temperatures, predicted_ice_creams, 'r')
plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Creams Sold')
plt.title('Ice Cream Sales Prediction vs Temperature')
plt.show()
