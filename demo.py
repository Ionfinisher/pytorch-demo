import torch
import matplotlib.pyplot as plt


temperatures = torch.tensor(
    [-10, -5, 0, 5, 10, 20, 25, 30, 35, 40]).float()
creme_vendues = torch.tensor(
    [0, 1, 3, 3, 4, 6, 40, 75, 80, 90]).float()


plt.scatter(temperatures, creme_vendues)
plt.xlabel('Température (°C)')
plt.ylabel('Cremes vendues')
plt.title('Cremes vendues vs Température')
plt.show()


model = torch.nn.Linear(1, 1)


criterion = torch.nn.MSELoss()


optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)


num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(temperatures.view(-1, 1))
    loss = criterion(outputs, creme_vendues.view(-1, 1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


predicted_ice_creams = model(temperatures.view(-1, 1)).detach()

plt.scatter(temperatures, creme_vendues)
plt.plot(temperatures, predicted_ice_creams, 'r')
plt.xlabel('Temperature (°C)')
plt.ylabel('Cremes vendues')
plt.title('Prediction des ventes de creme vs Temperature')
plt.show()
