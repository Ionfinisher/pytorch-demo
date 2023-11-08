import torch  # librairire de pytorch
import matplotlib.pyplot as plt  # librairie pour afficher des graphiques


# Générons des données d'exemple.
# La fonction tensor() transforme une liste en un tensor.
# Les tensors sont une structure de données spécialisée très similaire aux tableaux et aux matrices.
temperatures = torch.tensor(
    [-10, -5, 0, 5, 10, 20, 25, 30, 35, 40]).float()
creme_vendues = torch.tensor(
    [0, 1, 3, 3, 4, 6, 40, 75, 80, 90]).float()


# Toutes les fonctions de plt permettent de tracer des graphiques
# Avec en x la liste températures et en y la liste creme_vendues
plt.scatter(temperatures, creme_vendues)
plt.xlabel('Température (°C)')
plt.ylabel('Cremes vendues')
plt.title('Cremes vendues vs Température')
plt.show()

# Nous créons maintenent un Modèle de régression linéaire simple
# qui permet de prédire les cremes vendus en fonction de la température
model = torch.nn.Linear(1, 1)  # une entrée et une sortie

# Cette fonction mesure l'efficacité du modele à prédire
# les ventes de glaces en calculant la différence moyenne entre
# les ventes prédites et les ventes réelles.
criterion = torch.nn.MSELoss()  # Mean Squared Error loss


# Cette fonction est un algorithme qui aide le modele
# à améliorer ses prédictions en s'ajustant sur la base
# des erreurs qu'elle a commises lors de ses prédictions.
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


# Prediction des ventes de cremes pour les temperatures
predicted_ice_creams = model(temperatures.view(-1, 1)).detach()

plt.scatter(temperatures, creme_vendues)
plt.plot(temperatures, predicted_ice_creams, 'r')
plt.xlabel('Temperature (°C)')
plt.ylabel('Cremes vendues')
plt.title('Prediction des ventes de creme vs Temperature')
plt.show()
