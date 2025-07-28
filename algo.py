import matplotlib.pyplot as plt
n_estimators = [150, 200, 300, 500]
accuracies = [0.81, 0.82, 0.824, 0.823]  # Ejemplo basado en tu resultado
plt.plot(n_estimators, accuracies, marker='o')
plt.xlabel('Número de Árboles')
plt.ylabel('Precisión')
plt.title('Precisión vs. Número de Árboles')
plt.savefig('precision_plot.png')
plt.show()