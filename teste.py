import locale
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from rede_neural import RedeNeural

locale.setlocale(locale.LC_TIME, 'Portuguese_Brazil.1252')

input_size = 15
hidden_size = 3
output_size = 7

modelo = RedeNeural(input_size, hidden_size, output_size)
modelo.load_state_dict(torch.load("modelo.pth"))
custo = nn.MSELoss()
scaler_x, scaler_y = joblib.load("scaler_x.pkl"), joblib.load("scaler_y.pkl")

df = pd.read_excel("vazoes_jupia.xlsx")
df["DATA"] = pd.to_datetime(df["DATA"], format="%d/%b/%Y", dayfirst=True)

valores_teste = df[df["DATA"].dt.year > 2010]["VAZAO"].values.reshape(-1, 1)

teste_x = []
teste_y = []
for i in range(len(valores_teste) - input_size - output_size + 1):
  teste_x.append(valores_teste.flatten()[i:i + input_size])
  teste_y.append(valores_teste.flatten()[i + input_size:i + input_size + output_size])
tensor_x_teste = torch.tensor(scaler_x.transform(teste_x), dtype=torch.float32)
tensor_y_teste = torch.tensor(scaler_y.transform(teste_y), dtype=torch.float32)

modelo.eval()
with torch.no_grad():
  saida_teste = modelo(tensor_x_teste)
  perda_teste = custo(saida_teste, tensor_y_teste)
print(f"Erro médio quadrático no teste: {perda_teste.item():.6f}")

valores_reais = scaler_y.inverse_transform(tensor_y_teste.numpy())
valores_previstos = scaler_y.inverse_transform(saida_teste.numpy())

reais_plt = [None] * 7
previstos_plt = [None] * 7
for i in range(7):
  for j in range(len(valores_reais)):
    if not reais_plt[i]:
      reais_plt[i] = []
    reais_plt[i].append(valores_reais[j][i])
    if not previstos_plt[i]:
      previstos_plt[i] = []
    previstos_plt[i].append(valores_previstos[j][i])

# Gráfico de Valores Reais e Valores Previstos
std = np.std(valores_previstos)
x = range(len(valores_previstos))
for i in range(len(reais_plt)):
  plt.figure(figsize=(14, 6))
  plt.plot(x, reais_plt[i], label="Real", color="blue", ls="-")
  plt.plot(x, previstos_plt[i], label="Previsto", color="orange", ls="--")
  plt.fill_between(x, previstos_plt[i] - std, previstos_plt[i] + std, color="orange", alpha=0.2, label="Desvio Padrão")
  plt.title(f"Reais x Previstos - Dia {i + 1}")
  plt.xlabel("Dia")
  plt.ylabel("Valor")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(f"results/dia_{i + 1}.png")
  plt.show()

# Gráfico Valores Reais e Valores Previstos - Semana 1
std = np.std(valores_previstos[0])
x = range(len(valores_previstos[0]))
plt.figure(figsize=(14, 6))
plt.plot(x, valores_reais[0], label="Real", color="blue", ls="-")
plt.plot(x, valores_previstos[0], label="Previsto", color="orange", ls="--")
plt.fill_between(x, valores_previstos[0] - std, valores_previstos[0] + std, color="orange", alpha=0.2, label="Desvio Padrão")
plt.title("Reais x Previstos - Semana 1")
plt.xlabel("Dia")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/semana_1.png")
plt.show()

# Exporta os resultados como CSV
for i in range(len(reais_plt)):
  df_dia = pd.DataFrame({f"R{i + 1}": reais_plt[i], f"P{i + 1}": previstos_plt[i]})
  df_dia.to_csv(f"results/dia_{i + 1}.csv", index=False)

print("Resultados do teste salvos em 'results'.")