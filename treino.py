import locale
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import joblib
import os
from rede_neural import RedeNeural
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

locale.setlocale(locale.LC_TIME, 'Portuguese_Brazil.1252')
os.makedirs("results", exist_ok=True)

input_size = 15
hidden_size = 3
output_size = 7

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

df = pd.read_excel("vazoes_jupia.xlsx")
df["DATA"] = pd.to_datetime(df["DATA"], format="%d/%b/%Y", dayfirst=True)
valores_treinamento = df[df["DATA"].dt.year < 2011]["VAZAO"].values.reshape(-1, 1)

treinamento_x = []
treinamento_y = []
for i in range(len(valores_treinamento) - input_size - output_size + 1):
  treinamento_x.append(valores_treinamento.flatten()[i:i + input_size])
  treinamento_y.append(valores_treinamento.flatten()[i + input_size:i + input_size + output_size])
treinamento_x = np.array(treinamento_x)
treinamento_y = np.array(treinamento_y)

k_folds = 5

perdas_treino_folds = []
perdas_val_folds = []

tscv = TimeSeriesSplit(n_splits=k_folds)

for fold, (idx_treino, idx_val) in enumerate(tscv.split(treinamento_x)):
  print(f"Iniciando fold {fold + 1}/{k_folds}")

  modelo = RedeNeural(input_size, hidden_size, output_size)
  otimizador = optim.Adam(params=modelo.parameters(), lr=0.01)
  custo = nn.MSELoss()

  scaler_x = MinMaxScaler()
  scaler_y = MinMaxScaler()

  tensor_x_treino = torch.tensor(scaler_x.fit_transform(treinamento_x[idx_treino]), dtype=torch.float32)
  tensor_y_treino = torch.tensor(scaler_y.fit_transform(treinamento_y[idx_treino]), dtype=torch.float32)
  tensor_x_val = torch.tensor(scaler_x.transform(treinamento_x[idx_val]), dtype=torch.float32)
  tensor_y_val = torch.tensor(scaler_y.transform(treinamento_y[idx_val]), dtype=torch.float32)

  ciclos = 2000
  limiar_melhoria = 1e-5
  melhor_perda_val = float("inf")
  paciencia = 30
  ciclos_sem_melhoria = 0
  melhor_modelo = None
  melhor_scaler_x = None
  melhor_scaler_y = None

  perdas_treino_fold_atual = []
  perdas_val_fold_atual = []

  for ciclo in range(ciclos):
    # Treinamento
    modelo.train()
    otimizador.zero_grad()
    saida_treino = modelo(tensor_x_treino)
    perda_treino = custo(saida_treino, tensor_y_treino)
    perda_treino.backward()
    otimizador.step()
    perdas_treino_fold_atual.append(perda_treino.item())

    # Validação
    modelo.eval()
    with torch.no_grad():
      saida_val = modelo(tensor_x_val)
      perda_val = custo(saida_val, tensor_y_val)
      perdas_val_fold_atual.append(perda_val.item())
    if melhor_perda_val - perda_val.item() > limiar_melhoria:
      print(f'Ciclo {ciclo + 1} - Treino: {perda_treino.item():.6f} - Validação: {perda_val.item():.6f}')
      melhor_perda_val = perda_val.item()
      ciclos_sem_melhoria = 0
      melhor_modelo = deepcopy(modelo.state_dict())
      melhor_scaler_x = deepcopy(scaler_x)
      melhor_scaler_y = deepcopy(scaler_y)
    else:
      ciclos_sem_melhoria += 1
    if ciclos_sem_melhoria > paciencia:
      print(f"Early stopping ativado no ciclo {ciclo + 1}.")
      break

  print(f"Fim do Fold {fold + 1} - Melhor Perda de Validação: {melhor_perda_val:.6f}")
  perdas_treino_folds.append(perdas_treino_fold_atual)
  perdas_val_folds.append(perdas_val_fold_atual)

torch.save(melhor_modelo, "modelo.pth")
joblib.dump(melhor_scaler_x, "scaler_x.pkl")
joblib.dump(melhor_scaler_y, "scaler_y.pkl")
print("Modelo e scalers salvos em 'modelo.pth', 'scaler_x.pkl' e 'scaler_y.pkl'.")

# Número máximo de ciclos que qualquer fold alcançou
max_epochs = 0
for hist in perdas_treino_folds:
    if len(hist) > max_epochs:
      max_epochs = len(hist)

# Arrays NumPy preenchidos com NaN para lidar com o early stopping
perdas_treino = np.full((k_folds, max_epochs), np.nan)
perdas_val = np.full((k_folds, max_epochs), np.nan)

# Preencher os arrays com os dados de perda
for i, hist in enumerate(perdas_treino_folds):
    perdas_treino[i, :len(hist)] = hist
for i, hist in enumerate(perdas_val_folds):
    perdas_val[i, :len(hist)] = hist

# Calcular a média e o desvio padrão, ignorando os NaNs
media_perda_treino = np.nanmean(perdas_treino, axis=0)
std_perda_treino = np.nanstd(perdas_treino, axis=0)

media_perda_val = np.nanmean(perdas_val, axis=0)
std_perda_val = np.nanstd(perdas_val, axis=0)

# Gerar o gráfico
fig, ax = plt.subplots(figsize=(12, 7))

# Curva média de treino
ax.plot(media_perda_treino, color='royalblue', label='Perda Média de Treino', lw=1)
ax.fill_between(range(max_epochs), media_perda_treino - std_perda_treino, media_perda_treino + std_perda_treino, color='royalblue', alpha=0.2)
# Curva média de validação
ax.plot(media_perda_val, color='darkorange', label='Perda Média de Validação', lw=1.5)
ax.fill_between(range(max_epochs), media_perda_val - std_perda_val, media_perda_val + std_perda_val, color='darkorange', alpha=0.2)

# Ajustes visuais
ax.set_title(f'Curvas de Perda Médias da Validação Cruzada ({k_folds} Folds)', fontsize=16, weight='bold')
ax.set_xlabel('Ciclo (Época)', fontsize=12)
ax.set_ylabel('Perda (MSE)', fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(bottom=0)
ax.set_xlim(left=0, right=max_epochs)

plt.tight_layout()
plt.savefig("results/media_perdas_validacao_cruzada.png")
plt.show()