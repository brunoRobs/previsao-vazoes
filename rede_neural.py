import torch
import torch.nn as nn

class RedeNeural(nn.Module): 
  def __init__(self, input_size: int, hidden_size: int, output_size: int):
    super(RedeNeural, self).__init__()
    self.input = nn.Linear(input_size, hidden_size)
    self.hidden = nn.Sigmoid()
    self.output = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x = self.input(x)
    x = self.hidden(x)
    x = self.output(x)
    return x