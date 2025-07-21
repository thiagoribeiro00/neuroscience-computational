import torch
import torch.nn as nn
import norse.torch as norse

class SNNExoplanetDetector(nn.Module):
    """
    Modelo SNN para detectar exoplanetas.
    Arquitetura: Camada de entrada -> Camada recorrente LIF -> Camada de saída linear.
    """
    def __init__(self, input_features, hidden_features, output_features, record=False):
        super(SNNExoplanetDetector, self).__init__()
        self.hidden_features = hidden_features
        
        # Camada recorrente com neurônios Leaky Integrate-and-Fire (LIF)
        self.recurrent = norse.LIFRecc(input_features, hidden_features)
        
        # Camada linear de leitura para classificação
        self.fc_out = nn.Linear(hidden_features, output_features)

    def forward(self, x):
        # x tem shape (batch_size, sequence_length, input_features)
        # Inicializa o estado oculto da camada recorrente
        state = None
        
        # Processa a sequência temporalmente
        # A norse.LIFRecc retorna os spikes de saída para cada passo de tempo
        outputs, state = self.recurrent(x, state)
        
        # Usamos a soma dos spikes da camada oculta ao longo do tempo como
        # uma representação da atividade total para a classificação.
        # Outra opção seria usar o estado final da membrana.
        total_activity = torch.sum(outputs, dim=1)
        
        # Passa pela camada de saída para obter o logit
        out = self.fc_out(total_activity)
        return out

class LSTMExoplanetDetector(nn.Module):
    """
    Modelo LSTM de baseline para comparação.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMExoplanetDetector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x tem shape (batch_size, sequence_length, input_size)
        # Inicializa os estados h_0 e c_0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Passa pela LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decodifica o estado oculto do último passo de tempo
        out = self.fc(out[:, -1, :])
        return out