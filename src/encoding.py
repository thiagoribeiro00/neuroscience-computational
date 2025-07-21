def rate_encoder(time_series, timesteps=1):
    """
    Codifica uma série temporal em spikes usando Rate Coding.
    A intensidade do sinal é tratada como a probabilidade de disparar um spike.

    Args:
        time_series (np.array): A série temporal pré-processada (valores entre 0 e 1).
        timesteps (int): Número de passos de tempo para a simulação SNN.
                         Para rate coding simples, 1 é suficiente por ponto de dado.
    """
    # Cria uma dimensão extra para os timesteps da SNN
    spike_train = torch.from_numpy(time_series).unsqueeze(0) # Shape: (1, sequence_length)
    return spike_train