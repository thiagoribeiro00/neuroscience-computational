# Projeto Exoplaneta SNN

## 🌌 Projeto: Detecção de Exoplanetas com Spiking Neural Networks (SNNs)

### 📌 Descrição Geral
Este projeto propõe o uso de redes neurais espinhosas (SNNs) para detectar exoplanetas a partir das curvas de luz fornecidas pelo telescópio TESS da NASA. A abordagem se baseia no fato de que SNNs são eficientes em lidar com séries temporais esparsas, como os dados de brilho estelar que apresentam quedas sutis e periódicas quando um planeta transita na frente de sua estrela.

### 🧭 Objetivo Final
Detectar automaticamente se uma curva de luz contém um trânsito planetário (indicando um exoplaneta) ou não, utilizando uma rede SNN treinada com dados reais do TESS.

### 🧱 Etapas do Projeto

1.  **📥 Coleta e Preparação de Dados**: Fonte: MAST Archive – TESS Mission. Pré-processamento para remover ruído, normalizar e inverter o sinal de trânsito.
2.  **🔁 Codificação Temporal (Spike Encoding)**: Converter a série temporal contínua em trens de spikes usando métodos como Rate ou Latency Coding.
3.  **🧠 Construção da Rede Espinhosa (SNN)**: Usar o framework Norse para construir uma arquitetura com camadas de neurônios LIF.
4.  **🏋️ Treinamento e Validação**: Treinamento supervisionado com divisão de dados (70/15/15) e otimizadores como Adam.
5.  **📊 Avaliação e Métricas de Sucesso**: Avaliar o modelo com AUC-ROC, F1-Score, Precisão e Recall.
6.  **📈 Comparação com baseline tradicional**: Comparar a performance da SNN com uma MLP ou LSTM.

### ✅ Critérios de sucesso do projeto
| Critério | Valor Esperado |
|---|---|
| AUC-ROC | ≥ 0.85 (bom) / ≥ 0.90 (excelente) |
| F1-Score | ≥ 0.80 |
| Detecção correta de trânsitos | ≥ 85% |
| Comparativo com rede tradicional | SNN deve atingir performance similar com menor custo computacional |