# README - Modelo de Árvore de Decisão para Classificação e Regressão

## Descrição
Este projeto implementa um modelo de árvore de decisão para tarefas de classificação e regressão usando o conjunto de dados `wine.data`. O código inclui:
- Carregamento e tratamento dos dados.
- Normalização das variáveis preditoras.
- Divisão dos dados em treino e teste.
- Balanceamento de classes para classificação.
- Otimização de hiperparâmetros com `GridSearchCV`.
- Treinamento e avaliação de modelos de classificação e regressão.
- Visualização de matrizes de confusão, distribuições de erros e estrutura da árvore.

## Dependências
Este projeto requer as seguintes bibliotecas Python:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Estrutura do Código
1. **Importação das bibliotecas**
2. **Carregamento e tratamento dos dados**
   - Remoção de valores ausentes
   - Normalização dos dados
   - Divisão em treino e teste
3. **Treinamento dos modelos**
   - Classificação (com e sem balanceamento)
   - Regressão (com e sem balanceamento)
4. **Avaliação dos modelos**
   - Cálculo de acurácia, erro médio absoluto e R²
   - Matriz de confusão e histograma dos erros
5. **Visualização da árvore de decisão**

## Como Executar
1. Certifique-se de que o arquivo `wine.data` esteja no caminho correto.
2. Execute o script com:
```bash
python arvoredecisao.py
```

## Resultados
O script retorna os melhores hiperparâmetros encontrados, métricas de desempenho e gráficos para análise visual da qualidade dos modelos.

