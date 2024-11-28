# Resumo das Aulas de Aprendizado de Máquina

Este repositório contém um resumo detalhado das aulas do curso de aprendizado de máquina, organizado por tema e com fórmulas matemáticas formatadas para renderização no GitHub.

---

## 📖 Conteúdo

# Resumos dos PDFs

## **[slides01.pdf](https://work.caltech.edu/slides/slides01.pdf)**
### Introdução ao Problema de Aprendizado
#### Exemplo Prático: Avaliação de Filmes
- Objetivo: Usar aprendizado de máquina para prever como um espectador avaliará um filme.
  - Fatores avaliados: gênero, popularidade, atores (ex.: "Tom Cruise").
  - Solução: Combinar contribuições de fatores relacionados ao filme e ao espectador.

#### Componentes do Aprendizado
1. **Entrada (\(x\))**: Dados conhecidos (ex.: idade, salário).
2. **Saída (\(y\))**: Decisão ou valor previsto (ex.: bom ou mau cliente).
3. **Função Alvo (\(f\))**: Regra ideal, mas desconhecida.
4. **Conjunto de Hipóteses (\(H\))**: Possíveis fórmulas candidatas.
5. **Algoritmo de Aprendizado**: Escolhe uma hipótese \( g \) que melhor aproxima \( f \).

#### Modelo Simples: Perceptron
- Usado para decisões lineares.
- Exemplo: Aprovação de crédito com base em dados do cliente.
  - Fórmula:

$$
h(x) = \text{sign}\left(\sum_{i=1}^{d} w_i x_i - \text{threshold}\right)
$$

---

## **[slides02.pdf](https://work.caltech.edu/slides/slides02.pdf)**
### É Viável Aprender?
#### Probabilidade e Generalização
- Sem generalizar, não é possível prever além dos dados observados.
- Exemplo: Experimento com urnas e Hoeffding's Inequality:
  - Em um experimento de sorteio de bolinhas, a fração observada (\(\nu\)) tende a aproximar a probabilidade real (\(\mu\)).
  - Fórmula:

$$
P[|\nu - \mu| > \epsilon] \leq 2e^{-2\epsilon^2N}
$$

#### Ligação com Aprendizado
- No aprendizado, a "urna" representa um conjunto de hipóteses.
- O aprendizado generaliza se a hipótese escolhida (\(h\)) funcionar bem tanto para os dados de treinamento (\(\text{Ein}\)) quanto para dados desconhecidos (\(\text{Eout}\)).

---

## **[slides03.pdf](https://work.caltech.edu/slides/slides03.pdf)**
### Modelos Lineares I
#### Representação de Entrada
- Dados brutos (\(x\)) podem ser transformados em características mais úteis (ex.: intensidade, simetria).
- Um modelo linear usa pesos (\(w\)) para combinar essas características.

#### Classificação Linear vs Regressão Linear
- **Classificação**: Toma decisões binárias, como "sim" ou "não".
- **Regressão**: Prevê valores contínuos, como "quanto aprovar de crédito".

#### Medindo o Erro
- Regressão usa o erro quadrático médio:

$$
\text{Ein}(h) = \frac{1}{N} \sum_{n=1}^N (h(x_n) - y_n)^2
$$

---

## **[slides04.pdf](https://work.caltech.edu/slides/slides04.pdf)**
### Erro e Ruído
#### Transformações Não-Lineares
- Usadas para tratar dados não-linearmente separáveis.
- Exemplo: Aplicar uma função \(\phi(x)\) para mapear os dados para um espaço onde possam ser separados.

#### Medidas de Erro
- Escolher a medida de erro adequada depende do problema.
- Exemplos:
  - **Erro Quadrático**: Para problemas contínuos.
  - **Erro Binário**: Para decisões "certo" ou "errado".

#### Impacto do Ruído
- Em muitos problemas, a saída observada (\(y\)) é uma combinação da função alvo (\(f\)) mais um termo de ruído.

---

## **[slides05.pdf](https://work.caltech.edu/slides/slides05.pdf)**
### Treinamento vs Teste
#### Diferenças
- **Treinamento**: Dados usados para ajustar o modelo.
- **Teste**: Dados usados para avaliar a generalização.

#### Noção de "Break Point"
- Ponto onde o modelo não consegue mais separar corretamente os dados, indicando sobreajuste.

#### Função de Crescimento
- Mede o número de divisões que um modelo pode fazer em \(N\) pontos:

$$
m_H(N) \leq 2^N
$$

---

## **[slides06.pdf](https://work.caltech.edu/slides/slides06.pdf)**
### Teoria da Generalização
#### Prova de que \(m_H(N)\) é Polinomial
- \(m_H(N)\) cresce de forma limitada com o aumento dos dados.
- Fórmula recursiva relaciona o número máximo de divisões com o tamanho do conjunto:

$$
m_H(N) \leq \sum_{i=0}^{k-1} \binom{N}{i}
$$

#### Implicações
- Um crescimento controlado de \(m_H(N)\) indica boa generalização.

---

## **[slides07.pdf](https://work.caltech.edu/slides/slides07.pdf)**
### Dimensão VC
#### Definição
- A dimensão VC (\(d_{VC}\)) é o número máximo de pontos que podem ser separados completamente por um conjunto de hipóteses.

#### Aplicações
- Um \(d_{VC}\) maior indica maior capacidade do modelo, mas pode levar a sobreajuste.

---

## **[slides08.pdf](https://work.caltech.edu/slides/slides08.pdf)**
### Trade-Off Bias-Variância
#### Conceitos
- **Bias**: Erro devido à simplicidade do modelo.
- **Variância**: Erro devido à sensibilidade aos dados.

#### Curvas de Aprendizado
- Mostram como o erro muda com o aumento do número de dados (\(N\)).

---

## **[slides09.pdf](https://work.caltech.edu/slides/slides09.pdf)**
### Modelo Linear II
#### Transformações Não-Lineares
- Adicionar dimensões (ex.: \(x^2\)) para lidar com dados mais complexos.
- Exemplo: Transformar \(x\) em:

$$
z = (x_1, x_2, x_1x_2, x_1^2, x_2^2)
$$

#### Regressão Logística
- Modelo probabilístico:

$$
h(x) = \frac{1}{1 + e^{-w^Tx}}
$$

---

## **[slides10.pdf](https://work.caltech.edu/slides/slides10.pdf)**
### Redes Neurais
#### Stochastic Gradient Descent (SGD)
- Método iterativo para ajustar pesos.
- Escolhe exemplos aleatórios para calcular gradientes, reduzindo custo computacional.

#### Modelo de Rede Neural
- Combinação de perceptrons organizados em camadas.
- Exemplo: Usar múltiplas camadas para resolver problemas não-linearmente separáveis.

#### Backpropagation
- Algoritmo para ajustar pesos em redes profundas de forma eficiente.

---

## **[slides11.pdf](https://work.caltech.edu/slides/slides11.pdf)**
### Overfitting
- **Definição**: Ajustar os dados além do necessário, capturando o ruído em vez do padrão.
- **Impactos**:
  - \(\text{Ein} \downarrow, \text{Eout} \uparrow\): Perda de generalização.
  - **Ruído determinístico**: Parte da função alvo (\(f(x)\)) que o modelo não consegue capturar.

#### Fórmula do ruído determinístico:
$$
\text{Ruído determinístico} = f(x) - h^*(x)
$$

- **Caso prático**: Ajuste de dados com diferentes ordens polinomiais:
  - Modelo de 10ª ordem captura ruído.
  - Modelo de 2ª ordem generaliza melhor.

---

## **[slides12.pdf](https://work.caltech.edu/slides/slides12.pdf)**
### Regularização
- **Motivação**: Evitar overfitting restringindo a complexidade do modelo.
- **Erro Augmentado**:

$$
E_{\text{aug}}(w) = E_{\text{in}}(w) + \frac{\lambda}{N} w^T w
$$

- **Solução regularizada**:

$$
w_{\text{reg}} = (Z^T Z + \lambda I)^{-1} Z^T y
$$

- **Weight Decay**: Penaliza pesos altos, reduzindo a sensibilidade do modelo.

---

## **[slides13.pdf](https://work.caltech.edu/slides/slides13.pdf)**
### Validação e Seleção de Modelos
- **Conceito**: Dividir os dados em conjuntos de treino (\(D_{\text{train}}\)) e validação (\(D_{\text{val}}\)).
- **Erro de Validação**:

$$
E_{\text{val}}(h) = \frac{1}{K} \sum_{k=1}^K e(h(x_k), y_k)
$$

- **Validação Cruzada**:
  - Divide os dados em \(K\)-folds e alterna entre treino e validação.
  - Erro médio:

$$
E_{\text{cv}} = \frac{1}{K} \sum_{n=1}^N e(h_{-n}(x_n), y_n)
$$

---

## **[slides14.pdf](https://work.caltech.edu/slides/slides14.pdf)**
### Máquinas de Vetores de Suporte (SVM)
- **Maximização da Margem**:

$$
\text{Distância} = \frac{1}{\|w\|}
$$

- **Problema de otimização**:

$$
\min \frac{1}{2} w^T w \quad \text{sujeito a } y_n (w^T x_n + b) \geq 1, \, \forall n
$$

- **Transformações Não Lineares**:
  - Mapeiam os dados para um espaço de características \(Z\), onde se tornam linearmente separáveis.

---

## **[slides15.pdf](https://work.caltech.edu/slides/slides15.pdf)**
### Métodos de Kernel
- **Truque do Kernel**:
  - Substitui o produto interno no espaço \(Z\) pelo kernel \(K(x, x')\).
- **Exemplo de Kernel Polinomial**:

$$
K(x, x') = (1 + x^T x')^Q
$$

- **SVM com Kernel**:

$$
g(x) = \text{sign} \left( \sum_{n=1}^N \alpha_n y_n K(x_n, x) + b \right)
$$

---

## **[slides16.pdf](https://work.caltech.edu/slides/slides16.pdf)**
### Funções de Base Radial (RBF)
- **Modelo básico**:

$$
h(x) = \sum_{n=1}^N w_n \exp(-\gamma \|x - x_n\|^2)
$$

- **Aprendizado**:
  - Determinar os pesos \(w_n\) resolvendo:

$$
\Phi w = y, \quad \Phi_{ij} = \exp(-\gamma \|x_i - x_j\|^2)
$$

- **Redução de complexidade**:
  - Usar \(K \ll N\) centros (\(\mu_k\)) em vez de todos os pontos de dados.

---

## **[slides17.pdf](https://work.caltech.edu/slides/slides17.pdf)**
### Três Princípios de Aprendizado
1. **Navalha de Occam**: O modelo mais simples que explica os dados é o mais plausível.
2. **Viés de Amostragem**:
   - Diferenças entre distribuição de treino (\(P_{\text{train}}(x)\)) e teste (\(P_{\text{test}}(x)\)).
3. **Data Snooping**:
   - O uso repetido dos mesmos dados pode introduzir viés e comprometer a avaliação.

---

## **[slides18.pdf](https://work.caltech.edu/slides/slides18.pdf)**
### Epílogo: O Mapa do Aprendizado de Máquina
- **Técnicas e Paradigmas**:
  - Supervisionado, não supervisionado, aprendizado por reforço.
  - Métodos como redes neurais, SVM, RBF, entre outros.
- **Aprendizado Bayesiano**:
  - Integra informações de distribuições a priori para melhorar o aprendizado.
- **Métodos de Agregação**:
  - Combina vários modelos (\(h_1, h_2, \ldots, h_T\)):

$$
\text{Regressão: } \bar{h}(x) = \frac{1}{T} \sum_{t=1}^T h_t(x)
$$

$$
\text{Classificação: } \bar{h}(x) = \text{majority vote}
$$


---







## ✨ Créditos
Resumos baseados nas aulas de Yaser Abu-Mostafa no curso de Aprendizado de Máquina da Caltech. Disponível em [Learning From Data](https://work.caltech.edu/telecourse.html).
