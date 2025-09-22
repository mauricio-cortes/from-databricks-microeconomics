# Microeconomia com Python no Databricks

Uma biblioteca abrangente de microeconomia em Python, projetada especificamente para funcionar perfeitamente em ambientes Databricks. Esta biblioteca fornece ferramentas para análise econômica, modelagem e visualização de conceitos microeconômicos fundamentais.

## 🚀 Características Principais

- **Análise de Demanda e Oferta**: Curvas de demanda e oferta com diferentes formas funcionais
- **Equilíbrio de Mercado**: Cálculo automático de pontos de equilíbrio e excedentes
- **Teoria do Consumidor**: Funções de utilidade, curvas de indiferença e otimização
- **Teoria da Produção**: Funções de produção, isoquantas e minimização de custos
- **Análise de Elasticidade**: Elasticidades de preço, renda e cruzada
- **Visualizações Interativas**: Gráficos com Plotly para análise dinâmica
- **Compatibilidade Databricks**: Otimizado para notebooks Databricks

## 📦 Instalação

### No Databricks

1. Clone este repositório em seu workspace do Databricks
2. Instale as dependências executando em uma célula:

```python
%pip install numpy pandas matplotlib seaborn plotly scipy
```

3. Importe a biblioteca:

```python
import sys
sys.path.append('/Workspace/Repos/your-repo-name')

from microeconomics import *
```

### Instalação Local

```bash
git clone https://github.com/mauricio-cortes/from-databricks-microeconomics.git
cd from-databricks-microeconomics
pip install -r requirements.txt
```

## 🔧 Uso Rápido

### Análise de Demanda e Oferta

```python
from microeconomics import linear_demand, linear_supply, find_equilibrium, plot_market

# Criar curvas
demand = linear_demand(intercept=100, slope=2)
supply = linear_supply(intercept=-20, slope=3)

# Encontrar equilíbrio
eq_price, eq_quantity = find_equilibrium(demand, supply)
print(f"Preço: R$ {eq_price:.2f}, Quantidade: {eq_quantity:.2f}")

# Visualizar
fig = plot_market(demand, supply)
fig.show()
```

### Teoria do Consumidor

```python
from microeconomics.utility import cobb_douglas_utility, optimal_consumption

# Função de utilidade
utility = cobb_douglas_utility(alpha=0.6, beta=0.4)

# Otimização do consumo
income = 100
price1, price2 = 5, 4
x1_opt, x2_opt = optimal_consumption(utility, income, price1, price2)

# Plotar curvas de indiferença
fig = utility.plot_indifference_curves([1, 2, 3, 4, 5])
fig.show()
```

### Teoria da Produção

```python
from microeconomics.production import cobb_douglas_production, cost_minimization

# Função de produção
production = cobb_douglas_production(
    total_factor_productivity=2,
    labor_elasticity=0.7,
    capital_elasticity=0.3
)

# Minimização de custos
target_output = 20
wage, rental = 10, 15
L_opt, K_opt, min_cost = cost_minimization(production, target_output, wage, rental)

print(f"Trabalho: {L_opt:.2f}, Capital: {K_opt:.2f}, Custo: R$ {min_cost:.2f}")
```

## 📊 Notebooks de Exemplo

Este projeto inclui notebooks Jupyter demonstrando uso prático:

1. **[Análise de Demanda e Oferta](notebooks/01_analise_demanda_oferta.ipynb)**
   - Curvas de demanda e oferta
   - Equilíbrio de mercado
   - Análise de elasticidade
   - Choques de mercado

2. **[Teoria do Consumidor](notebooks/02_teoria_consumidor.ipynb)**
   - Funções de utilidade
   - Curvas de indiferença
   - Otimização do consumo
   - Análise de sensibilidade

3. **[Teoria da Produção](notebooks/03_teoria_producao.ipynb)**
   - Funções de produção
   - Isoquantas
   - Minimização de custos
   - Análise de eficiência

## 📚 Módulos da Biblioteca

### `microeconomics.demand`
- `DemandCurve`: Classe para curvas de demanda
- `linear_demand()`: Criar demanda linear
- `estimate_demand_from_data()`: Estimar parâmetros de dados

### `microeconomics.supply`
- `SupplyCurve`: Classe para curvas de oferta
- `linear_supply()`: Criar oferta linear
- `estimate_supply_from_data()`: Estimar parâmetros de dados

### `microeconomics.equilibrium`
- `find_equilibrium()`: Encontrar equilíbrio de mercado
- `market_analysis()`: Análise completa de mercado
- `plot_market()`: Visualizar oferta e demanda

### `microeconomics.elasticity`
- `price_elasticity()`: Elasticidade-preço da demanda
- `income_elasticity()`: Elasticidade-renda
- `cross_price_elasticity()`: Elasticidade cruzada

### `microeconomics.utility`
- `UtilityFunction`: Classe para funções de utilidade
- `cobb_douglas_utility()`: Utilidade Cobb-Douglas
- `optimal_consumption()`: Otimização do consumidor

### `microeconomics.production`
- `ProductionFunction`: Classe para funções de produção
- `cobb_douglas_production()`: Produção Cobb-Douglas
- `cost_minimization()`: Minimização de custos

## 🎯 Casos de Uso

### Pesquisa Acadêmica
- Análise empírica de mercados
- Modelagem econométrica
- Teste de hipóteses econômicas

### Educação
- Ensino de microeconomia
- Exercícios interativos
- Visualização de conceitos

### Análise de Negócios
- Análise de precificação
- Otimização de custos
- Estudos de elasticidade

### Política Pública
- Análise de impactos regulatórios
- Avaliação de bem-estar
- Estudos de competição

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**: Linguagem principal
- **NumPy**: Computação numérica
- **Pandas**: Manipulação de dados
- **Plotly**: Visualizações interativas
- **SciPy**: Otimização e métodos numéricos
- **Matplotlib/Seaborn**: Gráficos estáticos

## 🔄 Roadmap

- [ ] Implementar teoria de jogos
- [ ] Adicionar modelos de estruturas de mercado
- [ ] Expandir análises de bem-estar
- [ ] Incluir modelos de informação assimétrica
- [ ] Adicionar testes econométricos
- [ ] Melhorar documentação e tutoriais

## 🤝 Contribuição

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 📧 Contato

- **Autor**: Mauricio Cortes
- **Email**: mauricio.cortes@example.com
- **LinkedIn**: [mauricio-cortes](https://linkedin.com/in/mauricio-cortes)

## 🙏 Agradecimentos

- Comunidade Python científico
- Databricks pela plataforma
- Bibliotecas open source utilizadas

---

**Desenvolvido com ❤️ para a comunidade de economia e ciência de dados**