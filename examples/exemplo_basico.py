"""
Exemplo Simples - Microeconomia com Python

Este script demonstra o uso básico da biblioteca de microeconomia
para análise de mercado.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Adicionar o diretório do projeto ao PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microeconomics import (
    linear_demand, linear_supply, find_equilibrium, 
    market_analysis, plot_market
)

def main():
    print("=== ANÁLISE BÁSICA DE MERCADO ===")
    print()
    
    # 1. Criar curvas de demanda e oferta
    print("1. Criando curvas de demanda e oferta...")
    demand = linear_demand(intercept=100, slope=2)  # Qd = 100 - 2*P
    supply = linear_supply(intercept=-20, slope=3)  # Qs = -20 + 3*P
    
    print("   Demanda: Qd = 100 - 2*P")
    print("   Oferta: Qs = -20 + 3*P")
    print()
    
    # 2. Encontrar equilíbrio
    print("2. Calculando equilíbrio de mercado...")
    eq_price, eq_quantity = find_equilibrium(demand, supply)
    print(f"   Preço de Equilíbrio: R$ {eq_price:.2f}")
    print(f"   Quantidade de Equilíbrio: {eq_quantity:.2f} unidades")
    print()
    
    # 3. Análise completa
    print("3. Realizando análise completa...")
    analysis = market_analysis(demand, supply)
    print(f"   Excedente do Consumidor: R$ {analysis['consumer_surplus']:.2f}")
    print(f"   Excedente do Produtor: R$ {analysis['producer_surplus']:.2f}")
    print(f"   Excedente Total: R$ {analysis['total_surplus']:.2f}")
    print()
    
    # 4. Simular dados de mercado
    print("4. Simulando dados de mercado...")
    np.random.seed(42)
    prices = np.linspace(15, 35, 20)
    quantities = 100 - 2*prices + np.random.normal(0, 2, 20)
    quantities = np.maximum(0, quantities)  # Garantir quantidades não-negativas
    
    market_data = pd.DataFrame({
        'price': prices,
        'quantity': quantities
    })
    
    print("   Dados simulados (primeiras 5 observações):")
    print(market_data.head())
    print()
    
    # 5. Calcular elasticidade
    print("5. Calculando elasticidade-preço da demanda...")
    from microeconomics.elasticity import price_elasticity, classify_elasticity
    
    elasticity = price_elasticity(market_data['price'], market_data['quantity'])
    avg_elasticity = np.mean(elasticity)
    classification = classify_elasticity(avg_elasticity, 'price')
    
    print(f"   Elasticidade média: {avg_elasticity:.3f}")
    print(f"   Classificação: {classification}")
    print()
    
    # 6. Análise de choque de demanda
    print("6. Simulando choque positivo de demanda...")
    demand_new = linear_demand(intercept=120, slope=2)  # Demanda aumenta
    eq_price_new, eq_quantity_new = find_equilibrium(demand_new, supply)
    
    print(f"   Novo Preço: R$ {eq_price_new:.2f} (variação: +R$ {eq_price_new - eq_price:.2f})")
    print(f"   Nova Quantidade: {eq_quantity_new:.2f} (variação: +{eq_quantity_new - eq_quantity:.2f})")
    print()
    
    # 7. Criar visualização simples
    print("7. Criando visualização...")
    
    # Dados para plotar
    price_range = np.linspace(0, 50, 100)
    q_demand = demand.quantity_demanded(price_range)
    q_supply = supply.quantity_supplied(price_range)
    
    # Filtrar valores válidos
    valid_demand = q_demand >= 0
    valid_supply = q_supply >= 0
    
    plt.figure(figsize=(10, 6))
    
    # Plotar curvas
    plt.plot(q_demand[valid_demand], price_range[valid_demand], 
             'b-', linewidth=2, label='Demanda')
    plt.plot(q_supply[valid_supply], price_range[valid_supply], 
             'r-', linewidth=2, label='Oferta')
    
    # Ponto de equilíbrio
    plt.plot(eq_quantity, eq_price, 'go', markersize=8, label='Equilíbrio')
    
    # Linhas de equilíbrio
    plt.axhline(y=eq_price, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=eq_quantity, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Quantidade')
    plt.ylabel('Preço (R$)')
    plt.title('Análise de Mercado - Demanda vs Oferta')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 80)
    plt.ylim(0, 50)
    
    # Adicionar anotações
    plt.annotate(f'Equilíbrio\n(P={eq_price:.1f}, Q={eq_quantity:.1f})', 
                xy=(eq_quantity, eq_price), xytext=(eq_quantity+10, eq_price+5),
                arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.tight_layout()
    plt.show()
    
    print("   Gráfico exibido!")
    print()
    
    print("=== ANÁLISE CONCLUÍDA ===")
    print("Para análises mais avançadas, consulte os notebooks em /notebooks/")

if __name__ == "__main__":
    main()