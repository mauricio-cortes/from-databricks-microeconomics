"""
Production Functions Module

This module provides classes and functions for production analysis,
including various production functions, cost functions, and profit maximization.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Union, Tuple, List, Dict


class ProductionFunction:
    """
    Represents a production function for firm analysis.
    """
    
    def __init__(self, function_type: str = 'cobb_douglas', **parameters):
        """
        Initialize a production function.
        
        Parameters:
        -----------
        function_type : str
            Type of production function ('cobb_douglas', 'linear', 'leontief', 'ces')
        **parameters : dict
            Parameters specific to the function type
        """
        self.function_type = function_type
        self.parameters = parameters
        
    def output(self, labor: Union[float, np.ndarray], 
               capital: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate output for given inputs of labor and capital.
        
        Parameters:
        -----------
        labor : float or array-like
            Labor input
        capital : float or array-like  
            Capital input
            
        Returns:
        --------
        float or array-like
            Output quantity
        """
        if self.function_type == 'cobb_douglas':
            # Q = A * L^α * K^β
            A = self.parameters.get('total_factor_productivity', 1)
            alpha = self.parameters.get('labor_elasticity', 0.7)
            beta = self.parameters.get('capital_elasticity', 0.3)
            return A * np.power(labor, alpha) * np.power(capital, beta)
        
        elif self.function_type == 'linear':
            # Q = a*L + b*K
            a = self.parameters.get('labor_productivity', 1)
            b = self.parameters.get('capital_productivity', 1)
            return a * labor + b * capital
        
        elif self.function_type == 'leontief':
            # Q = min(L/a, K/b)
            a = self.parameters.get('labor_requirement', 1)
            b = self.parameters.get('capital_requirement', 1)
            return np.minimum(labor / a, capital / b)
        
        elif self.function_type == 'ces':
            # Q = A * (α*L^ρ + β*K^ρ)^(1/ρ)
            A = self.parameters.get('total_factor_productivity', 1)
            alpha = self.parameters.get('labor_share', 0.5)
            beta = self.parameters.get('capital_share', 0.5)
            rho = self.parameters.get('substitution_parameter', 0)
            
            if abs(rho) < 1e-10:  # ρ → 0 gives Cobb-Douglas
                return A * np.power(labor, alpha) * np.power(capital, beta)
            else:
                return A * np.power(alpha * np.power(labor, rho) + beta * np.power(capital, rho), 1/rho)
        
        else:
            raise ValueError(f"Unknown function type: {self.function_type}")
    
    def marginal_product_labor(self, labor: Union[float, np.ndarray], 
                              capital: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate marginal product of labor (MPL).
        """
        if self.function_type == 'cobb_douglas':
            A = self.parameters.get('total_factor_productivity', 1)
            alpha = self.parameters.get('labor_elasticity', 0.7)
            beta = self.parameters.get('capital_elasticity', 0.3)
            return A * alpha * np.power(labor, alpha-1) * np.power(capital, beta)
        
        elif self.function_type == 'linear':
            return self.parameters.get('labor_productivity', 1)
        
        elif self.function_type == 'leontief':
            a = self.parameters.get('labor_requirement', 1)
            b = self.parameters.get('capital_requirement', 1)
            # MPL is 1/a when L/a < K/b, 0 otherwise
            return np.where(labor/a < capital/b, 1/a, 0)
        
        else:
            # Numerical approximation
            h = 1e-8
            return (self.output(labor + h, capital) - self.output(labor, capital)) / h
    
    def marginal_product_capital(self, labor: Union[float, np.ndarray], 
                                capital: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate marginal product of capital (MPK).
        """
        if self.function_type == 'cobb_douglas':
            A = self.parameters.get('total_factor_productivity', 1)
            alpha = self.parameters.get('labor_elasticity', 0.7)
            beta = self.parameters.get('capital_elasticity', 0.3)
            return A * beta * np.power(labor, alpha) * np.power(capital, beta-1)
        
        elif self.function_type == 'linear':
            return self.parameters.get('capital_productivity', 1)
        
        elif self.function_type == 'leontief':
            a = self.parameters.get('labor_requirement', 1)
            b = self.parameters.get('capital_requirement', 1)
            # MPK is 1/b when K/b < L/a, 0 otherwise
            return np.where(capital/b < labor/a, 1/b, 0)
        
        else:
            # Numerical approximation
            h = 1e-8
            return (self.output(labor, capital + h) - self.output(labor, capital)) / h
    
    def isoquant(self, output_level: float, 
                 labor_range: Tuple[float, float] = (0.1, 10),
                 points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate points on an isoquant for a given output level.
        
        Parameters:
        -----------
        output_level : float
            The output level for the isoquant
        labor_range : tuple
            Range of labor values to plot
        points : int
            Number of points to generate
            
        Returns:
        --------
        tuple
            (labor_values, capital_values) for the isoquant
        """
        labor_values = np.linspace(labor_range[0], labor_range[1], points)
        capital_values = np.zeros_like(labor_values)
        
        if self.function_type == 'cobb_douglas':
            A = self.parameters.get('total_factor_productivity', 1)
            alpha = self.parameters.get('labor_elasticity', 0.7)
            beta = self.parameters.get('capital_elasticity', 0.3)
            # From Q = A * L^α * K^β, solve for K: K = (Q/(A*L^α))^(1/β)
            capital_values = np.power(output_level / (A * np.power(labor_values, alpha)), 1/beta)
        
        elif self.function_type == 'linear':
            a = self.parameters.get('labor_productivity', 1)
            b = self.parameters.get('capital_productivity', 1)
            # From Q = a*L + b*K, solve for K: K = (Q - a*L)/b
            capital_values = (output_level - a * labor_values) / b
        
        elif self.function_type == 'leontief':
            a = self.parameters.get('labor_requirement', 1)
            b = self.parameters.get('capital_requirement', 1)
            # Q = min(L/a, K/b), so at optimal: L/a = K/b = Q
            labor_optimal = output_level * a
            capital_optimal = output_level * b
            # L-shaped isoquant
            labor_values = np.array([labor_optimal, labor_optimal, labor_range[1]])
            capital_values = np.array([labor_range[1]*b, capital_optimal, capital_optimal])
        
        # Filter out negative values
        valid_mask = (labor_values >= 0) & (capital_values >= 0)
        return labor_values[valid_mask], capital_values[valid_mask]
    
    def plot_isoquants(self, output_levels: List[float],
                       labor_range: Tuple[float, float] = (0.1, 10),
                       title: str = "Production Isoquants") -> go.Figure:
        """
        Plot multiple isoquants.
        """
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, q_level in enumerate(output_levels):
            labor_vals, capital_vals = self.isoquant(q_level, labor_range)
            
            fig.add_trace(go.Scatter(
                x=labor_vals, y=capital_vals,
                mode='lines',
                name=f'Q = {q_level}',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Labor (L)',
            yaxis_title='Capital (K)',
            template='plotly_white'
        )
        
        return fig


def cobb_douglas_production(total_factor_productivity: float = 1,
                           labor_elasticity: float = 0.7,
                           capital_elasticity: float = 0.3) -> ProductionFunction:
    """
    Create a Cobb-Douglas production function: Q = A * L^α * K^β
    
    Parameters:
    -----------
    total_factor_productivity : float
        Total factor productivity (A)
    labor_elasticity : float
        Labor elasticity (α)
    capital_elasticity : float
        Capital elasticity (β)
        
    Returns:
    --------
    ProductionFunction
        Cobb-Douglas production function
    """
    return ProductionFunction('cobb_douglas', 
                            total_factor_productivity=total_factor_productivity,
                            labor_elasticity=labor_elasticity,
                            capital_elasticity=capital_elasticity)


def cost_minimization(production_function: ProductionFunction,
                     output_target: float,
                     wage_rate: float,
                     rental_rate: float) -> Tuple[float, float, float]:
    """
    Find cost-minimizing input combination for a given output level.
    
    Parameters:
    -----------
    production_function : ProductionFunction
        The firm's production function
    output_target : float
        Target output level
    wage_rate : float
        Wage rate (price of labor)
    rental_rate : float
        Rental rate (price of capital)
        
    Returns:
    --------
    tuple
        (optimal_labor, optimal_capital, minimum_cost)
    """
    if production_function.function_type == 'cobb_douglas':
        # For Cobb-Douglas cost minimization:
        # L* = Q * (α/β * r/w)^(β/(α+β)) / A^(1/(α+β))
        # K* = Q * (β/α * w/r)^(α/(α+β)) / A^(1/(α+β))
        
        A = production_function.parameters.get('total_factor_productivity', 1)
        alpha = production_function.parameters.get('labor_elasticity', 0.7)
        beta = production_function.parameters.get('capital_elasticity', 0.3)
        
        # Calculate optimal inputs
        factor1 = np.power(output_target / A, 1/(alpha + beta))
        
        labor_optimal = factor1 * np.power(alpha/beta * rental_rate/wage_rate, beta/(alpha + beta))
        capital_optimal = factor1 * np.power(beta/alpha * wage_rate/rental_rate, alpha/(alpha + beta))
        
        # Calculate minimum cost
        min_cost = wage_rate * labor_optimal + rental_rate * capital_optimal
        
        return labor_optimal, capital_optimal, min_cost
    
    elif production_function.function_type == 'leontief':
        # Fixed proportions: L/a = K/b = Q
        a = production_function.parameters.get('labor_requirement', 1)
        b = production_function.parameters.get('capital_requirement', 1)
        
        labor_optimal = output_target * a
        capital_optimal = output_target * b
        min_cost = wage_rate * labor_optimal + rental_rate * capital_optimal
        
        return labor_optimal, capital_optimal, min_cost
    
    elif production_function.function_type == 'linear':
        # Perfect substitutes: choose the cheaper input
        a = production_function.parameters.get('labor_productivity', 1)
        b = production_function.parameters.get('capital_productivity', 1)
        
        if wage_rate/a < rental_rate/b:
            # Labor is relatively cheaper
            labor_optimal = output_target / a
            capital_optimal = 0
        else:
            # Capital is relatively cheaper
            labor_optimal = 0
            capital_optimal = output_target / b
        
        min_cost = wage_rate * labor_optimal + rental_rate * capital_optimal
        
        return labor_optimal, capital_optimal, min_cost
    
    else:
        raise ValueError(f"Cost minimization not implemented for {production_function.function_type}")


def returns_to_scale(production_function: ProductionFunction,
                    labor: float = 1, capital: float = 1) -> str:
    """
    Determine returns to scale for a production function.
    
    Parameters:
    -----------
    production_function : ProductionFunction
        The production function to analyze
    labor : float
        Base labor input
    capital : float
        Base capital input
        
    Returns:
    --------
    str
        Returns to scale classification
    """
    if production_function.function_type == 'cobb_douglas':
        alpha = production_function.parameters.get('labor_elasticity', 0.7)
        beta = production_function.parameters.get('capital_elasticity', 0.3)
        
        sum_elasticities = alpha + beta
        
        if abs(sum_elasticities - 1) < 1e-10:
            return "Constant Returns to Scale"
        elif sum_elasticities > 1:
            return "Increasing Returns to Scale"
        else:
            return "Decreasing Returns to Scale"
    
    else:
        # Numerical check: compare f(2L, 2K) with 2*f(L, K)
        base_output = production_function.output(labor, capital)
        scaled_output = production_function.output(2*labor, 2*capital)
        
        ratio = scaled_output / (2 * base_output)
        
        if abs(ratio - 1) < 1e-10:
            return "Constant Returns to Scale"
        elif ratio > 1:
            return "Increasing Returns to Scale"
        else:
            return "Decreasing Returns to Scale"