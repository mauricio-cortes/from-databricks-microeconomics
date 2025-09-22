"""
Utility Functions Module

This module provides classes and functions for utility analysis,
including various utility functions, indifference curves, and consumer choice theory.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Union, Tuple, List, Callable


class UtilityFunction:
    """
    Represents a utility function for consumer choice analysis.
    """
    
    def __init__(self, function_type: str = 'cobb_douglas', **parameters):
        """
        Initialize a utility function.
        
        Parameters:
        -----------
        function_type : str
            Type of utility function ('cobb_douglas', 'perfect_substitutes', 'perfect_complements', 'linear')
        **parameters : dict
            Parameters specific to the function type
        """
        self.function_type = function_type
        self.parameters = parameters
        
    def utility(self, x1: Union[float, np.ndarray], 
                x2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate utility for given quantities of two goods.
        
        Parameters:
        -----------
        x1 : float or array-like
            Quantity of good 1
        x2 : float or array-like  
            Quantity of good 2
            
        Returns:
        --------
        float or array-like
            Utility value(s)
        """
        if self.function_type == 'cobb_douglas':
            # U = x1^a * x2^b
            a = self.parameters.get('alpha', 0.5)
            b = self.parameters.get('beta', 0.5)
            return np.power(x1, a) * np.power(x2, b)
        
        elif self.function_type == 'perfect_substitutes':
            # U = a*x1 + b*x2
            a = self.parameters.get('alpha', 1)
            b = self.parameters.get('beta', 1)
            return a * x1 + b * x2
        
        elif self.function_type == 'perfect_complements':
            # U = min(a*x1, b*x2)
            a = self.parameters.get('alpha', 1)
            b = self.parameters.get('beta', 1)
            return np.minimum(a * x1, b * x2)
        
        elif self.function_type == 'linear':
            # U = a*x1 + b*x2 (same as perfect substitutes)
            return self.utility_perfect_substitutes(x1, x2)
        
        else:
            raise ValueError(f"Unknown function type: {self.function_type}")
    
    def marginal_utility_x1(self, x1: Union[float, np.ndarray], 
                           x2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate marginal utility with respect to good 1.
        """
        if self.function_type == 'cobb_douglas':
            a = self.parameters.get('alpha', 0.5)
            b = self.parameters.get('beta', 0.5)
            return a * np.power(x1, a-1) * np.power(x2, b)
        
        elif self.function_type == 'perfect_substitutes':
            return self.parameters.get('alpha', 1)
        
        elif self.function_type == 'perfect_complements':
            a = self.parameters.get('alpha', 1)
            b = self.parameters.get('beta', 1)
            # MU is infinite when a*x1 < b*x2, zero when a*x1 > b*x2
            return np.where(np.abs(a*x1 - b*x2) < 1e-10, np.inf, 0)
        
        else:
            # Numerical approximation
            h = 1e-8
            return (self.utility(x1 + h, x2) - self.utility(x1, x2)) / h
    
    def marginal_utility_x2(self, x1: Union[float, np.ndarray], 
                           x2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate marginal utility with respect to good 2.
        """
        if self.function_type == 'cobb_douglas':
            a = self.parameters.get('alpha', 0.5)
            b = self.parameters.get('beta', 0.5)
            return b * np.power(x1, a) * np.power(x2, b-1)
        
        elif self.function_type == 'perfect_substitutes':
            return self.parameters.get('beta', 1)
        
        elif self.function_type == 'perfect_complements':
            a = self.parameters.get('alpha', 1)
            b = self.parameters.get('beta', 1)
            return np.where(np.abs(a*x1 - b*x2) < 1e-10, np.inf, 0)
        
        else:
            # Numerical approximation
            h = 1e-8
            return (self.utility(x1, x2 + h) - self.utility(x1, x2)) / h
    
    def mrs(self, x1: Union[float, np.ndarray], 
            x2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate marginal rate of substitution (MRS = MU1/MU2).
        """
        mu1 = self.marginal_utility_x1(x1, x2)
        mu2 = self.marginal_utility_x2(x1, x2)
        
        # Avoid division by zero
        return np.where(mu2 != 0, mu1 / mu2, np.inf)
    
    def indifference_curve(self, utility_level: float, 
                          x1_range: Tuple[float, float] = (0.1, 10),
                          points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate points on an indifference curve for a given utility level.
        
        Parameters:
        -----------
        utility_level : float
            The utility level for the indifference curve
        x1_range : tuple
            Range of x1 values to plot
        points : int
            Number of points to generate
            
        Returns:
        --------
        tuple
            (x1_values, x2_values) for the indifference curve
        """
        x1_values = np.linspace(x1_range[0], x1_range[1], points)
        x2_values = np.zeros_like(x1_values)
        
        if self.function_type == 'cobb_douglas':
            a = self.parameters.get('alpha', 0.5)
            b = self.parameters.get('beta', 0.5)
            # From U = x1^a * x2^b, solve for x2: x2 = (U/x1^a)^(1/b)
            x2_values = np.power(utility_level / np.power(x1_values, a), 1/b)
        
        elif self.function_type == 'perfect_substitutes':
            a = self.parameters.get('alpha', 1)
            b = self.parameters.get('beta', 1)
            # From U = a*x1 + b*x2, solve for x2: x2 = (U - a*x1)/b
            x2_values = (utility_level - a * x1_values) / b
        
        elif self.function_type == 'perfect_complements':
            a = self.parameters.get('alpha', 1)
            b = self.parameters.get('beta', 1)
            # U = min(a*x1, b*x2), so at equilibrium: a*x1 = b*x2 = U
            x1_optimal = utility_level / a
            x2_optimal = utility_level / b
            # L-shaped indifference curve
            x1_values = np.array([0, x1_optimal, x1_optimal, x1_range[1]])
            x2_values = np.array([x2_optimal, x2_optimal, 0, 0])
        
        # Filter out negative values
        valid_mask = (x1_values >= 0) & (x2_values >= 0)
        return x1_values[valid_mask], x2_values[valid_mask]
    
    def plot_indifference_curves(self, utility_levels: List[float],
                                x1_range: Tuple[float, float] = (0.1, 10),
                                title: str = "Indifference Curves") -> go.Figure:
        """
        Plot multiple indifference curves.
        """
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, u_level in enumerate(utility_levels):
            x1_vals, x2_vals = self.indifference_curve(u_level, x1_range)
            
            fig.add_trace(go.Scatter(
                x=x1_vals, y=x2_vals,
                mode='lines',
                name=f'U = {u_level}',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Good 1 (x₁)',
            yaxis_title='Good 2 (x₂)',
            template='plotly_white'
        )
        
        return fig


def cobb_douglas_utility(alpha: float = 0.5, beta: float = 0.5) -> UtilityFunction:
    """
    Create a Cobb-Douglas utility function: U = x1^α * x2^β
    
    Parameters:
    -----------
    alpha : float
        Exponent for good 1
    beta : float
        Exponent for good 2
        
    Returns:
    --------
    UtilityFunction
        Cobb-Douglas utility function
    """
    return UtilityFunction('cobb_douglas', alpha=alpha, beta=beta)


def perfect_substitutes_utility(alpha: float = 1, beta: float = 1) -> UtilityFunction:
    """
    Create a perfect substitutes utility function: U = α*x1 + β*x2
    """
    return UtilityFunction('perfect_substitutes', alpha=alpha, beta=beta)


def perfect_complements_utility(alpha: float = 1, beta: float = 1) -> UtilityFunction:
    """
    Create a perfect complements utility function: U = min(α*x1, β*x2)
    """
    return UtilityFunction('perfect_complements', alpha=alpha, beta=beta)


def optimal_consumption(utility_function: UtilityFunction,
                       income: float,
                       price1: float,
                       price2: float) -> Tuple[float, float]:
    """
    Find optimal consumption bundle given budget constraint.
    
    Parameters:
    -----------
    utility_function : UtilityFunction
        The consumer's utility function
    income : float
        Consumer's income/budget
    price1 : float
        Price of good 1
    price2 : float
        Price of good 2
        
    Returns:
    --------
    tuple
        (optimal_x1, optimal_x2)
    """
    if utility_function.function_type == 'cobb_douglas':
        # For Cobb-Douglas: x1* = (α/(α+β)) * I/p1, x2* = (β/(α+β)) * I/p2
        alpha = utility_function.parameters.get('alpha', 0.5)
        beta = utility_function.parameters.get('beta', 0.5)
        
        x1_optimal = (alpha / (alpha + beta)) * income / price1
        x2_optimal = (beta / (alpha + beta)) * income / price2
        
        return x1_optimal, x2_optimal
    
    elif utility_function.function_type == 'perfect_substitutes':
        # Consumer spends all income on the cheaper good (per unit of utility)
        alpha = utility_function.parameters.get('alpha', 1)
        beta = utility_function.parameters.get('beta', 1)
        
        if price1/alpha < price2/beta:
            # Good 1 is relatively cheaper
            return income/price1, 0
        elif price1/alpha > price2/beta:
            # Good 2 is relatively cheaper
            return 0, income/price2
        else:
            # Indifferent - any combination on budget line
            return income/(2*price1), income/(2*price2)
    
    elif utility_function.function_type == 'perfect_complements':
        # Optimal consumption at the corner: a*x1 = b*x2
        alpha = utility_function.parameters.get('alpha', 1)
        beta = utility_function.parameters.get('beta', 1)
        
        # Budget constraint: p1*x1 + p2*x2 = I
        # Substituting x2 = (alpha/beta)*x1 into budget constraint
        x1_optimal = income / (price1 + price2 * alpha/beta)
        x2_optimal = (alpha/beta) * x1_optimal
        
        return x1_optimal, x2_optimal
    
    else:
        raise ValueError(f"Optimization not implemented for {utility_function.function_type}")