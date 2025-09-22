"""
Supply Analysis Module

This module provides classes and functions for supply curve analysis,
including linear and non-linear supply functions, supply shifts, and
supply elasticity calculations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Union, Tuple, List, Optional


class SupplyCurve:
    """
    Represents a supply curve with various functional forms.
    """
    
    def __init__(self, function_type: str = 'linear', **parameters):
        """
        Initialize a supply curve.
        
        Parameters:
        -----------
        function_type : str
            Type of supply function ('linear', 'log', 'power')
        **parameters : dict
            Parameters specific to the function type
        """
        self.function_type = function_type
        self.parameters = parameters
        
    def quantity_supplied(self, price: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate quantity supplied at given price(s).
        
        Parameters:
        -----------
        price : float or array-like
            Price or array of prices
            
        Returns:
        --------
        float or array-like
            Quantity supplied
        """
        if self.function_type == 'linear':
            # Q = a + b*P
            a = self.parameters.get('intercept', -10)
            b = self.parameters.get('slope', 1)
            return np.maximum(0, a + b * price)
        
        elif self.function_type == 'log':
            # Q = a * ln(P) + b
            a = self.parameters.get('log_coeff', 10)
            b = self.parameters.get('intercept', -20)
            return np.maximum(0, a * np.log(price) + b)
            
        elif self.function_type == 'power':
            # Q = a * P^b
            a = self.parameters.get('coefficient', 10)
            b = self.parameters.get('elasticity', 1)
            return a * np.power(price, b)
            
        else:
            raise ValueError(f"Unknown function type: {self.function_type}")
    
    def plot(self, price_range: Tuple[float, float] = (0.1, 20), 
             points: int = 100, title: str = "Supply Curve") -> go.Figure:
        """
        Plot the supply curve using Plotly.
        
        Parameters:
        -----------
        price_range : tuple
            Range of prices to plot
        points : int
            Number of points to plot
        title : str
            Title for the plot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive plot of the supply curve
        """
        prices = np.linspace(price_range[0], price_range[1], points)
        quantities = self.quantity_supplied(prices)
        
        # Filter out negative quantities
        valid_mask = quantities >= 0
        prices = prices[valid_mask]
        quantities = quantities[valid_mask]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=quantities, y=prices,
            mode='lines',
            name='Supply Curve',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Quantity',
            yaxis_title='Price',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def shift(self, shift_amount: float, shift_type: str = 'parallel'):
        """
        Shift the supply curve.
        
        Parameters:
        -----------
        shift_amount : float
            Amount to shift the curve
        shift_type : str
            Type of shift ('parallel', 'rotation')
        """
        if self.function_type == 'linear':
            if shift_type == 'parallel':
                self.parameters['intercept'] += shift_amount
            elif shift_type == 'rotation':
                self.parameters['slope'] *= (1 + shift_amount)
        else:
            # For non-linear functions, adjust the main coefficient
            if 'coefficient' in self.parameters:
                self.parameters['coefficient'] *= (1 + shift_amount)
            elif 'intercept' in self.parameters:
                self.parameters['intercept'] += shift_amount


def linear_supply(intercept: float = -10, slope: float = 1) -> SupplyCurve:
    """
    Create a linear supply curve: Q = intercept + slope * P
    
    Parameters:
    -----------
    intercept : float
        Supply intercept (can be negative for minimum price threshold)
    slope : float
        Slope of the supply curve (should be positive)
        
    Returns:
    --------
    SupplyCurve
        Linear supply curve object
    """
    return SupplyCurve('linear', intercept=intercept, slope=slope)


def estimate_supply_from_data(data: pd.DataFrame, 
                             price_col: str = 'price',
                             quantity_col: str = 'quantity') -> SupplyCurve:
    """
    Estimate supply curve parameters from price and quantity data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price and quantity data
    price_col : str
        Name of the price column
    quantity_col : str
        Name of the quantity column
        
    Returns:
    --------
    SupplyCurve
        Estimated supply curve
    """
    from scipy.stats import linregress
    
    # Perform linear regression: quantity = intercept + slope * price
    slope, intercept, r_value, p_value, std_err = linregress(
        data[price_col], data[quantity_col]
    )
    
    return linear_supply(intercept=intercept, slope=slope)