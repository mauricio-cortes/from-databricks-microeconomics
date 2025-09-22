"""
Elasticity Module

This module provides functions for calculating various types of elasticity
including price elasticity of demand, income elasticity, and cross-price elasticity.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple


def price_elasticity(prices: Union[List, np.ndarray], 
                    quantities: Union[List, np.ndarray],
                    method: str = 'midpoint') -> Union[float, np.ndarray]:
    """
    Calculate price elasticity of demand.
    
    Parameters:
    -----------
    prices : array-like
        Array of prices
    quantities : array-like
        Array of corresponding quantities
    method : str
        Method for calculation ('midpoint', 'point', 'arc')
        
    Returns:
    --------
    float or array
        Price elasticity values
    """
    prices = np.array(prices)
    quantities = np.array(quantities)
    
    if len(prices) != len(quantities):
        raise ValueError("Prices and quantities must have the same length")
    
    if len(prices) < 2:
        raise ValueError("Need at least 2 data points to calculate elasticity")
    
    if method == 'midpoint':
        # Midpoint method: Ed = ((Q2-Q1)/((Q2+Q1)/2)) / ((P2-P1)/((P2+P1)/2))
        price_changes = np.diff(prices)
        quantity_changes = np.diff(quantities)
        price_midpoints = (prices[1:] + prices[:-1]) / 2
        quantity_midpoints = (quantities[1:] + quantities[:-1]) / 2
        
        # Avoid division by zero
        nonzero_mask = (price_midpoints != 0) & (quantity_midpoints != 0)
        elasticity = np.zeros_like(price_changes)
        
        elasticity[nonzero_mask] = (
            (quantity_changes[nonzero_mask] / quantity_midpoints[nonzero_mask]) /
            (price_changes[nonzero_mask] / price_midpoints[nonzero_mask])
        )
        
        return elasticity
    
    elif method == 'point':
        # Point elasticity: Ed = (dQ/dP) * (P/Q)
        if len(prices) == 2:
            dQ_dP = (quantities[1] - quantities[0]) / (prices[1] - prices[0])
            avg_price = np.mean(prices)
            avg_quantity = np.mean(quantities)
            return dQ_dP * (avg_price / avg_quantity) if avg_quantity != 0 else 0
        else:
            # Use numerical differentiation for multiple points
            dQ_dP = np.gradient(quantities, prices)
            return dQ_dP * (prices / quantities)
    
    elif method == 'arc':
        # Arc elasticity: Ed = ((Q2-Q1)/(Q2+Q1)) / ((P2-P1)/(P2+P1))
        if len(prices) != 2 or len(quantities) != 2:
            raise ValueError("Arc method requires exactly 2 data points")
        
        price_change = prices[1] - prices[0]
        quantity_change = quantities[1] - quantities[0]
        price_sum = prices[1] + prices[0]
        quantity_sum = quantities[1] + quantities[0]
        
        if price_sum == 0 or quantity_sum == 0:
            return 0
        
        return (quantity_change / quantity_sum) / (price_change / price_sum)


def income_elasticity(incomes: Union[List, np.ndarray],
                     quantities: Union[List, np.ndarray],
                     method: str = 'midpoint') -> Union[float, np.ndarray]:
    """
    Calculate income elasticity of demand.
    
    Parameters:
    -----------
    incomes : array-like
        Array of income levels
    quantities : array-like
        Array of corresponding quantities demanded
    method : str
        Method for calculation ('midpoint', 'point', 'arc')
        
    Returns:
    --------
    float or array
        Income elasticity values
    """
    # Income elasticity uses the same calculation as price elasticity
    # but with income instead of price
    return price_elasticity(incomes, quantities, method)


def cross_price_elasticity(price_x: Union[List, np.ndarray],
                          quantity_y: Union[List, np.ndarray],
                          method: str = 'midpoint') -> Union[float, np.ndarray]:
    """
    Calculate cross-price elasticity of demand.
    
    Parameters:
    -----------
    price_x : array-like
        Array of prices for good X
    quantity_y : array-like
        Array of quantities demanded for good Y
    method : str
        Method for calculation ('midpoint', 'point', 'arc')
        
    Returns:
    --------
    float or array
        Cross-price elasticity values
    """
    return price_elasticity(price_x, quantity_y, method)


def classify_elasticity(elasticity_value: float, elasticity_type: str = 'price') -> str:
    """
    Classify elasticity based on its magnitude and type.
    
    Parameters:
    -----------
    elasticity_value : float
        The elasticity value to classify
    elasticity_type : str
        Type of elasticity ('price', 'income', 'cross_price')
        
    Returns:
    --------
    str
        Classification of the elasticity
    """
    abs_elasticity = abs(elasticity_value)
    
    if elasticity_type == 'price':
        if abs_elasticity < 1:
            return "Inelastic"
        elif abs_elasticity == 1:
            return "Unit Elastic"
        else:
            return "Elastic"
    
    elif elasticity_type == 'income':
        if elasticity_value < 0:
            return "Inferior Good"
        elif 0 < elasticity_value < 1:
            return "Normal Good (Necessity)"
        elif elasticity_value > 1:
            return "Normal Good (Luxury)"
        else:
            return "Neutral"
    
    elif elasticity_type == 'cross_price':
        if elasticity_value > 0:
            return "Substitute Goods"
        elif elasticity_value < 0:
            return "Complementary Goods"
        else:
            return "Independent Goods"
    
    return "Unknown"


def elasticity_summary(data: pd.DataFrame,
                      price_col: str = 'price',
                      quantity_col: str = 'quantity',
                      income_col: str = None) -> pd.DataFrame:
    """
    Create a summary of various elasticity measures for a dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing market data
    price_col : str
        Name of the price column
    quantity_col : str
        Name of the quantity column
    income_col : str
        Name of the income column (optional)
        
    Returns:
    --------
    pd.DataFrame
        Summary of elasticity measures
    """
    summary = []
    
    # Price elasticity
    if len(data) >= 2:
        price_elast = price_elasticity(
            data[price_col].values,
            data[quantity_col].values,
            method='midpoint'
        )
        
        avg_price_elast = np.mean(price_elast)
        summary.append({
            'Elasticity Type': 'Price Elasticity',
            'Value': avg_price_elast,
            'Classification': classify_elasticity(avg_price_elast, 'price')
        })
        
        # Income elasticity if income data available
        if income_col and income_col in data.columns:
            income_elast = income_elasticity(
                data[income_col].values,
                data[quantity_col].values,
                method='midpoint'
            )
            
            avg_income_elast = np.mean(income_elast)
            summary.append({
                'Elasticity Type': 'Income Elasticity',
                'Value': avg_income_elast,
                'Classification': classify_elasticity(avg_income_elast, 'income')
            })
    
    return pd.DataFrame(summary)