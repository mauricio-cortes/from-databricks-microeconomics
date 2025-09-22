"""
Market Equilibrium Module

This module provides functions for finding market equilibrium points,
analyzing market dynamics, and visualizing supply and demand interactions.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple, Dict, Optional
from scipy.optimize import fsolve
from .demand import DemandCurve
from .supply import SupplyCurve


def find_equilibrium(demand_curve: DemandCurve, 
                    supply_curve: SupplyCurve,
                    initial_guess: float = 10.0) -> Tuple[float, float]:
    """
    Find the market equilibrium point where supply equals demand.
    
    Parameters:
    -----------
    demand_curve : DemandCurve
        The demand curve object
    supply_curve : SupplyCurve
        The supply curve object  
    initial_guess : float
        Initial guess for the equilibrium price
        
    Returns:
    --------
    tuple
        (equilibrium_price, equilibrium_quantity)
    """
    
    def excess_demand(price):
        """Calculate excess demand at given price"""
        qd = demand_curve.quantity_demanded(price)
        qs = supply_curve.quantity_supplied(price)
        return qd - qs
    
    try:
        # Find the price where excess demand = 0
        eq_price = fsolve(excess_demand, initial_guess)[0]
        
        # Calculate equilibrium quantity
        eq_quantity = demand_curve.quantity_demanded(eq_price)
        
        # Ensure we have a valid equilibrium (positive price and quantity)
        if eq_price > 0 and eq_quantity > 0:
            return float(eq_price), float(eq_quantity)
        else:
            return None, None
            
    except:
        return None, None


def market_analysis(demand_curve: DemandCurve, 
                   supply_curve: SupplyCurve,
                   price_range: Tuple[float, float] = (0.1, 20)) -> Dict:
    """
    Perform comprehensive market analysis.
    
    Parameters:
    -----------
    demand_curve : DemandCurve
        The demand curve object
    supply_curve : SupplyCurve
        The supply curve object
    price_range : tuple
        Range of prices to analyze
        
    Returns:
    --------
    dict
        Dictionary containing market analysis results
    """
    # Find equilibrium
    eq_price, eq_quantity = find_equilibrium(demand_curve, supply_curve)
    
    # Generate price points for analysis
    prices = np.linspace(price_range[0], price_range[1], 100)
    demand_quantities = demand_curve.quantity_demanded(prices)
    supply_quantities = supply_curve.quantity_supplied(prices)
    
    # Calculate consumer and producer surplus if equilibrium exists
    consumer_surplus = None
    producer_surplus = None
    
    if eq_price is not None and eq_quantity is not None:
        # Consumer surplus (area under demand curve above equilibrium price)
        if demand_curve.function_type == 'linear':
            # For linear demand: CS = 0.5 * base * height
            max_price = demand_curve.parameters.get('intercept', 100) / demand_curve.parameters.get('slope', 1)
            consumer_surplus = 0.5 * eq_quantity * (max_price - eq_price)
        
        # Producer surplus (area above supply curve below equilibrium price)
        if supply_curve.function_type == 'linear':
            # For linear supply: PS = 0.5 * base * height
            min_price = -supply_curve.parameters.get('intercept', -10) / supply_curve.parameters.get('slope', 1)
            if min_price < eq_price:
                producer_surplus = 0.5 * eq_quantity * (eq_price - min_price)
    
    return {
        'equilibrium_price': eq_price,
        'equilibrium_quantity': eq_quantity,
        'consumer_surplus': consumer_surplus,
        'producer_surplus': producer_surplus,
        'total_surplus': (consumer_surplus or 0) + (producer_surplus or 0),
        'price_range': prices,
        'demand_quantities': demand_quantities,
        'supply_quantities': supply_quantities
    }


def plot_market(demand_curve: DemandCurve, 
                supply_curve: SupplyCurve,
                price_range: Tuple[float, float] = (0.1, 20),
                title: str = "Market Analysis") -> go.Figure:
    """
    Create an interactive plot showing supply, demand, and equilibrium.
    
    Parameters:
    -----------
    demand_curve : DemandCurve
        The demand curve object
    supply_curve : SupplyCurve
        The supply curve object
    price_range : tuple
        Range of prices to plot
    title : str
        Title for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive market plot
    """
    # Get market analysis
    analysis = market_analysis(demand_curve, supply_curve, price_range)
    
    # Create figure
    fig = go.Figure()
    
    # Plot demand curve
    demand_mask = analysis['demand_quantities'] >= 0
    fig.add_trace(go.Scatter(
        x=analysis['demand_quantities'][demand_mask],
        y=analysis['price_range'][demand_mask],
        mode='lines',
        name='Demand',
        line=dict(color='blue', width=2)
    ))
    
    # Plot supply curve
    supply_mask = analysis['supply_quantities'] >= 0
    fig.add_trace(go.Scatter(
        x=analysis['supply_quantities'][supply_mask],
        y=analysis['price_range'][supply_mask],
        mode='lines',
        name='Supply',
        line=dict(color='red', width=2)
    ))
    
    # Plot equilibrium point if it exists
    if analysis['equilibrium_price'] is not None:
        fig.add_trace(go.Scatter(
            x=[analysis['equilibrium_quantity']],
            y=[analysis['equilibrium_price']],
            mode='markers',
            name='Equilibrium',
            marker=dict(color='green', size=10, symbol='diamond')
        ))
        
        # Add equilibrium lines
        fig.add_hline(
            y=analysis['equilibrium_price'],
            line_dash="dash",
            line_color="gray",
            annotation_text=f"P* = {analysis['equilibrium_price']:.2f}"
        )
        
        fig.add_vline(
            x=analysis['equilibrium_quantity'],
            line_dash="dash", 
            line_color="gray",
            annotation_text=f"Q* = {analysis['equilibrium_quantity']:.2f}"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title='Quantity',
        yaxis_title='Price',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def calculate_deadweight_loss(original_analysis: Dict, 
                             new_analysis: Dict) -> float:
    """
    Calculate deadweight loss from market intervention.
    
    Parameters:
    -----------
    original_analysis : dict
        Market analysis before intervention
    new_analysis : dict
        Market analysis after intervention
        
    Returns:
    --------
    float
        Deadweight loss amount
    """
    original_surplus = original_analysis.get('total_surplus', 0)
    new_surplus = new_analysis.get('total_surplus', 0)
    
    return max(0, original_surplus - new_surplus)