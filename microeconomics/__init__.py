"""
Microeconomics Python Library for Databricks
=============================================

A comprehensive library for microeconomic analysis, modeling, and visualization
designed to work seamlessly with Databricks environments.

Modules:
    - demand: Demand curve analysis and modeling
    - supply: Supply curve analysis and modeling
    - equilibrium: Market equilibrium calculations
    - elasticity: Price and income elasticity calculations
    - utility: Utility functions and consumer theory
    - production: Production functions and cost analysis
"""

__version__ = "1.0.0"
__author__ = "Microeconomics Team"
__email__ = "support@microeconomics.com"

# Import main classes and functions
try:
    from .demand import DemandCurve, linear_demand
    from .supply import SupplyCurve, linear_supply
    from .equilibrium import find_equilibrium, market_analysis, plot_market
    from .elasticity import price_elasticity, income_elasticity
    from .utility import UtilityFunction, cobb_douglas_utility
    from .production import ProductionFunction, cobb_douglas_production
    
    __all__ = [
        'DemandCurve',
        'SupplyCurve', 
        'linear_demand',
        'linear_supply',
        'find_equilibrium',
        'market_analysis',
        'plot_market',
        'price_elasticity',
        'income_elasticity',
        'UtilityFunction',
        'ProductionFunction',
        'cobb_douglas_utility',
        'cobb_douglas_production'
    ]
    
except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")
    print("Make sure all required dependencies are installed:")
    print("pip install numpy pandas matplotlib seaborn plotly scipy")
    
    __all__ = []