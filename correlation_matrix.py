"""
Pearson correlation matrix of daily returns for the portfolio basket (heatmap).

Original file: LSE_003 correlation matrix.py
"""

import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

tickers = ["GSK.L", "SCT.L", "BAES.L", "HILS.L", "RKH.L", "NG.L", "SGRO.L", "INCH.L", "HSBA.L", "BT-A.L"]


data = yf.download(tickers, start='2023-01-01', end='2025-10-01')['Close']
print(data)

# Step 2: Calculate daily returns
returns = data.pct_change().dropna()

# Step 3: Compute correlation matrix
corr_matrix = returns.corr()
print("Correlation Matrix:")
#print(corr_matrix)

# Step 4: Visualize
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Asset Returns')
plt.show()


