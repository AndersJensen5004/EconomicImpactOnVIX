####################
# Anders Jensen    #
####################
import pandas as pd
import matplotlib.pyplot as plt

# Load
houst = pd.read_csv('HOUST.csv', parse_dates=['DATE'], index_col='DATE')
m2sl = pd.read_csv('M2SL.csv', parse_dates=['DATE'], index_col='DATE')
ppiaco = pd.read_csv('PPIACO.csv', parse_dates=['DATE'], index_col='DATE')
unrate = pd.read_csv('UNRATE.csv', parse_dates=['DATE'], index_col='DATE')
vix = pd.read_csv('VIX_HISTORY.csv', parse_dates=['DATE'], index_col='DATE')

# Rename
houst.rename(columns={houst.columns[0]: 'HOUST'}, inplace=True)
m2sl.rename(columns={m2sl.columns[0]: 'M2SL'}, inplace=True)
ppiaco.rename(columns={ppiaco.columns[0]: 'PPIACO'}, inplace=True)
unrate.rename(columns={unrate.columns[0]: 'UNRATE'}, inplace=True)
vix.rename(columns={'CLOSE': 'VIX'}, inplace=True)

# Filter
start_date = '1990-01-01'
houst = houst[houst.index >= start_date]
m2sl = m2sl[m2sl.index >= start_date]
ppiaco = ppiaco[ppiaco.index >= start_date]
unrate = unrate[unrate.index >= start_date]
vix = vix[vix.index >= start_date]

# Handle missing values
houst = houst.dropna()
m2sl = m2sl.dropna()
ppiaco = ppiaco.dropna()
unrate = unrate.dropna()
vix = vix[['VIX']].dropna()

# Resample the economic indicator data to monthly frequency
houst = houst.resample('M').last()
m2sl = m2sl.resample('M').last()
ppiaco = ppiaco.resample('M').last()
unrate = unrate.resample('M').last()

# Resample the VIX data to monthly frequency and forward-fill NaN values
vix = vix.resample('M').last().ffill()

# Align the datasets by date
data = houst.join([m2sl, ppiaco, unrate, vix], how='outer')

# Ensure all dates are the first of the month
data.index = data.index.to_period('M').to_timestamp()

# Save
data.to_csv('combined_data.csv')
print("\nCombined Data:")
print(data.head())

# Plot data
plt.figure(figsize=(14, 7))

plt.subplot(2, 2, 1)
plt.plot(data.index, data['HOUST'], label='Housing Starts YoY % Change')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(data.index, data['M2SL'], label='M2 Money Supply YoY % Change')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(data.index, data['PPIACO'], label='PPI YoY % Change')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(data.index, data['UNRATE'], label='Unemployment Rate')
plt.legend()

plt.tight_layout()
plt.show()

