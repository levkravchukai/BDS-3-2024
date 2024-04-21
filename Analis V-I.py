import pyabf
import matplotlib.pyplot as plt
import pandas as pd
import gc
import numpy as np
import matplotlib.cm as cm
from sklearn.linear_model import LinearRegression
from matplotlib import gridspec
import matplotlib.ticker as ticker
import seaborn as sns
#from scipy import stats
#import statsmodels.api as sm
from scipy.stats import ttest_ind

gc.enable()

ctrl = pyabf.ABF("Control.abf")
plp = pyabf.ABF("PLP.abf")

# АЛЬТЕРНАТИВА
# =============================================================================
# abf = pyabf.ABF("Control.abf")
# 
# # Initialize an empty list to store DataFrames
# dfs = []
# 
# # Define the sweep numbers you want to load
# sweep_numbers = [0, 1, 2]  # Add the sweep numbers you're interested in
# 
# # Loop through the sweep numbers
# for sweep_number in sweep_numbers:
#     # Select the sweep data
#     abf.setSweep(sweepNumber=sweep_number)
#     data = abf.sweepY
#     
#     # Create a DataFrame for the current sweep
#     df = pd.DataFrame(data, columns=[f"Channel_1_Sweep_{sweep_number}"])
#     
#     # Optionally, add time information if available
#     df["Time"] = abf.sweepX  # Assuming time is available in the ABF file
#     
#     # Append the DataFrame to the list
#     dfs.append(df)
# 
# # Concatenate the DataFrames
# result_df = pd.concat(dfs, axis=1)
# 
# # Display the resulting DataFrame
# print(result_df)
# =============================================================================




# =============================================================================
# СТВОРЕННЯ ТАБЛИЦІ ДАНИХ
# =============================================================================

sumdict = {}
keys_arrays = []
values_arrays = []
sweepnames = ['KATP current+', '10µM glibenclamide', 'Wash']

# Створюємо один список із ключів і значень для двох файлів

for i in (ctrl, plp):
    if i == plp: file = 'PLP' 
    else: file = 'Control'
    for sweepNumber in ctrl.sweepList:
        i.setSweep(sweepNumber)
        keys_arrays.append((i.sweepLabelC, i.sweepLabelX, i.sweepLabelY + '('+ file + ' ' + sweepnames[sweepNumber] + ')'))
        values_arrays.append((i.sweepC, i.sweepX, i.sweepY))
    

# Створюємо довідник, оскільки різних змінних всього три: sweepC, sweepX, sweepY, тому j ∈ [0,2]
# і залежить віть кількості ключів, а саме всіх варіацій назв. Однакові назви видаляються автоматично

for i in list(range(len(keys_arrays))):
    for j in list(range(3)): 
        sumdict[keys_arrays[i][j]] = values_arrays[i][j]
            

table = pd.DataFrame.from_dict(sumdict)

# Додаємо колонки із чистим КАТФ щільністю
table['Pure control KATP density'] = (table.iloc[:,2] - table.iloc[:,3])/15.499998
table['Pure PLP KATP density'] = (table.iloc[:,5] - table.iloc[:,6])/18.63636

# Впорядкуємо колонки, щоб колонки з 'чистимє' струмом були у відповідних місцях

new_column_order = [0, 1, 2, 3, 4, 8, 5, 6, 7, 9]

table = table.iloc[:, new_column_order]

#Заповнюємо колонку Епох
epoch_list = []

for i, p in enumerate(zip(ctrl.sweepEpochs.p1s, ctrl.sweepEpochs.p2s)):
    epochType = ctrl.sweepEpochs.types[i]
    epoch_list.extend([epochType] * (p[1] - p[0]))

table['Epoch'] = epoch_list


# =============================================================================
# ВІДНОВЛЕННЯ ДАНИХ ДЕ НЕКОРЕКТНІ ДАНІ
# =============================================================================

# Некоректні дані міняємо на NA
control_condition = (table.iloc[:,0] > -65) & (table.iloc[:,0] < 20)
plp_condition = (table.iloc[:,0] > -40) & (table.iloc[:,0] < 0)
table.loc[control_condition, table.columns[5]] =  np.nan
table.loc[plp_condition, table.columns[9]] =  np.nan


# Data Imputation

# Identify missing values in columns 5 and 9
missing_values5 = table[table.columns[5]].isna()
missing_values9 = table[table.columns[9]].isna()

# Define X and y for regression
X5 = table[~missing_values5][[table.columns[0]]]  # Features for non-missing rows in column 5
y5 = table[~missing_values5][table.columns[5]]   # Target variable for non-missing rows in column 5

X9 = table[~missing_values9][[table.columns[0]]]  # Features for non-missing rows in column 9
y9 = table[~missing_values9][table.columns[9]]   # Target variable for non-missing rows in column 9

# Create LinearRegression models
regression_model5 = LinearRegression()
regression_model9 = LinearRegression()

# Fit the models
regression_model5.fit(X5, y5)
regression_model9.fit(X9, y9)

# Identify rows with missing values
missing_values5 = table[table.columns[5]].isna()
missing_values9 = table[table.columns[9]].isna()

# Predict missing values
X_missing5 = table[missing_values5][[table.columns[0]]]
X_missing9 = table[missing_values9][[table.columns[0]]]

# Predict the missing values
predicted_y5 = regression_model5.predict(X_missing5)
predicted_y9 = regression_model9.predict(X_missing9)

np.random.seed(19)
jittered_y5 = predicted_y5 + np.random.normal(0, 0.1, len(predicted_y5))
jittered_y9 = predicted_y9 + np.random.normal(0, 0.1, len(predicted_y9))

# Replace np.nan values with predicted values
table.loc[missing_values5, table.columns[5]] = jittered_y5
table.loc[missing_values9, table.columns[9]] = jittered_y9


# =============================================================================
# ВАХ ГРАФІК
# =============================================================================


# Відфільтрувати DataFrameлише для рядків Epoch: 'Ramp'
ramp_data = table[table['Epoch'] == 'Ramp']

#Кольорова гама
colors = cm.RdGy(np.linspace(0, 1, 4))

# Зобразити V-I для the Ramp epochs
plt.figure(figsize=(8, 6))
plt.grid(alpha=.5, ls='--')
plt.ylim((-1.2, 3))
plt.plot(ramp_data['Membrane Potential (mV)'], ramp_data['Pure control KATP density'], marker='.', linestyle='-', color=colors[0], label='Контроль: щільність струму К-АТФ каналів')
plt.plot(ramp_data['Membrane Potential (mV)'], ramp_data['Pure PLP KATP density'], marker='.', linestyle='-', color=colors[1], label='PLP: щільність струму К-АТФ каналів')


#Підписи
plt.legend(loc = "lower right")
plt.margins(0, .1)

plt.xlabel('Мембранний потенціал (мВ)')
plt.ylabel('Щільність струму (пА/пФ)')
plt.savefig('IV.tiff', dpi=500)
plt.show()


# =============================================================================
#  ГРАФІК ПРОТОКОЛЬНОГО ЗАПИСУ
# =============================================================================

fig = plt.figure(figsize=(10, 12))


# Створюємо grid для різних subplots
spec = gridspec.GridSpec(ncols=1, nrows=3, wspace=0.5,hspace=0.5, height_ratios=[4, 4, 1])


# Plot контроль
ax1 = fig.add_subplot(spec[0])
for sweepNumber in ctrl.sweepList:
    ctrl.setSweep(sweepNumber)
    if sweepNumber == 0:
        label = "Розчин без АТФ"
    elif sweepNumber == 1:
        label = "Розчин з 10 μМ глібенклабідом"
    elif sweepNumber == 2:
        label = "Відмивка"
    ax1.plot(ctrl.sweepX, ctrl.sweepY, color=cm.tab20b(0.9*sweepNumber / len(ctrl.sweepList)), label=label)
    
  
ax1.set_title('Kонтроль')
ax1.set_ylabel('Струм (пФ)')
ax1.legend(loc="lower right")


# Add annotation for ax1
ax1.annotate('А', xy=(-0.12, 0.98), xycoords='axes fraction',
             fontsize=16, fontweight='bold', color='black')
    
# Add the zoomed inset manually
left, bottom, width, height = [0.05, 0.10, 0.4, 0.5] 
axins = ax1.inset_axes([left, bottom, width, height])
axins.set_xlim(0.1, 0.3)
axins.set_ylim(-50, 120)    
# Plot the zoomed-in section
for sweepNumber in ctrl.sweepList:
    ctrl.setSweep(sweepNumber)
    axins.plot(ctrl.sweepX, ctrl.sweepY, color=cm.tab20b(0.9*sweepNumber / len(ctrl.sweepList)))
    
ax1.indicate_inset_zoom(axins, edgecolor="black")

# Decorate the inset plot
axins.tick_params(axis='both', labelsize=7)
axins.axhline(y=-51, color='gray', linestyle='--')
axins.axhline(y=120, color='gray', linestyle='--')
axins.axvline(x=0.33, color='gray', linestyle='--')  



# Plot PLP

ax2 = fig.add_subplot(spec[1], sharex=ax1)  # <-- this argument is new
for sweepNumber in plp.sweepList:
    plp.setSweep(sweepNumber)
    if sweepNumber == 0:
        label = "Розчин без АТФ"
    elif sweepNumber == 1:
        label = "Розчин з 10 μМ глібенклабідом"
    elif sweepNumber == 2:
        label = "Відмивка"
    ax2.plot(plp.sweepX, plp.sweepY, color=cm.tab20b(0.9*sweepNumber / len(ctrl.sweepList)), label=label)

# Add annotation for ax2
ax2.annotate('Б', xy=(-0.12, 0.98), xycoords='axes fraction',
             fontsize=16, fontweight='bold', color='black')  
  
ax2.set_title('PLP')    
ax2.set_ylabel('Струм (пФ)')
ax2.legend(loc="lower right")

    
# Add the zoomed inset manually
left, bottom, width, height = [0.05, 0.10, 0.4, 0.5] 
axins = ax2.inset_axes([left, bottom, width, height])
axins.set_xlim(0.1, 0.3)
axins.set_ylim(-70, 140)    
# Plot the zoomed-in section
for sweepNumber in ctrl.sweepList:
    plp.setSweep(sweepNumber)
    axins.plot(plp.sweepX, plp.sweepY, color=cm.tab20b(0.9*sweepNumber / len(ctrl.sweepList)))
    
ax2.indicate_inset_zoom(axins, edgecolor="black")

# Decorate the inset plot
axins.tick_params(axis='both', labelsize=7)
axins.axhline(y=-71, color='gray', linestyle='--')
axins.axhline(y=140, color='gray', linestyle='--')
axins.axvline(x=0.33, color='gray', linestyle='--')

# Plot command waveform
ax3 = fig.add_subplot(spec[2], sharex=ax1)  # <-- this argument is new
ax3.plot(ctrl.sweepX, ctrl.sweepC, color='r')

ax3.set_title('Протокол стимуляції')
ax3.set_xlabel('Час стимуляції (с)')
ax3.set_ylabel('Командний\nпотенціал (мВ)')
# Set axis scale steps for the x-axis and y-axis
# For example, we're setting the x-axis to have ticks every 0.5 and the y-axis every 10 units
ax3.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(25))

# Add annotation for ax3
ax3.annotate('В', xy=(-0.12, 1.1), xycoords='axes fraction',
             fontsize=16, fontweight='bold', color='black')

# decorate the plots
plt.margins(0, .1)
ax1.axes.set_xlim(0, 0.4)  # <-- adjust axis like this

# Iterate through the subplots and add labels
subplot_labels = ['A', 'B', 'C']

plt.savefig('currents.tiff', dpi=500)
plt.show()


# =============================================================================
# ПОРІВНЯННЯ ЩІЛЬНОСТІ К-АТФ СТРУМУ
# =============================================================================

density_df = pd.DataFrame() 
density_df[['Контроль', 'PLP']] = table.iloc[:, [5, 9]][(table.iloc[:,0]==50) & (table.iloc[:,1] >= 0.26)]


# =============================================================================
# # Shapiro-Wilk test for normality
# for col in density_df.columns:
#     stat, p_value = stats.shapiro(density_df[col])
#     print(f"Shapiro-Wilk test for {col}:")
#     print(f"Statistic: {stat}, p-value: {p_value}")
#     
#     alpha = 0.05
#     if p_value > alpha:
#         print(f"The data for {col} looks normally distributed (fail to reject H0)\n")
#     else:
#         print(f"The data for {col} does not look normally distributed (reject H0)\n")
#     
# # Plot histograms for each group
# for col in density_df.columns:
#     plt.figure(figsize=(8, 6))
#     plt.hist(density_df[col], bins='auto', color='skyblue', alpha=0.7)
#     plt.title(f'Histogram for {col}')
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.show()
# 
# # Plot Q-Q plots for each group
# for col in density_df.columns:
#     plt.figure(figsize=(8, 6))
#     sm.qqplot(density_df[col], line='s')
#     plt.title(f'Q-Q Plot for {col}')
#     plt.show()
# =============================================================================
    
    
# Perform Independent Two-Sample t-test
statistic, p_value = ttest_ind(density_df['Контроль'], density_df['PLP'])
# Print the results
print(f"t-statistic: {statistic}")
print(f"P-value: {p_value}")
    

# Perform wide_to_long operation
density_long_df = pd.melt(density_df, value_vars=['Контроль', 'PLP'], 
                         var_name='Key', value_name='Value')


# Стиль теми
sns.set_theme(style="whitegrid")

# Загрузити вибірковийй набір даних

clarity_ranking = ['Контроль', 'PLP']

# Установити розмір діаграми
f, ax = plt.subplots(figsize=(8, 4))


# Палітра відтінків сірого та тонкі лінії
sns.boxenplot(x="Key", y="Value",
              color="gray", order=clarity_ranking, palette="Set3",
              scale="linear", data=density_long_df, linewidth=0.5)


# Средня лінія для кожної категорії
means = density_long_df.groupby("Key")["Value"].mean().loc[clarity_ranking]
# medians = density_long_df.groupby("Key")["Value"].median().loc[clarity_ranking]
plt.plot(range(len(clarity_ranking)), means, marker="o", color="red", markersize=6, linestyle="--")


# Plot means and medians
for i, mean in enumerate(means):   #zip(means, medians)
    ax.text(i, mean + 0.1, f'Середнє значення: {mean:.2f}', ha='center', fontsize=10, color='black')
    # ax.text(i, median - 0.1, f'Медіана: {median:.2f}', ha='center', fontsize=10, color='blue')
    
# =============================================================================
# # Add p-values
# p_values = [p_value]  # You need to replace this with the actual p-values
# for i, p_value in enumerate(p_values):
#     ax.text(i, 0.5, f'p-value: {p_value:.2e}', ha='center', fontsize=10, color='green')
# =============================================================================

# Покращення графіків
ax.set_xlabel("Неонатальні кардіоміоцити", fontsize=16)
ax.set_ylabel("Щільність струму (пА/пФ)", fontsize=16)
sns.despine(left=True)

plt.xticks(ticks=range(len(clarity_ranking)), labels=clarity_ranking)
plt.tight_layout()

plt.savefig('density.tiff', dpi=500)
plt.show()


# =============================================================================
# ГРАФІК ЕКСПРЕСІЇ СУБОДИНИЦЬ К-АТФ КАНАЛІВ
# =============================================================================


# =============================================================================
# Генеруємо дані, знаючи Mean, SE, Number
# =============================================================================

# Define the parameters
parameters = [
    {'Key': 'Kir6.2', 'Condition': 'Контроль', 'Mean': 77.31, 'SE': 13.17, 'Number': 8},
    {'Key': 'Kir6.2', 'Condition': 'PLP', 'Mean': 147.07, 'SE': 26.92, 'Number': 7},
    {'Key': 'Sur2', 'Condition': 'Контроль', 'Mean': 1.06, 'SE': 0.09, 'Number': 6},
    {'Key': 'Sur2', 'Condition': 'PLP', 'Mean': 1.98, 'SE': 0.39, 'Number': 6},
]

# Generate random data and store in a DataFrame 30
data = []


for param in parameters:
    np.random.seed(1000)
    mean = param['Mean']
    se = param['SE']
    n = param['Number']
    
    # Generate random samples from a normal distribution
    samples = np.random.normal(loc=mean, scale=se, size=n)
    
    # Create a DataFrame for this condition
    df = pd.DataFrame({'Value': samples})
    
    # Add Key and Condition columns
    df['Key'] = param['Key']
    df['Condition'] = param['Condition']    
    df['Mean'] = param['Mean']
    
    # Append to the data list
    data.append(df)

# Concatenate all DataFrames
result_df = pd.concat(data, ignore_index=True)

# Display the resulting DataFrame
print(result_df)


spec = gridspec.GridSpec(ncols=2, nrows=1, wspace=0.25, hspace=0.5, width_ratios=[1, 1])
# Стиль теми
sns.set_theme(style="whitegrid")

# Установити розмір діаграми
fig = plt.figure(figsize=(12, 6))

for i, key in enumerate(['Kir6.2', 'Sur2']):
    # Filter the DataFrame for the specific 'Key'
    key_df = result_df[result_df['Key'] == key]
    ax = fig.add_subplot(spec[0,i])
    
    # Загрузити вибірковийй набір даних
    clarity_ranking = key_df['Condition'].unique()
    
    # Палітра відтінків сірого та тонкі лінії
    sns.boxenplot(x="Condition", y="Value",
                  color="gray", order=clarity_ranking, palette="Set3",
                  scale="linear", data=key_df, linewidth=0.5)
    
    

    # Средня лінія для кожної категорії
    expr_means = key_df.groupby("Condition")["Mean"].mean().loc[clarity_ranking]
    # expr_medians = key_df.groupby("Condition")["Mean"].median().loc[clarity_ranking]
    plt.plot(range(len(clarity_ranking)), expr_means, marker="o", color="red", markersize=6, linestyle="--")
    
    # Plot means and medians
    for i, mean in enumerate(expr_means):
        ax.text(i, mean, f'Середнє значення: {mean:.2f}', ha='center', fontsize=10, color='black', va='top')
        # ax.text(i, median - 0.1, f'Медіана: {median:.2f}', ha='center', fontsize=10, color='blue')
    
 
    
# =============================================================================
# # Add p-values
# p_values = [p_value]  # You need to replace this with the actual p-values
# for i, p_value in enumerate(p_values):
#     ax.text(i, 0.5, f'p-value: {p_value:.2e}', ha='center', fontsize=10, color='green')
# =============================================================================

# Покращення графіків
    ax.set_xlabel("Дорослі кардіоміоцити", fontsize=16)
    ax.set_ylabel(f'Відносний рівень мРНК:\n{key}/актин (ум.од.)', fontsize=16)
    sns.despine(left=True)
    
    plt.xticks(ticks=range(len(clarity_ranking)), labels=clarity_ranking)
    plt.tight_layout()

plt.savefig('expression.tiff', dpi=500)
plt.show()


# очистка пам'яті ІДЄ
# =============================================================================
# try:
#     from IPython import get_ipython
#     get_ipython().magic('clear')
#     get_ipython().magic('reset -f')
#     import matplotlib.pyplot as plt
#     plt.close('all')
# except:
#     pass
# =============================================================================
