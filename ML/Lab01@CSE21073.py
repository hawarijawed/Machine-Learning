import pandas as pd 
import numpy as np 
import statistics as st
import matplotlib.pyplot as plt 
# df = pd.DataFrame({'Yes':[23,43],'No':[43,65]})
# print((df))

df = pd.read_csv("/home/jawedhawari/Downloads/lab_data.csv")
# print(df.head())
print("Dimension of the data fram: ",df.shape)
# Removing unnamed columns and unwanted columns
unnamed_columns = [col for col in df.columns if col.startswith('Unnamed')]
df = df.drop(columns=unnamed_columns)

# Save the modified DataFrame to a new CSV file if needed
df.to_csv('new_file.csv', index=False)
# print(df.head())

cols_to_remove = ['Candy','Mango','Milk']
df = df.drop(columns=cols_to_remove)
df.to_csv('new_file.csv', index=False)
print(df.head())
# Loading the data into two matrices A and C
A_data = ['Candies (#)','Mangoes (Kg)','Milk Packets (#)']
C_data = ['Payment (Rs)']

A = df[A_data].values
C = df[C_data].values
# print("Matrix A")
# print(A)
# print("Matrix C:")
# print(C)

# Dimensionality of the vector space
print("Dimension of matrix A: ",A.shape)
print("Dimenstio of matrix C: ",C.shape)

# Rank of the matrix A
rank_A = np.linalg.matrix_rank(A)
print("Rank of the matrix A: ",rank_A)

# Pseudo inverse of A 
A_inv = np.linalg.pinv(A)
#print("Pseudo inverse of matrix A: \n",A_inv)

# A2. Use the Pseudo-inverse to calculate the model vector X for predicting the cost of the products
#     available with the vendor.

X = np.dot(A_inv,C)
print("Price of each items: \n",X)

# A3. Mark all customers (in “Purchase Data” table) with payments above Rs. 200 as RICH and others
#     as POOR. Develop a classifier model to categorize customers into RICH or POOR class based on
#     purchase behavior.
#  Add an extra feature called Purchase category for Rich and Poor classification
df['Purchase Category'] = df['Payment (Rs)'].apply(lambda x: 'Rich' if x > 200 else 'Poor')
# print(df.head())

# A4. Please refer to the data present in “IRCTC Stock Price” data sheet of the above excel file. Do the
# following after loading the data to your programming platform.

# Loading the CSV file into irctc variable
irctc = pd.read_excel("irctc.xlsx",sheet_name="IRCTC Stock Price")
irctc = irctc.dropna(axis=1)
# print(irctc.head())

# Calculate the mean and variance of the Price data present in column D.
# (Suggestion: if you use Python, you may use statistics.mean() &
# statistics.variance() methods).

# First Select feature Price
# PRICE = irctc['Price'].dropna(axis=1)
M = st.mean(irctc["Price"])
Var = st.variance(irctc["Price"])
print("Mean of the data: ",M)
print("Variance        : ",Var)

#  Select the price data for all Wednesdays and calculate the sample mean. Compare the mean
#  with the population mean and note your observations.
Wed = irctc[irctc['Day']=='Wed']
# print(Wed.head())

Wed_mean = st.mean(Wed['Price'])
# print("Sample mean: ",Wed_mean)

# Comparing the sample mean with population mean
if Wed_mean < M:
    print("Mean of price data for all Wednesdays is lesser than the mean of all price")
else:
    print("Mean of price data for all Wednesdays is greater than the mean of all price")

# Select the price data for the month of Apr and calculate the sample mean. Compare the
# mean with the population mean and note your observations.
Apr_data = irctc[irctc['Month']=='Apr']
Apr_mean = st.mean(Apr_data['Price'])
print("Mean of price of April month: ",Apr_mean)

if Apr_mean < M:
    print("Population mean is greater than the sample mean of April month")
else:
    print("Population mean is greater than the sample mean of April month")
# From the Chg% (available in column I) find the probability of making a loss over the stock.
# (Suggestion: use lambda function to find negative values)

loss = irctc[irctc['Chg%']<0]

loss_pr = len(loss)/len(irctc)
print("Probability of loss over the stock price: ",loss_pr)

'''Calculate the probability of making a profit on Wednesday.'''
profit = irctc[irctc['Chg%']>0]
profit_wed = len(Wed)/len(profit) # Wed data we have calculated earlier
print("Probability of profit on Wednesday       : ",profit_wed)
'''Calculate the conditional probability of making profit, given that today is Wednesday.'''
Total_wed = len(Wed)

# Total_profit = len(profit)
profit_on_Wed = len(Wed['Chg%']>0)
Cnd_pr = profit_on_Wed/Total_wed
print("Conditional probability: ",Cnd_pr)

'''Make a scatter plot of Chg% data against the day of the week'''
X = irctc['Chg%']
Y = irctc['Day']
plt.scatter(X,Y,label='Data Points',color='blue')
plt.xlabel("Chg%")
plt.ylabel("Week")
# plt.show()
