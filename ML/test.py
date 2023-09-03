import pandas as pd 
import numpy as np 
import statistics as st
# import matplotlib.pyplot as plt 
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
