"""
Pandas is a Python Library that provides data structures (Series, Data FRame and Panels) and data analysis tools.
Thus, finding extensive use in fields of data science and machine learning.
"""

""" Importing Pandas """
import pandas as pd

""" Series [ 1-D ] """
# ele = [6, 9, 4, 2, 0]
# s1 = pd.Series(ele)                                                               # Using simple list
# print(s1)

# idx = ['a', 'b', 'c', 'd', 'e']                                                   # Custom indexing and data type
# s2 = pd.Series(ele, index=idx)
# print(s2)
     
# s3 = pd.Series({'a': 6, 'b': 9, 'c': 4, 'd': 2, 'e': 0})                          # Using dictionary
# print(s3)

# s4 = pd.Series(15, index=[1, 2, 3])                                               # Using scalar value
# print(s4)

# print(s1[2])                                                                      # Accessing elements
# print(s3['d'])

""" Data Frame [ 2-D ] """
# d1 = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 5, 6, 7], 'C': [7, 8, 9, 0]})      # Similar to series
# print(d1)

# tmp = {'g': pd.Series([7, 2, 1, 1, 4]), 's': pd.Series([1, 0, 1, 1, 4])}          # Using series
# d2 = pd.DataFrame(tmp)
# print(d2)

# d3 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'], index=['a', 'b', 'c'])
# print(d3)

# print(d3['A'])                                                                    # Accessing columns
# print(d3.loc['a'])                                                                # Accessing rows
# print(d3.iloc[0])                                                                 # Accessing rows using index
# print(d3.iloc[1, 1])                                                              # Accessing element

""" Operations """
# a = pd.Series([1, 2, 3, 4, 5])
# b = pd.Series([6, 9, 4, 2, 0])
# c = pd.Series(25, index=[1, 2, 3, 4, 5])
# d = pd.Series(35, index=[6, 7, 8, 9, 0])
# print(a + b)                                                                      # Arithmetic operation
# print(b + c)                                                                      # Missing values
# print(c + d)                                                                      # Different indexes
# print(a.add(b, fill_value=0))                                                     # Fill missing values

# e = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 5, 6, 7], 'C': [7, 8, 9, 0]})
# print(e["A"] + e["B"])                                                            # Column wise operation
# print(e.iloc[0] + e.iloc[-1])                                                     # Row wise operations

# f = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
# f["B < 5"] = f["B"] < 5                                                           # Conditional operations
# print(f)

""" Data Manipulation """
# g = pd.DataFrame({'A': [1, 2, 3], 'C': [7, 8, 9]})
# print(g)

# g.loc[1, "C"] = 0                                                                  # Changing a value
# print(g)

# print(g.replace(1, 0))                                                             # Replacing a value (to, with)
# print(g.replace({1: 0, 7: 0}))                                                     # Replacing multiple values
# print(g.replace([1, 2, 3], 5))                                                     # Replacing multiple values
# print(g.replace(1, 0, limit=2))                                                    # Replacing limited values

# g.insert(1, 'B', [4, 5, 6])                                                        # Inserting a column
# print(g)

# g["tmp"] = g["C"][:2]                                                              # Slicing a column
# print(g)

# print(g.pop("tmp"))                                                                # Removing a column
# print(g)

# g = g.drop(1)                                                                      # Removing a row
# print(g)
