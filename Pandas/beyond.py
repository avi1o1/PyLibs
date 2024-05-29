import pandas as pd

""" Writing CSV files """
# dat = {'Name': ['GuSe', 'YaMi', 'ViPa'], 'Age': [19, 15, 18], 'City': ['DL', 'HY', 'UP']}
# frame = pd.DataFrame(dat)
# print(frame)
# frame.to_csv('People.csv', index=False)

""" Reading CSV files """
# people = pd.read_csv('People.csv')                                            # Reading CSV file normally
# print(people)

# tmp = pd.read_csv('People.csv', dtype={"Age": "float"})                       # Change data type of a column
# print(tmp)

# tmp = pd.read_csv('People.csv', nrows=2)                                      # Reading first 2 rows
# print(tmp)

# tmp = pd.read_csv('People.csv', usecols=["Name", "City"])                     # Reading specific columns
# print(tmp)
# tmp = pd.read_csv('People.csv', usecols=[0, 2])                               # Specific columns using index 
# print(tmp)

# tmp = pd.read_csv('People.csv', skiprows=[1, 3])                              # Skip certain rows [ 1-indexed ]
# print(tmp)

# tmp = pd.read_csv('People.csv', names=[1, 2, 3])                              # Set header [shifts header as 0-data]
# print(tmp)
# tmp = pd.read_csv('People.csv', index_col="Age")                              # Set index column
# print(tmp)
# tmp = pd.read_csv('People.csv', header=[1])                                   # Set header row [data before is skipped]
# print(tmp)

""" Missing Data Handlings """
# people = pd.read_csv('People.csv')
# print(people.isnull())                                                        # Check for missing values
# print(people.isnull().sum())                                                  # Count of missing values per column

# print(people.dropna())                                                        # Drop missing valued rows [default]
# print(people.dropna(axis=1))                                                  # Drop missing valued columns
# print(people.dropna(thresh=2))                                                # Drop rows with atleast 2 missing values
# print(people.dropna(how="all"))                                               # Drop rows with atleast all missing values
# print(people.dropna(subset=["Age"]))                                          # Drop rows with missing val in specific col

# print(people.fillna(0))                                                       # Fill missing values with 0
# print(people.fillna("NA", limit=1))                                           # Fill with NA, but max 1 per row
# print(people.fillna({"Age": 0, "City": "NA"}))                                # Fill with specific column values
# print(people.ffill())                                                         # Fill with previous value
# print(people.bfill())                                                         # Fill with next value

# print(people.interpolate())                                                   # Fill with interpolated values

""" Further Functions """
# people = pd.read_csv('People.csv')                                            # Reading file

# print(people.shape)                                                           # Shape of the data
# print(people.index)                                                           # Index values
# print(people.columns)                                                         # Column names
# print(people.head())                                                          # First 5 rows
# print(people.tail(1))                                                         # Last 3 rows

# print(people.describe())                                                      # Statistics of the data
# print(people['Name'].value_counts())                                          # Count of unique values
# print(people['Age'].sum())                                                    # Sum of a column

# print(people.loc[[0, 2], ["Name", "Age"]])                                    # Accessing specific data
# print(people.sort_values('Age', ascending=False))                             # Sort by a column

# print(people.index.array)                                                     # Convert to array
# print(people.to_numpy())                                                      # Convert to numpy array

""" Merge, Concat, Join and Append """
# a = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# b = pd.DataFrame({'B': [4, 0, 6], 'C': [7, 8, 9]})
# print(pd.merge(a, b))                                                           # Merging (non similar rows are dropped)
# print(pd.merge(a, b, on="B"))                                                   # Merging along a column
# print(pd.merge(a, b, how="left"))                                               # Left merge (all rows of left)
# print(pd.merge(a, b, how="right"))                                              # Right merge (all rows of right)
# print(pd.merge(a, b, how="outer", indicator=True))                              # Outer merge (all rows)

# a = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
# b = pd.DataFrame({'A': [9, 8, 7], 'B': [6, 5, 4], 'C': [3, 2, 1]})
# print(pd.merge(a, b, right_index=True, left_index=True, suffixes=('i', 'j')))   # Multi-index merge

# a = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# b = pd.DataFrame({'B': [4, 5, 6], 'C': [7, 8, 9]})
# c = pd.Series([10, 20, 30], name='D')
# d = pd.Series([40, 50, 60], name='E')
# print(pd.concat([c, d]))                                                      # Concatenation for series
# print(pd.concat([a, b]))                                                      # Concatenation along rows
# print(pd.concat([a, b], axis=1))                                              # Concatenation along columns
# print(pd.concat([a, b], join="inner"))                                        # Only common columns
# print(pd.concat([a, b], keys=["d1", "d2"]))                                   # Distinguishing the concats

# a = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# b = pd.DataFrame({'C': [7, 8], 'D': [10, 11]})
# print(a.join(b))                                                              # Joining along columns
# print(a.join(b, how="right"))                                                 # Right join
# print(a.join(b, how="outer"))                                                 # All columns
# print(a.join(b, how="inner"))                                                 # Common only
# print(a.join(b, lsuffix="left", rsuffix='right'))                             # Dealing with common column

# a = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# b = pd.DataFrame({'C': [7, 8], 'D': [10, 11]})
# a._append(b)                                                                   # Concat, but worse (deprecated)
# print(a)

""" Grouping """
# data = pd.DataFrame({'Name': ['GuSe', 'YaMi', 'ViPa', 'YaMi', 'ViPa'],
#                      'Age': [19, 15, 18, 19, 18],
#                      'City': ['DL', 'HY', 'UP', 'HY', 'UP']})

# print(data)
# grouped = data.groupby('City')                                                # Grouping by a column
# for x, y in grouped:                                                          # Printing grouped data
#     print(x, y, sep="\n", end="\n\n")
# print(grouped.get_group('DL'))                                                # Getting a specific group
# print(grouped.sum())                                                          # Sum of each group
# print(grouped.describe())                                                     # Statistics of each group

""" Melting and Pivot Tables """
# data = pd.DataFrame({'Name': ['GuSe', 'YaMi', 'ViPa', 'YaMi', 'ViPa'],
#                      'Age': [19, 15, 18, 19, 18],
#                      'City': ['DL', 'HY', 'UP', 'HY', 'UP']})

# print(pd.melt(data))                                                          # Melting the data
# print(pd.melt(data, id_vars=["Age"], var_name="field"))                       # Melting the data

# print(data.pivot(columns="City"))                                             # Pivot table
# print(data.pivot(columns="City", values="Name"))                              # Getting specific values
# print(data.pivot_table(index="Name", columns="City", aggfunc="mean"))         # Pivot table with aggregation
