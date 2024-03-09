"""
Numpy is a Python Library that provides a multi-dimensional array objects, which are faster and more efficient.
It also provides a large number of in-built functions that are quite useful in advanced analysis and data manipulation.
"""

""" Importing NumPy """
import numpy as np

""" Creating a NumPy Array """
# arr = np.array([1, 3, 5, 7, 9], ndmin=3, dtype=float)     # list to array, with dimension 3 and type int16
# arr = np.arange(1, 10, 2)                                 # start, end, step

# print(np.zeros((3, 3)))                                   # 3x3 array of zeros
# print(np.ones((3, 3)))                                    # 3x3 array of ones
# print(np.empty((3, 3)))                                   # 3x3 array of random values
# print(np.eye(3))                                          # 3x3 identity matrix
# print(np.linspace(1, 9, 5))                               # start, end, number of elements
# print(np.random.randint(1, 9, 5))                         # Generates random elements [ may use rand, randn, randf ]

""" Array Properties """    
# print(arr)                                                # printing the array
# print(arr.dtype)                                          # array elements type
# print(arr.ndim)                                           # number of dimensions
# print(arr.shape)                                          # shape of the array

""" Basic Manipulation """
# print(arr[0][0][2])                                       # Indexing (same as lists)
# print(arr[0, 0, 2])                                       # Indexing (similar to lists)

# print(arr[0][0][1:4:2])                                   # Slicing (same as lists)
# print(arr[0, 0, 1:4:2])                                   # Slicing (similar to lists)

# print(np.float32(arr))                                    # Changing element type
# print(arr.astype(float))                                  # Changing element type

# new = arr.reshape(1, 5, 1)                                # Reshapping
# new = np.resize(arr, (1, 5, 1))                           # Resizing
# print(new.reshape(-1))                                    # Converting to a 1D array
# print(new)

# temp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(temp.flatten(order="K"))                            # F: column-major, C: row-major, K: Original, A: Fortran-like
# print(temp.ravel(order="A"))                              # Same as flatten, but returns a view

# np.random.shuffle(arr)                                    # Shuffling the array
# print(arr)

# print(np.unique(arr,return_index=True,return_counts=True))# Unique elements in the array

# for ele in np.nditer(arr):                                # Iterating the array
#     print(ele)

# for index, ele in np.ndenumerate(arr):                    # Iterating array, with index
#     print(index, ele)                                     # shape of the array

""" Insert & Delete """
# print(np.append(arr, [2, 4, 6.9, 8]))                     # Appending elements at the end
# print(np.insert(arr, (1, 2, 3, 4), (2, 4, 6.9, 8)))       # Inserting elements at given indices

# temp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(np.append(temp, [[4, 2, 0]], axis=0))               # Appending arrays
# print(np.insert(temp, 1, [4, 2, 0], axis=1))              # Inserting arrays

# print(np.delete(arr, 2))                                  # Deleting elements at given indices
# print(np.delete(temp, 1, axis=0))                         # Deleting arrays

""" Arithmetic Operations """
# a = np.array([[1, 2, 3], [4, 5, 6]])
# b = np.array([[9, 8, 7], [6, 5, 4]])
# c = 3

# Addition
# print(a + c)
# print(a + b)
# print(np.add(a, b))

# Subtraction
# print(b - c)
# print(a - b)
# print(np.subtract(a, b))


# Multiplication
# print(a * c)
# print(a * b)
# print(np.multiply(a, b))


# Division
# print(a / c)
# print(a / b)
# print(np.divide(a, b))


# Modulus
# print(a % c)
# print(a % b)
# print(np.mod(a, b))


# Exponent
# print(a ** c)
# print(a ** b)
# print(np.power(a, b))

# Others
# print(np.reciprocal(a))
# print(np.sqrt(b))
# print(np.sin(a))
# print(np.cos(b))

""" Arithmetic Functions """
# arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# print(f"Minimum value of {np.min(arr)} at index {np.argmin(arr)}.")                     # for the whole array
# print(f"Maximum value of {np.max(arr)} at index {np.argmax(arr)}.")                     # for the whole array

# print(f"Maximum value of {np.min(arr, axis=1)} at index {np.argmin(arr, axis=1)}.")     # column-wise
# print(f"Maximum value of {np.max(arr, axis=0)} at index {np.argmax(arr, axis=0)}.")     # row-wise

# print(np.cumsum(arr))

""" Broadcasting : Technique for performing operations between different-sized matrices.
    Condition : Same dimension, or 1 for one of the array """
# a = np.array([[1], [2]])
# b = np.array([[3, 4, 5], [6, 7, 8]])
# print(a)
# print()
# print(b)
# print()
# print(a+b)

""" Copy (creates a new entity) vs View (shadow of the original array only) """
# c = arr.copy()
# v = arr.view()
# print('Original :', arr, '\nCopy \t :', c, '\nView \t :', v) 

# c[0] = 69
# print('\nOriginal :', arr, '\nCopy \t :', c)


# v[0] = 69
# print('\nOriginal :', arr, '\nView \t :', v)

""" Join & Split """
# a = np.array([[1, 2], [3, 4]])
# b = np.array([[5, 6], [7, 8]])

# print(np.concatenate((a, b), axis=0))
# print(np.concatenate((a, b), axis=1))

# print(np.stack((a, b), axis=0))
# print(np.stack((a, b), axis=1))
# print(np.hstack((a, b)))                                  # along row
# print(np.vstack((a, b)))                                  # along column
# print(np.dstack((a, b)))                                  # along height

# print(np.array_split(a, 3, axis=0))
# print(np.array_split(a, 2, axis=1))

""" Sort """
# arr = np.array([[3, 2, 4], [5, 0, 1]])
# print(np.sort(arr, axis=0))
# print(np.sort(arr, axis=1))

""" Search """
# arr = np.array([0, 1, 1, 1, 1, 1, 2, 4, 4, 7])            # arr must be sorted
# print(np.where( (arr%2) == 1 ))                           # indexes of the search value
# print(np.searchsorted(arr, 9))                            # index of the value where it would occur in sorted array
# print(np.searchsorted(arr, [6, 9], side="left"))          # search an array of elements, indexed from right end

""" Filter """
# arr = np.array([7, 2, 1, 1, 4, 1, 0, 1, 1, 4])
# f = [True, False, True, False, True, False, True, False, True, False]
# print(arr[f])

""" Matrices """
# a = np.matrix([[1, 2, 2], [9, 1, 8], [1, 1, 2]])
# b = np.matrix('1 0 0; 0 1 0; 0 0 1')
# print(a)
# print(b)

# print(a * b)                                              # Matrix multiplication
# print(a.dot(b))                                           # Matrix Multiplication

# print(np.transpose(a))                                    # Transpose of a matrix
# print(a.T)                                                # Transpose of a matrix
# print(np.swapaxes(b, 1, 0))                               # Swapping axes

# print(np.linalg.det(b))                                   # Determinant of a matrix
# print(np.linalg.inv(b))                                   # Inverse of a matrix

# print(np.linalg.matrix_power(a, 1))                       # Power of a matix
# print(np.linalg.matrix_power(a, 0))                       # Identity matrix
# print(np.linalg.matrix_power(a, -1))                      # Power of inverse of the matrix
