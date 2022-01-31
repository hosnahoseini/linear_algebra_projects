5import numpy as np
import copy

def change_row(matrix, row1, row2):
    """change two row in a matrix
    """
    tmp = copy.copy(matrix[row1])
    matrix[row1] = matrix[row2]
    matrix[row2] = tmp

def get_row_multiplied(row, coefficient):
    """multiply row elements by coefficient
    """
    result = []
    for element in row:
        result.append(element * coefficient)
    return result

def print_array(array):
    """print array
    """
    for row in array:
        for element in row:
            print(element, end=" ")
        print()

def echelon_matrix(argumented, n, m):
    """convert an n * m argumented matrix to echlon from
    """
    last_pivot_row = 0
    pivots = []
    for column in range(m):
        pivot_founded = False
        
        # change row if pivot is 0
        if abs(argumented[last_pivot_row, column]) <= 0.00001:
            for row in range(last_pivot_row + 1, n):
                if argumented[row, column] != 0:
                    change_row(argumented, last_pivot_row, row)
                    
                    pivot_founded = True
                    break
            if not pivot_founded:
                #this column doesn't have pivot
                continue
                
        pivots.append((last_pivot_row, column))
        # make pivot one
        pivot = argumented[last_pivot_row, column]
        argumented[last_pivot_row] = get_row_multiplied(argumented[last_pivot_row], 1 / pivot)
        
        
        # make row under pivot zero
        for row in range(last_pivot_row + 1, n):
            argumented[row] += get_row_multiplied(argumented[last_pivot_row],-argumented[row, column])
            
        
        if last_pivot_row == n-1:
            break
            
        last_pivot_row += 1
        

    return pivots[::-1]

def reduced_form(echlon, pivots):
    """convert an n * m echlon from matrix to reduced form
    """
    for row, column in pivots:
        for upper_row in range(row - 1, -1, -1):
            echlon[upper_row] += get_row_multiplied(echlon[row],-echlon[upper_row, column])

def find_values(reduced, pivots, m):
    """find output values from reduced form of matrix
    """
    for row, column in pivots:
            x[column] = 0
            for col in range(m - 1):
                if col == column:
                    continue
                x[column] += -x[col] * reduced[row, col]
            x[column] += reduced[row, m - 1]

def print_result(array, x):
    """print valueof variables
    """
    print_array(array)
    for indx, variable in enumerate(x):
        print(f'X{indx + 1} = {variable}')

# get input
n, m = map(int, input().split())
argumented = np.array([input().strip().split() for _ in range(n)], float)
# array for output variables
x = [10] * (m - 1)

pivots = echelon_matrix(argumented, n, m)

reduced_form(argumented, pivots)

find_values(argumented, pivots, m)

print_result(argumented, x)


