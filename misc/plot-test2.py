import numpy as np
import matplotlib.pyplot as plt

def sort(table):
    n = len(table)
    
    for i in range (n):
        for j in range (n):
            if table[i] < table[j]:
                tmp = table[i]
                table[i] = table[j]
                table[j] = tmp
            plt.plot(table, 'ro')
            plt.title(f"i {i} j {j}")
            plt.pause(0.001)
            plt.clf() # clear figure
    return table

n = 50
table =  np.random.randint(1,101,n)
sort(table)