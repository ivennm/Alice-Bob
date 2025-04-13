import matplotlib.pyplot as plt
    
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 3, 5]
    
    # Plotting with different colors
plt.rcParams['axes.facecolor'] = '#000080'
plt.plot(x, y, color='blue')
plt.plot(x, [i+1 for i in y], color='#00FF00')
plt.plot(x, [i+2 for i in y], color=(0, 0, 1, 1))
plt.plot(x, [i+3 for i in y], color='c')
plt.plot(x, [i+4 for i in y], color='0.5')
plt.plot(x, [i+5 for i in y], color='xkcd:sky blue')
plt.plot(x, [i+6 for i in y], color='tab:pink')
plt.plot(x, [i+7 for i in y], color='C2')

plt.show()