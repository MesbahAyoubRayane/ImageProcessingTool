import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a 1x2 grid of subplots
plt.subplot(1, 2, 1)  # 1 row, 2 columns, select the first subplot
plt.plot(x, y1)
plt.title('Sin Function')

plt.subplot(1, 2, 2)  # 1 row, 2 columns, select the second subplot
plt.plot(x, y2)
plt.title('Cos Function')

plt.show()