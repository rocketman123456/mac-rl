import numpy as np
import matplotlib.pyplot as plt

# Create a 2D array (e.g., a 10x10 matrix)
data = np.random.rand(10, 10)  # Random values between 0 and 1

# Plot the heatmap
plt.imshow(data, cmap="viridis", aspect="auto")
plt.colorbar()  # Show color scale
plt.title("2D Array Heatmap")

# Show the plot
plt.show()
