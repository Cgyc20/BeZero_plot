import matplotlib.pyplot as plt
from import_data import ImportData
import os


current_dir = os.path.dirname(os.path.abspath(__file__))  # Current directory where script is located
parent_dir = os.path.dirname(current_dir)  # Parent directory
archive_dir = os.path.join(parent_dir, 'Archive')  # 'Archive' directory i


model = ImportData(archive_dir) #Initialise the class

sorted_list = model.sort_timesteps() #Sort the timesteps from the least Nan values to the most (better to plot with). Choose the index whi

for elements in sorted_list:
    print(f"Time index: {elements[0]}, Total data points {elements[1]}" )

#Note that the first element in the list (within the list) is the time index.
#THe second value is the number of non Nan-Values
model.time_for_index(31)

model.plot_NDVI(31)

