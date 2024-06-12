import matplotlib.pyplot as plt
from import_data import ImportData

filename = 'Archive' #The filename (must be in current directory)

model = ImportData(filename) #Initialise the class

sorted_list = model.sort_timesteps() #Sort the timesteps from the least Nan values to the most (better to plot with). Choose the index whi

#Note that the first element in the list (within the list) is the time index.
#THe second value is the number of non Nan-Values
model.time_for_index(9)

model.plot_NDVI(9)

