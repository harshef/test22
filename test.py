from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
import matplotlib.pyplot as plt
"""
#random.shuffle(sequence_containing_x_vals)
#random.shuffle(sequence_containing_y_vals)
#random.shuffle(sequence_containing_z_vals)

x=np.array(([0.65343297079605855, 0.7694821123789386], [0.64511762540490381, 0.770163633869833], [0.65343297079605855, 0.7694821123789386], [0.65343297079605855, 0.7694821123789386], [0.65343297079605855, 0.7694821123789386], [0.65343297079605855, 0.7694821123789386], [0.64748555577181621, 0.77356314707744789], [0.64748555577181621, 0.77356314707744789], [0.64748555577181621, 0.77356314707744789], [0.64748555577181621, 0.77356314707744789], [0.64748555577181621, 0.77356314707744789], [0.64748555577181621, 0.77356314707744789], [0.64748555577181621, 0.77356314707744789], [0.65792054980771963, 0.7694821123789386], [0.65752844079213868, 0.77075952179618168]))
#figure = plt.figure()
#axes = figure.add_subplot(111)
#axes.scatter(x[:, 0], x[:, 1], '.r')
plt.scatter(x[:,0],x[:,1], s=15)

#ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
plt.show()
"""
temp1=np.array(([1,2,3,4]))
temp2=np.array(([4,5,6,7]))

dist = np.linalg.norm(temp1 - temp2)
print dist