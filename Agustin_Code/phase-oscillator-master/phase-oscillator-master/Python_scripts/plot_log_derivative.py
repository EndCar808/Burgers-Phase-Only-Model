#Agustin Arguedas
#Example script for calculating log-derivative of data
#17.12.20

import numpy as np
#Central derivative.
#Left and right elements of the array are derivated by right and left finite differences, respectively.
def central_derivative(f,x):
    result=np.empty(f.size)
    result[1:-1]=(f[2:]-f[:-2])/(x[2:]-x[:-2])
    result[0]=(f[1]-f[0])/(x[1]-x[0])
    result[-1]=(f[-1]-f[-2])/(x[-1]-x[-2])
    return result
#Get the log-derivative of data.
#Transform the data into log-log space. Apply central derivative there.
#Then return the log-derivative together with the corresponding x_data coordinates.
def get_log_derivative(x_data,
                       y_data
                       ):
    log_x=np.log(x_data)
    log_y=np.log(y_data)
    return x_data, central_derivative(log_y,log_x)