# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
#from pylab import *

##################
# IMPORT DATA
##################

# Load the data
data = np.loadtxt('data.csv', delimiter = ',', skiprows = 1)

# Assign the results of the measurement to variables and normalise y-axis
x1 = np.array(data[:, 0])
y1 = np.array(data[:, 1])
y1 = np.divide(y1, max(y1))

# Assign the results of the second measurement but don't use them here
x2 = np.array(data[:, 2])
y2 = np.array(data[:, 3])

##################
# FITTING DATA
##################

# Define the function to fit the data with (in this case an error function)
def func(x, p1, p2, p3, p4):
  return (p1 / 2) * sp.special.erf(np.sqrt(2) * (x - p2) / p3) + p4

# Fit the data using 'curve_fit'
popt, pcov = curve_fit(func, x1, y1, p0 = (1.0, 3.5, 1.4, 1.0))

# Extract the fit parameters
p1 = popt[0]
p2 = popt[1]
p3 = popt[2]
p4 = popt[3]

# Calculate the fit residuals
residuals = y1 - func(x1, p1, p2, p3, p4)

# Calculate the residual sum of squares
fres = sum(residuals**2)

# Calculate the error estimates for the fit parameters
error = [] 
for i in range(len(popt)):
    try:
      error.append(np.absolute(pcov[i][i])**0.5)
    except:
      error.append( 0.00 )

# Assign the fit outputs to more explanatory variable names
pfit_curvefit = popt
perr_curvefit = np.array(error)

# Define a new (higher binned) x-axis to plot the corresponding Gaussian curve
xNew = np.linspace(min(x1), max(x1), 100)

# Calculate the error function that best fits the data with x = xNew
yFit = func(xNew, *popt)

# Calculate the Gaussian curve correspondiong to the fit using x = xNew
yGauss = p1 * np.exp(-pow((xNew - p2), 2) / (2 * pow(p3 / 2, 2)))

# Calculate the Gaussian curve corresponding to the raw x-axis
yGaussRaw = p1 * np.exp(-pow((x1 - p2), 2) / (2 * pow(p3 / 2, 2)))

##################
# PLOTTING RESULTS
##################

# Define the plot dimensions (inches)
fig_width = 7
fig_height = 9
fig_size = [fig_width, fig_height]

# Define the plot parameters
col_back        = '#393939'
col_data_edge   = 'black'#'#9D9D9D'
col_data_face   = 'red'#'#F05A60'
col_line        = '#9D9D9D'
col_red         = 'red'
col_fill        = '#5A9BD3'
style_fit       = '-'
style_data      = 'o'
linewidth       = 2
dotsize         = 150

# Set plot background color to dark grey
mpl.rcParams['figure.facecolor'] = col_back

# Set figure size to the specified dimensions
plt.figure(figsize = fig_size)

# Set the spacing between axes using gridspec from matplotlib
# I wanted a two panel plot but I only wanted one x-axis. This is the solution
# that worked best after googling for tips
gs1 = gridspec.GridSpec(2, 1)
gs1.update(wspace = 0.001, hspace = 0.001) # set the spacing between ax1es. 

# Specify the plot font parameters
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 16}
plt.rc('font', **font)

# Sepcify the axes limits
# The weird values are so that the data points don't get cut off
xlimlow     = 0.89
xlimhigh    = 6.12
x_lim       = [xlimlow, xlimhigh]
ylimlow     = -0.04
ylimhigh    = 1.05
y_lim       = [ylimlow, ylimhigh]

# Specify the spine limits
xspinelow   = 1
xspinehigh  = 6
yspinelow   = 0
yspinehigh  = 1

# Specify the start and finish positions of the fills
# The frac variable can be looped over to create the frames for a gif image
frac        = .50;
frac_str    = str(round((1-frac)*100))
frac_str    = frac_str.rjust(3, '0')
fill_start  = 0
fill_end    = round(frac * len(xNew))

##################
# FIRST SUBPLOT
##################

ax1 = plt.subplot(gs1[0])

# Set panel background color to dark grey
ax1.set_axis_bgcolor(col_back)

# axes labels
#xLab = 'Position (mm)' # commented out becauas I don't want an x label here
yLab = 'Signal (arb. units)'

# Plot the Gaussians corresponding to the fit (solid) and the raw data (points)
# Determine which plot is on top of the others using zorder
plt.plot(xNew, yGauss, style_fit, lw = linewidth, c = col_line, zorder = 2)
plt.scatter(x1, yGaussRaw, s = dotsize, facecolors = col_data_face,
            edgecolors = col_data_edge, zorder = 3)
ax1.fill_between(xNew[fill_start:fill_end], 0, yGauss[fill_start:fill_end],
                 facecolor = col_data_face, zorder = 1, alpha = 0.25)

# Set visibilities of axes and slightly move the left and top spines outwards
ax1.spines['bottom'] .set_bounds(xspinelow, xspinehigh)
ax1.spines["bottom"] .set_color(col_line)
ax1.spines["bottom"] .set_position(('outward', 10))
ax1.spines["bottom"] .set_visible(False)
ax1.spines['left']   .set_bounds(yspinelow, yspinehigh)
ax1.spines["left"]   .set_color(col_line)
ax1.spines["left"]   .set_position(('outward', 10))
ax1.spines["left"]   .set_visible(True)
ax1.spines["right"]  .set_visible(False)
ax1.spines["top"]    .set_visible(False)
  
# Ticks only on the bottom and left axes
ax1.get_xaxis().tick_bottom()    
ax1.get_yaxis().tick_left()

# Another method for above
#ax1.xax1is.set_ticks_position('bottom')
#ax1.yax1is.set_ticks_position('left')    

# Set the y-axis properties
ax1.tick_params(axis = 'y', colors = col_line)
ax1.set_ylim(y_lim) # A little bit outside range so points are visible
ax1.set_yticks((0, 0.5, 1.0))

# Set the x-axis properties
ax1.tick_params(axis = 'x', colors = col_line)
ax1.set_xlim(x_lim) # A little bit outside range so points are visible
ax1.set_xticks([]) # No ticks

# Set labels
ax1.set_ylabel(yLab, color = col_line)

##################
# SECOND SUBPLOT
##################

ax2 = plt.subplot(gs1[1]) 

# Set panel background color to dark grey
ax2.set_axis_bgcolor(col_back)

# axes labels
xLab = 'Razor position (mm)'
yLab = 'Laser Power (arb. units)'

# Plot the raw data and the error function corresponding to the fit (solid)
# Determine which plot is on top of the others using zorder
plt.plot(xNew, yFit, style_fit, lw = linewidth, c = col_line, zorder = 2)
plt.scatter(x1, y1, s = dotsize, facecolors = col_data_face,
            edgecolors = col_data_edge, zorder = 3)
ax2.fill_between(xNew[fill_start:fill_end], 0, yFit[fill_start:fill_end],
                 facecolor = col_data_face, zorder = 1, alpha = 0.25)

# Set visibilities of axes and slightly move the left and top spines outwards
ax2.spines['bottom'] .set_bounds(xspinelow, xspinehigh)
ax2.spines["bottom"] .set_color(col_line)
ax2.spines["bottom"] .set_position(('outward', 10))
ax2.spines["bottom"] .set_visible(True)
ax2.spines['left']   .set_bounds(yspinelow, yspinehigh)
ax2.spines["left"]   .set_color(col_line)
ax2.spines["left"]   .set_position(('outward', 10))
ax2.spines["left"]   .set_visible(True)
ax2.spines["right"]  .set_visible(False)
ax2.spines["top"]    .set_visible(False)
  
# Ticks only on the bottom and left axes    
ax2.get_xaxis().tick_bottom()    
ax2.get_yaxis().tick_left()   

# Set the x-axis properties
ax2.tick_params(axis = 'x', colors = col_line)
ax2.set_xlim(x_lim) # A little bit outside range so points are visible
ax2.set_xticks((1, 2, 3, 4, 5, 6))

# Set the y-axis properties
ax2.tick_params(axis = 'y', colors = col_line)
ax2.set_ylim(y_lim) # A little bit outside range so points are visible
ax2.set_yticks((0, 0.5, 1.0))

# Set labels
ax2.set_xlabel(xLab, color = col_line)
ax2.set_ylabel(yLab, color = col_line)

##################
# SECOND SUBPLOT
##################

fileName = '.'.join(['plot', frac_str, 'png'])
plt.savefig(fileName, facecolor = col_back, transparent = False,
            bbox_inches='tight')