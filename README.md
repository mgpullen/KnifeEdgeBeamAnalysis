# KnifeEdgeBeamAnalysis
The analysis performed using the files in this repository was utilised in one of my blog posts: https://mgpullen.wordpress.com/2016/08/26/python-lasers-and-non-linear-regression-part-2/

A script is provided that analyses and plots the results of a so-called 'knife-edge' beam profile measurement:
http://massey.dur.ac.uk/resources/grad_skills/KnifeEdge.pdf

The data is contained in the accompanying .csv file. The data is imported and then fitted with an error function using the 'curve_fit' function. The results of the fit are used to construct the corresponding Gaussian profile of the laser beam. A two panel figure is created that summarises the results.
