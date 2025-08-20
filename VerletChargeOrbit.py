"""
A program to model the orbit of a charged particle in 3D space about fixed point charges
Using the Verlet velocity method and plotly 3D plotting library.
Author: Aidan Gebbie
Date: 17 August 2025
"""

# Model assumption: 
# * All charged objects are modelled as sets of evenly distributed point charges
# * Other forces are insignificant compared to the coulombic forces
# * Use of classical newtonian mechanics and Verlet-integration approximation

import numpy as np
import plotly.io as pio
import plotly.graph_objects as go

#declare constants (valid shapes: rod, ring, disk, sphere, dipole)
shape = "rod"
radius = 1.0
charge = 5e-8

#charge of interest
qsys = -1.6e-19                  #charge on partcle in Coulombs
m =  9e-31  #1.7e-27             #mass of particle (electron/proton)
vsys = np.array([0,0,1e7])       #initial velocity of particle
rsys = np.array([0.2, 0.5, 0])   #initial position of particle

# final time
tf = 1e-5
dt = 1e-10

# export filename
filename = "plot.html"

#Coulomb's constant
coul_const = 9e9   

# number of point charge to model shape as
n = 1000

# function to create an array of point charges for a dipole
def dipole(seperation, charge):
    point_charges = np.zeros([2,4])
    point_charges[0,0] = -0.5 * seperation
    point_charges[1,0] = 0.5 * seperation
    point_charges[0,3] = -charge
    point_charges[1,3] = charge
    
    return point_charges    

# function to create an array of point charges for the approximation of a ring
def ring(n, ring_radius, charge):
    unit_charge = charge/n
    point_charges = np.zeros([n,4])
    for i in range(n):
        point_charges[i,0] = 0
        point_charges[i,1] = ring_radius * np.cos(2 * np.pi / n * i)
        point_charges[i,2] = ring_radius * np.sin(2 * np.pi / n * i)
        point_charges[i,3] = unit_charge
        
    return point_charges

# function to create an array of point charges for the approximation of a rod
def rod(n, rod_length, charge):
    unit_charge = charge/n
    # array of vectors of the point charges constituting the ring
    point_charges = np.zeros([n,4])
    for i in range(n):
        point_charges[i,0] = 0
        point_charges[i,1] = (1/n * i - 0.5) * rod_length
        point_charges[i,2] = 0
        point_charges[i,3] = unit_charge
        
    return point_charges


# function to create an array of point charges for the approximation of a disk
def disk(n, radius, charge):
    point_charges = np.zeros([n, 4])
    unit_charge = charge/n
    
    # Uses the golden angle to evenly distribute points on a disk
    
    phi = np.pi * (3 - np.sqrt(5))  # ≈ 2.39996 radians

    for i in range(n):
        r = np.sqrt(i / n)  # radial distance
        theta = i * phi           # angle in radians

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        point_charges[i,0] = 0
        point_charges[i,1] = x * radius
        point_charges[i,2] = y * radius
        point_charges[i,3] = unit_charge

    return point_charges


# function to create an array of point charges for the approximation of a sphere
def sphere(n, radius, charge): # uses the golden angle to evenly distribute points on the surface of a sphere
    unit_charge = charge/n
    phi = np.pi * (3. - np.sqrt(5))  # golden angle in radians
    
    point_charges = np.zeros([n, 4])

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        r = np.sqrt(1 - y * y)     # radius at y

        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        
        point_charges[i,0] = x * radius
        point_charges[i,1] = y * radius
        point_charges[i,2] = z * radius
        point_charges[i,3] = unit_charge
        
    return point_charges


# function to determine the acceleration of a particle at a given point in space
def update_accel(rsys, qsys, m, charges):
    # charges: 0,1,2 -> x,y,z; 3 -> charge
    
    # calculate all the r vectors from the point charges to charge of interest
    r = rsys - charges[:, :3]
    
    # calculate the magnitudes of r
    magr = np.linalg.norm(r, axis=1)
    
    # E = 1/4πε₀ * q/|r|² * r_hat = 1/4πε₀ * q/|r|³ * r
    E_contrib = r * (charges[:,3] / magr**3)[:, None]
    
    # sum all the components of E and multiply by coulomb's constant
    E = np.sum(E_contrib, axis=0) * coul_const

    # Return the acceleration: F = qE; a = F/m
    return qsys * E / m 

# generate array of point charges
charges = None
if   shape == "ring": charges = ring(n, radius, charge)
elif shape == "rod": charges = rod(n, radius, charge)
elif shape == "sphere": charges = sphere(n, radius, charge)
elif shape == "disk": charges = disk(n, radius, charge)
elif shape == "dipole": charges = dipole(radius, charge)

# generate plot labels
if shape == "rod": shape_label = shape + f" (length = {radius:.2g}m, charge = {charge:.2g}C)"
if shape == "dipole": shape_label = shape + f" (seperation = {radius:.2g}m, charge = {charge:.2g}C)"
else: shape_label = shape + f" (radius = {radius:.2g}m, charge = {charge:.2g}C)"
point_label = f"point charge (start v = ⟨{vsys[0]:.2g}, {vsys[1]:.2g}, {vsys[2]:.2g}⟩m∙s⁻¹)"
start_label = f"start position = ⟨{rsys[0]:.2g}, {rsys[1]:.2g}, {rsys[2]:.2g}⟩m"
title=f"A point charge ({qsys:.2g}C) orbiting a charged {shape} for an interval of {tf:.2g} seconds."

# generate first particle acceleration
asys = update_accel(rsys, qsys, m, charges)

# initialize arrays of x,y,z, and speed components
indeces = int(tf/dt) + 1
x = np.zeros([indeces])
y = np.zeros([indeces])
z = np.zeros([indeces])
v = np.zeros([indeces])

i = 0
for i in range(indeces):
    t = i * dt    #calculate next time
    
    # export current particle data
    x[i] = rsys[0]
    y[i] = rsys[1]
    z[i] = rsys[2]
    v[i] = np.sqrt(vsys[0]**2 + vsys[1]**2 + vsys[2]**2) # get velocity magnitude
    
    """ verlet velocity method """
    # Description of algorithm from wikipedia article:
    # calculate x(t+Δt) = x(t) + v(t)Δt + 1/2 a(t)Δt²
    # Derive a(t+Δt) from interaction potential using x(t+Δt)
    # Calculate v(t+Δt) = v(t) + 1/2 (a(t) + a(t+Δt))Δt    
    
    rsys = rsys + vsys * dt + 0.5 * asys * dt * dt
    a_new = update_accel(rsys, qsys, m, charges)
    vsys = vsys + 0.5 * (asys + a_new) * dt
    asys = a_new
    



""" plotly render""" # using code provided by Microsoft Copilot

# Create line segments for trajectory
trajectory = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='lines',
    line=dict(
        color=v,
        colorscale='Viridis',
        colorbar=dict(title='v (m∙s⁻¹)'),
        width=4
    ),
    name=point_label
)


# positive point charges as red scatter points
pos = charges[:][charges[:,3]>0]
pos = go.Scatter3d(
    x=pos[:, 0],
    y=pos[:, 1],
    z=pos[:, 2],
    mode='markers',
    marker=dict(size=5, color='red'),
    name=shape_label
)

# negative point charges as blue scatter points

neg = charges[:][charges[:,3]<0]
neg = go.Scatter3d(
    x=neg[:, 0],
    y=neg[:, 1],
    z=neg[:, 2],
    mode='markers',
    marker=dict(size=5, color='blue'),
    name="negative charge"
)

# Start and stop markers
start = go.Scatter3d(
    x=[x[0]], y=[y[0]], z=[z[0]],
    mode='markers',
    marker=dict(size=6, color='black', symbol='diamond'),
    name=start_label
)

stop = go.Scatter3d(
    x=[x[-1]], y=[y[-1]], z=[z[-1]],
    mode='markers',
    marker=dict(size=6, color='black', symbol='square'),
    name='stop position'
)

# Combine all traces
fig = go.Figure(data=[pos, neg, trajectory, start, stop])

# Set layout
fig.update_layout(
    title=title,
    scene=dict(
        xaxis_title='x (m)',
        yaxis_title='y (m)',
        zaxis_title='z (m)'
    ),
    legend=dict(x=0.02, y=0.98)
)

# render plot in webbrowser
pio.renderers.default = 'browser'
fig.show()

# export plot as an html file
fig.write_html(filename)