# -*- coding: utf-8 -*-

import collections
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

def _to_it(x):
    """If x is not iterable, puts it inside one"""
    if isinstance(x, collections.Iterable):
        return x
    else:
        return (x,)

def integrate(f, y0, t_array, args=None):
    """RK4 integrator
    Solves y' = f(t, y, *params) over interval T
    y may be a vector
    N: number of points to calculate; returns N+1 points
    """
    y0 = np.array(y0)
    if args is None:
        args = ()
        def g(t, y, *p):
            return f(t, y)
    else:
        g = f
    integrator = scipy.integrate.ode(g)
    integrator.set_f_params(*args)
    integrator.set_initial_value(y0, t_array[0])
    y = np.zeros([len(t_array), len(_to_it(y0))])
    y[0] = y0
    last_point = 0
    for n in range(len(t_array)):
        if y[n][0] < 1.5:
            break
        
        last_point += 1
    return (y, last_point)

n_points = 10000

# Measured in units of Sch. radius
r0 = 50

def F(t, y, b):
    """y is a tuple (r,v,phi)"""
    (r, v, phi) = y
    if r < 1.5:
        return np.array((0, 0, 0))
    acceleration = 0.5*(-1+r)*(2*r**3 + b**2 * (5-7*r+2*r**2))/r**6
    return np.array((v, acceleration, b*(1-1/r)/r**2))


    
n_rays = 200

fig = plt.figure()

#Draw horizon and photon sphere
r_h = 1
r_ps = 1.5
phi = np.linspace(0, 2*math.pi, 1000)

b_ps = 3*math.sqrt(3)/2
beta_ps = -math.acos(math.sqrt(1-1/r0)*b_ps/r0)
alpha_ps = math.pi/2 - beta_ps

def _cubic(x):
    """Maps [0,1] to [0,1] like a cubic: flatter near x=1/2"""
    return 0.5 * (2*(x-1/2))**3 + 1/2

rays = np.zeros((n_rays, 2, n_points))
alphas = np.zeros(n_rays)
last_ray = 0

for i in range(n_rays):
    # alpha is measured from x-axis up, beta is measured from r=const to the right
    alpha = 7*math.pi/8 + ((i/n_rays-1)**15 + 1) * (alpha_ps - 7*math.pi/8) 
    beta = math.pi/2 - alpha
    b = math.cos(beta) * r0 / math.sqrt(1-1/r0)
    v0 = math.sin(beta) * (1-1/r0)
    data = scipy.integrate.odeint(F, (r0, v0, 0), np.linspace(0, 200, n_points), args=(b,))
    (r, v, phi) = np.transpose(data)
    (x, y) = (r*np.cos(phi), r*np.sin(phi))
    rays[i][0] = x
    rays[i][1] = y
    alphas[i] = alpha
    r_min = np.min(r)
    print(i, alpha,  b, r_min)
    last_ray = i
    if r_min < 1.5:
        break
    
def plot_bh():
    plt.xlim(-r0, r0)
    plt.ylim(-r0, r0)
    plt.axes().set_aspect("equal")    
    plt.plot([0],[0],"ok")
    plt.plot(r_h*np.cos(phi), r_h*np.sin(phi), 'k')
    plt.plot(r_ps*np.cos(phi), r_ps*np.sin(phi), 'k')

def plot_animated(rays, last_ray, alphas):
    for i in range(last_ray+1):
        plt.cla()
        plot_bh()
        plt.plot(rays[i][0], rays[i][1])
        plt.xlabel("angle = {:08.5f}º - i = {}".format(180-alphas[i]*180/math.pi, i))
        fig.canvas.draw()
        plt.pause(0.05)
    
def press(event):
    if event.key == "r":
        plot_animated(rays, last_ray, alphas)
    if event.key == "b":
        plot_bh()

fig.canvas.mpl_connect("key_press_event", press)
plot_bh()
plt.show()