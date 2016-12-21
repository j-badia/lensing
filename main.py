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
    integrator.set_integrator("dopri5")
    integrator.set_f_params(*args)
    integrator.set_initial_value(y0, t_array[0])
    y = np.zeros([len(t_array), len(_to_it(y0))])
    y[0] = y0
    last_point = 0
    for n in range(len(t_array)-1):
        if y[n][0] < 1.5 or not integrator.successful():
            break
        y[n+1] = integrator.integrate(t_array[n+1])
        last_point += 1
    return (y, last_point)

def F(t, y, b):
    """y' = F(t, y, b)
    y is a tuple (r,v,phi)
    """
    (r, v, phi) = y
    acceleration = 0.5 * b**2 * (2*r-3) / r**4
    return np.array((v, acceleration, b/r**2))

def jac(t, y, b):
    """Jacobian for F"""
    return np.array([[0, 1, 0],
                     [3*(2-r)/r**5, 0, 0],
                     [-2*b/r**3, 0, 0]])

### Parameters
    
n_points = 1000
n_rays = 100
r0 = 50 # Measured in units of Sch. radius



fig = plt.figure()

#Draw horizon and photon sphere
r_h = 1
r_ps = 1.5
phi_bh = np.linspace(0, 2*math.pi, 1000)

b_ps = 3*math.sqrt(3)/2
alpha_ps = math.pi - math.asin(math.sqrt(1-1/r0)*b_ps/r0)

rays = np.zeros((n_rays, 2, n_points))
alphas = np.zeros(n_rays)
last_ray = 0

for i in range(n_rays):
    # alpha is measured from x-axis up
    alpha = 7*math.pi/8 + ((i/n_rays-1)**15 + 1) * (alpha_ps - 7*math.pi/8+0.00000001)
#    alpha = alpha_ps+0.00000001
    b = math.sin(alpha) * r0 / math.sqrt(1-1/r0)
    v0 = math.cos(alpha)
    (data, last_point) = integrate(F, (r0, v0, 0), np.linspace(0, 200, n_points), args=(b,))
    (r, v, phi) = np.transpose(data[0:last_point+1])
    (x, y) = (r*np.cos(phi), r*np.sin(phi))
    r_min = np.min(r)
    print(i, alpha_ps - alpha, b, r_min/math.sqrt(1-1/r_min), r_min)
    if r_min < 1.5:
        break
    rays[i][0] = x
    rays[i][1] = y
    alphas[i] = alpha
    last_ray = i
    
def plot_bh():
    plt.xlim(-r0, r0)
    plt.ylim(-r0, r0)
    plt.axes().set_aspect("equal")    
    plt.plot([0],[0],"ok")
    plt.plot(r_h*np.cos(phi_bh), r_h*np.sin(phi_bh), 'k')
    plt.plot(r_ps*np.cos(phi_bh), r_ps*np.sin(phi_bh), 'k')

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
        plt.cla()
        plot_bh()

fig.canvas.mpl_connect("key_press_event", press)
plot_bh()
plt.show()