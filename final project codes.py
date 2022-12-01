import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from scipy import stats
from scipy.stats import norm


# Compute Heat Equation Solution using Explicit Method
# Spatial discretization
m = 100
x = numpy.linspace(-2, 0.5, m)
delta_x = 2.5 / (m - 1.0)

# Time discretization - Choose \Delta t based on accuracy constraints
C = 0.05
delta_t = C * delta_x
t = numpy.arange(0.0, 0.06, delta_t)
N = len(t)
r=0.1
sigma=0.2
T=3
K=100
k=2*r/sigma**2
alpha=(1-k)/2
beta=-(k+1)**2/4
# Solution array
U = numpy.empty((N + 1, m))

#initial condition
for i in range(m):
    if x[i]>0:
        U[0,i]=(np.exp(x[i])-1)/np.exp(alpha*x[i])
    else:
        U[0,i]=0

#boundary condition
t_true=lambda t: T-2*t/sigma**2
d1=lambda a,t: (a+(r+1/2*sigma**2)*(T-t_true(t)))/sigma/np.sqrt(T-t_true(t))#use tau here
d2=lambda a,t: (a+(r-1/2*sigma**2)*(T-t_true(t)))/sigma/np.sqrt(T-t_true(t))

u_true = lambda x, t:  (np.exp(x)*norm.cdf(d1(x,t))-np.exp(-r*(T-t_true(t)))*norm.cdf(d2(x,t)))/np.exp(alpha*x+beta*t)
g_0=u_true(-2,t)
g_1=u_true(0.5,t)

# Build solving matrix
e = numpy.ones(m) * delta_t / (2.0 * delta_x**2)
B = sparse.spdiags([e, 1.0 - 2.0 * e, e], [-1, 0, 1],  m, m).tocsr()

# Time stepping loop
for n in range(len(t)-1):
    # Construct right-hand side
    b = B.dot(U[n, :])
    b[0] += delta_t / (2.0 * delta_x**2) * (g_0[n])
    b[-1] += delta_t / (2.0 * delta_x**2) * (g_1[n])
    
    # Solve system
    U[n+1, :] = b

x_fine = numpy.linspace(-2, 0.5, 25)

y=numpy.empty(len(t))
for i in range(len(t)):
    y[i]=(U[i,:]-u_true(x,t[i])).dot(U[i,:]-u_true(x,t[i]))/len(x)
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
axes.plot(t,y)
axes.set_xlabel("tau")
axes.set_ylabel("error")
axes.set_title("errors of the solution")

# Plot a few solutions
colors = ['k', 'r', 'b', 'g', 'c']
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
for (i, n) in enumerate((0, 1, 2, 3, 4)):
    axes.plot(x, U[n, :], colors[n], label='t=%s' % numpy.round(T-2*t[n]/sigma**2, 4))
    axes.plot(x_fine,u_true(x_fine,t[n]),"o%s"%colors[n])
    axes.set_xlabel("x")
    axes.set_ylabel("u(x,t)")
    axes.set_title("Solution to Black Scholes PDE using Explicit Method")
    axes.set_xlim([-2,0.6])
    axes.set_ylim([0,2])
    axes.legend()
plt.show()

import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from scipy import stats
from scipy.stats import norm


# Compute Heat Equation Solution using Implicit Method
# Spatial discretization
m = 100
x = numpy.linspace(-2, 0.5, m)
delta_x = 2.5 / (m - 1.0)

# Time discretization - Choose \Delta t based on accuracy constraints
C = 0.05
delta_t = C * delta_x
t = numpy.arange(0.0, 0.06, delta_t)
N = len(t)
r=0.1
sigma=0.2
T=3
K=100
k=2*r/sigma**2
alpha=(1-k)/2
beta=-(k+1)**2/4
# Solution array
U = numpy.empty((N + 1, m))

#initial condition
for i in range(m):
    if x[i]>0:
        U[0,i]=(np.exp(x[i])-1)/np.exp(alpha*x[i])
    else:
        U[0,i]=0

#boundary condition
t_true=lambda t: T-2*t/sigma**2
d1=lambda a,t: (a+(r+1/2*sigma**2)*(T-t_true(t)))/sigma/np.sqrt(T-t_true(t))#use tau here
d2=lambda a,t: (a+(r-1/2*sigma**2)*(T-t_true(t)))/sigma/np.sqrt(T-t_true(t))

u_true = lambda x, t:  (np.exp(x)*norm.cdf(d1(x,t))-np.exp(-r*(T-t_true(t)))*norm.cdf(d2(x,t)))/np.exp(alpha*x+beta*t)
g_0=u_true(-2,t)
g_1=u_true(0.5,t)

# Build solving matrix
e = numpy.ones(m) * delta_t / (2.0 * delta_x**2)
A = sparse.spdiags([-e, 1.0 + 2.0 * e, -e], [-1, 0, 1], m, m).tocsr()

# Time stepping loop
for n in range(len(t)-1):
    # Construct right-hand side
    b = A.dot(U[n, :])
    b[0] -= delta_t / (2.0 * delta_x**2) * (g_0[n])
    b[-1] -= delta_t / (2.0 * delta_x**2) * (g_1[n])
    
    # Solve system
    U[n+1, :] = b

x_fine = numpy.linspace(-2, 0.5, 25)

y=numpy.empty(len(t))
for i in range(len(t)):
    y[i]=(U[i,:]-u_true(x,t[i])).dot(U[i,:]-u_true(x,t[i]))/len(x)
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
axes.plot(t,y)
axes.set_xlabel("tau")
axes.set_ylabel("error")
axes.set_title("errors of the solution")
colors = ['k', 'r', 'b', 'g', 'c']
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
for (i, n) in enumerate((0, 1, 2, 3, 4)):
    axes.plot(x, U[n, :], colors[n], label='t=%s' % numpy.round(T-2*t[n]/sigma**2, 4))
    axes.plot(x_fine,u_true(x_fine,t[n]),"o%s"%colors[n])
    axes.set_xlabel("x")
    axes.set_ylabel("u(x,t)")
    axes.set_title("Solution to Black Scholes PDE using Implicit Method")
    axes.set_xlim([-2,0.6])
    axes.set_ylim([0,2])
    axes.legend()
plt.show()

import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from scipy import stats
from scipy.stats import norm


# Compute Heat Equation Solution using Crank-Nicholson
# Spatial discretization
m = 100
x = numpy.linspace(-2, 0.5, m)
delta_x = 2.5 / (m - 1.0)

# Time discretization - Choose \Delta t based on accuracy constraints
C = 0.05
delta_t = C * delta_x
t = numpy.arange(0.0, 0.06, delta_t)
N = len(t)
r=0.1
sigma=0.2
T=3
K=100
k=2*r/sigma**2
alpha=(1-k)/2
beta=-(k+1)**2/4
# Solution array
U = numpy.empty((N + 1, m))

#initial condition
for i in range(m):
    if x[i]>0:
        U[0,i]=(np.exp(x[i])-1)/np.exp(alpha*x[i])
    else:
        U[0,i]=0

#boundary condition
t_true=lambda t: T-2*t/sigma**2
d1=lambda a,t: (a+(r+1/2*sigma**2)*(T-t_true(t)))/sigma/np.sqrt(T-t_true(t))#use tau here
d2=lambda a,t: (a+(r-1/2*sigma**2)*(T-t_true(t)))/sigma/np.sqrt(T-t_true(t))

u_true = lambda x, t:  (np.exp(x)*norm.cdf(d1(x,t))-np.exp(-r*(T-t_true(t)))*norm.cdf(d2(x,t)))/np.exp(alpha*x+beta*t)
g_0=u_true(-2,t)
g_1=u_true(0.5,t)

# Build solving matrix
e = numpy.ones(m) * delta_t / (2.0 * delta_x**2)
A = sparse.spdiags([-e, 1.0 + 2.0 * e, -e], [-1, 0, 1], m, m).tocsr()
# Build matrix for the right hand side computation
# Note that we also have to deal with boundary conditions in the actual loop
# since they could be time dependent
B = sparse.spdiags([e, 1.0 - 2.0 * e, e], [-1, 0, 1],  m, m).tocsr()

# Time stepping loop
for n in range(len(t)-1):
    # Construct right-hand side
    b = B.dot(U[n, :])
    b[0] += delta_t / (2.0 * delta_x**2) * (g_0[n] + g_0[n+1])
    b[-1] += delta_t / (2.0 * delta_x**2) * (g_1[n] + g_1[n+1])
    
    # Solve system
    U[n+1, :] = linalg.spsolve(A, b)

x_fine = numpy.linspace(-2, 0.5, 25)
y=numpy.empty(len(t))
for i in range(len(t)):
    y[i]=(U[i,:]-u_true(x,t[i])).dot(U[i,:]-u_true(x,t[i]))/len(x)
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
axes.plot(t,y)
axes.set_xlabel("tau")
axes.set_ylabel("error")
axes.set_title("errors of the solution")
# Plot a few solutions
colors = ['k', 'r', 'b', 'g', 'c']
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
for (i, n) in enumerate((0, 1, 2, 3, 4)):
    axes.plot(x, U[n, :], colors[n], label='t=%s' % numpy.round(T-2*t[n]/sigma**2, 4))
    axes.plot(x_fine,u_true(x_fine,t[n]),"o%s"%colors[n])
    axes.set_xlabel("x")
    axes.set_ylabel("u(x,t)")
    axes.set_title("Solution to Black Scholes PDE using CN")
    axes.set_xlim([-2,0.6])
    axes.set_ylim([0,2])
    axes.legend()
plt.show()


#compute relative error of these three methods with changing of volatility
def solve_bs_error_cn(sigma):
# Compute Heat Equation Solution using Crank-Nicholson
# Spatial discretization
    m = 100
    x = numpy.linspace(-2, 0.5, m)
    delta_x = 2.5 / (m - 1.0)

# Time discretization - Choose \Delta t based on accuracy constraints
    C = 0.05
    delta_t = C * delta_x
    t = numpy.arange(0.0, 0.06, delta_t)
    N = len(t)
    r=0.1
    T=3
    K=100
    k=2*r/sigma**2
    alpha=(1-k)/2
    beta=-(k+1)**2/4
    # Solution array
    U = numpy.empty((N + 1, m))

    #initial condition
    for i in range(m):
        if x[i]>0:
            U[0,i]=(np.exp(x[i])-1)/np.exp(alpha*x[i])
        else:
            U[0,i]=0

#boundary condition
    t_true=lambda t: T-2*t/sigma**2
    d1=lambda a,t: (a+(r+1/2*sigma**2)*(T-t_true(t)))/sigma/np.sqrt(T-t_true(t))#use tau here
    d2=lambda a,t: (a+(r-1/2*sigma**2)*(T-t_true(t)))/sigma/np.sqrt(T-t_true(t))

    u_true = lambda x, t:  (np.exp(x)*norm.cdf(d1(x,t))-np.exp(-r*(T-t_true(t)))*norm.cdf(d2(x,t)))/np.exp(alpha*x+beta*t)
    g_0=u_true(-2,t)
    g_1=u_true(0.5,t)

# Build solving matrix
    e = numpy.ones(m) * delta_t / (2.0 * delta_x**2)
    A = sparse.spdiags([-e, 1.0 + 2.0 * e, -e], [-1, 0, 1], m, m).tocsr()
# Build matrix for the right hand side computation
# Note that we also have to deal with boundary conditions in the actual loop
# since they could be time dependent
    B = sparse.spdiags([e, 1.0 - 2.0 * e, e], [-1, 0, 1],  m, m).tocsr()

# Time stepping loop
    for n in range(len(t)-1):
    # Construct right-hand side
        b = B.dot(U[n, :])
        b[0] += delta_t / (2.0 * delta_x**2) * (g_0[n] + g_0[n+1])
        b[-1] += delta_t / (2.0 * delta_x**2) * (g_1[n] + g_1[n+1])
    
    # Solve system
        U[n+1, :] = linalg.spsolve(A, b)
    
    return(U[-25,:]-u_true(x,t[-25])).dot(U[-25,:]-u_true(x,t[-25]))/len(x)

def solve_bs_error_explicit(sigma):
# Compute Heat Equation Solution using Crank-Nicholson
# Spatial discretization
    m = 100
    x = numpy.linspace(-2, 0.5, m)
    delta_x = 2.5 / (m - 1.0)

# Time discretization - Choose \Delta t based on accuracy constraints
    C = 0.05
    delta_t = C * delta_x
    t = numpy.arange(0.0, 0.06, delta_t)
    N = len(t)
    r=0.1
    T=3
    K=100
    k=2*r/sigma**2
    alpha=(1-k)/2
    beta=-(k+1)**2/4
    # Solution array
    U = numpy.empty((N + 1, m))

    #initial condition
    for i in range(m):
        if x[i]>0:
            U[0,i]=(np.exp(x[i])-1)/np.exp(alpha*x[i])
        else:
            U[0,i]=0

#boundary condition
    t_true=lambda t: T-2*t/sigma**2
    d1=lambda a,t: (a+(r+1/2*sigma**2)*(T-t_true(t)))/sigma/np.sqrt(T-t_true(t))#use tau here
    d2=lambda a,t: (a+(r-1/2*sigma**2)*(T-t_true(t)))/sigma/np.sqrt(T-t_true(t))

    u_true = lambda x, t:  (np.exp(x)*norm.cdf(d1(x,t))-np.exp(-r*(T-t_true(t)))*norm.cdf(d2(x,t)))/np.exp(alpha*x+beta*t)
    g_0=u_true(-2,t)
    g_1=u_true(0.5,t)

# Build solving matrix
    e = numpy.ones(m) * delta_t / (2.0 * delta_x**2)
    B = sparse.spdiags([e, 1.0 - 2.0 * e, e], [-1, 0, 1],  m, m).tocsr()

# Time stepping loop
    for n in range(len(t)-1):
    # Construct right-hand side
        b = B.dot(U[n, :])
        b[0] += delta_t / (2.0 * delta_x**2) * (g_0[n])
        b[-1] += delta_t / (2.0 * delta_x**2) * (g_1[n])
    
    # Solve system
        U[n+1, :] = b
    
    return(U[-25,:]-u_true(x,t[-25])).dot(U[-25,:]-u_true(x,t[-25]))/len(x)

def solve_bs_error_implicit(sigma):
# Compute Heat Equation Solution using Crank-Nicholson
# Spatial discretization
    m = 100
    x = numpy.linspace(-2, 0.5, m)
    delta_x = 2.5 / (m - 1.0)

# Time discretization - Choose \Delta t based on accuracy constraints
    C = 0.05
    delta_t = C * delta_x
    t = numpy.arange(0.0, 0.06, delta_t)
    N = len(t)
    r=0.1
    T=3
    K=100
    k=2*r/sigma**2
    alpha=(1-k)/2
    beta=-(k+1)**2/4
    # Solution array
    U = numpy.empty((N + 1, m))

    #initial condition
    for i in range(m):
        if x[i]>0:
            U[0,i]=(np.exp(x[i])-1)/np.exp(alpha*x[i])
        else:
            U[0,i]=0

#boundary condition
    t_true=lambda t: T-2*t/sigma**2
    d1=lambda a,t: (a+(r+1/2*sigma**2)*(T-t_true(t)))/sigma/np.sqrt(T-t_true(t))#use tau here
    d2=lambda a,t: (a+(r-1/2*sigma**2)*(T-t_true(t)))/sigma/np.sqrt(T-t_true(t))

    u_true = lambda x, t:  (np.exp(x)*norm.cdf(d1(x,t))-np.exp(-r*(T-t_true(t)))*norm.cdf(d2(x,t)))/np.exp(alpha*x+beta*t)
    g_0=u_true(-2,t)
    g_1=u_true(0.5,t)

# Build solving matrix
    e = numpy.ones(m) * delta_t / (2.0 * delta_x**2)
    A = sparse.spdiags([-e, 1.0 + 2.0 * e, -e], [-1, 0, 1], m, m).tocsr()

# Time stepping loop
    for n in range(len(t)-1):
    # Construct right-hand side
        b = A.dot(U[n, :])
        b[0] -= delta_t / (2.0 * delta_x**2) * (g_0[n])
        b[-1] -= delta_t / (2.0 * delta_x**2) * (g_1[n])
    
    # Solve system
        U[n+1, :] = b
    
    return(U[-25,:]-u_true(x,t[-25])).dot(U[-25,:]-u_true(x,t[-25]))/(len(x))

sigma=[0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
error=numpy.empty((3,len(sigma)))
for i in range(len(sigma)):
    error[0,i]=numpy.log(solve_bs_error_cn(sigma[i]))
    error[1,i]=numpy.log(solve_bs_error_explicit(sigma[i]))
    error[2,i]=numpy.log(solve_bs_error_implicit(sigma[i]))
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
axes.plot(sigma,error[0,:],'r')
axes.plot(sigma,error[1,:],'b')
axes.plot(sigma,error[2,:],'g')
axes.set_xlabel("sigma")
axes.set_ylabel("error")
axes.set_title("errors of the solution")
