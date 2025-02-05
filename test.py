import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.interpolate as spi
import scipy.sparse as sp
import scipy.linalg as sla
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(1031)
dt_hex = '#2B4750'    # dark teal,  RGB = 43,71,80
r_hex = '#DC2624'     # red,        RGB = 220,38,36
g_hex = '#649E7D'     # green,      RGB = 100,158,125
tl_hex = '#45A0A2'    # teal,       RGB = 69,160,162
tn_hex = '#C89F91'    # tan,        RGB = 200,159,145



def solve_thoma(a, b, c, d):
    '''
    利用追赶法求解解线性方程组
    ---
    
    a1 = cn = 0, 即a[0] = c[-1] = 0

    ::
        
        A = [b1  c1  0   0   ...  0     0   ]   b = [ d1 ]
            [a2  b2  c2  0   ...  0     0   ]       [ d2 ]
            [0   a3  b3  c3  ...  0     0   ]       [ d3 ]
            [0   0   a4  b4  ...  0     0   ]       [ d4 ]
            [:   :   :   :   ...  :     :   ]       [ :  ]
            [0   0   0   0   ...  bn-1  cn-1]       [dn-1]
            [0   0   0   0   ...  an    bn  ]       [ dn ]
    '''
    aa = a.copy()
    bb = b.copy()
    cc = c.copy()
    dd = d.copy()
    n = len(dd)
    x = np.ones_like(dd, dtype=float)
    y = np.ones_like(dd, dtype=float)
    aa[0] = 0
    cc[0] = cc[0] / bb[0]
    y[0] = dd[0] / bb[0]    # update
    for i in range(1, n): # forward
        bb[i] = bb[i] - aa[i] * cc[i - 1] # store l in b and l[0]=b[0]
        y[i] = (dd[i] - aa[i] * y[i - 1]) / bb[i]
        cc[i] = cc[i] / bb[i] # store u in c and and u[0]=c[0]/b[0]

    x[n - 1] = y[n - 1] # backward
    for i in range(n - 2, -1, -1):  # update
        x[i] = y[i] - cc[i] * x[i + 1]
    return x

            # _tmp = self.grid[j+1, 1:-1]
            # _tmp[0] += self.tau * self.a * self.grid[j, 0]
            # _tmp[-1] += self.tau * self.c * self.grid[j, -1]

            # u_j = solve_thoma(
            #     a = [- self.tau * self.a]*(self.Nx-1),
            #     b = [1 - self.tau * self.b]*(self.Nx-1),
            #     c = [- self.tau * self.c]*(self.Nx-1),
            #     d = list(_tmp)
            # )
            # self.grid[j, 1:-1] = u_j



def time_costing(func):
    '''装饰器: 记录运行时间'''
    def wrapper(*args, **kwargs):
        start = datetime.datetime.now()
        x = func(*args, **kwargs)
        print('time costing:', datetime.datetime.now() - start)
        return x
    return wrapper


'''BS公式'''
def blackscholes( S0=100, K=100, r=0.01, T=1, sigma=0.2, option_type='call' ):

    if option_type == 'call':
        omega = 1
    elif option_type == 'put':
        omega = -1

    discount = np.exp(-r*T)
    forward = S0*np.exp(r*T)
    moneyness = np.log(forward/K)
    vol_sqrt_T = sigma*np.sqrt(T)
    
    d1 = moneyness / vol_sqrt_T + 0.5*vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    
    V = omega * discount * (forward*norm.cdf(omega*d1) - K*norm.cdf(omega*d2))
    return V

(S, K, r, T, sigma, option_type) = (40, 50, 0.3, 1, 0.5, 'call')



'''有限差分法'''

class OptionPricingMethod():
    
    def __init__(self, S, K, r, T, sigma, option_type):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.option_type = option_type
        self.is_call = (option_type[0].lower()=='c')
        self.omega = 1 if self.is_call else -1


class FiniteDifference(OptionPricingMethod):
    '''变量代换: tau=T-t, x=ln(S)'''
    
    def __init__(self, S, K, r, T, sigma, option_type, Nx=400, Nt=600):
        super().__init__(S, K, r, T, sigma, option_type)
        self.X = np.log(self.S)
        self.Xmin = np.log(0.1*np.maximum(self.S, self.K))
        self.Xmax = np.log(4*np.maximum(self.S, self.K))
        self.Nx = int(Nx)
        self.Nt = int(Nt)
        self.dX = (self.Xmax-self.Xmin)/self.Nx * 1.0
        self.h = self.dX
        self.dt = T/Nt*1.0
        self.tau = self.dt
        # self.Svec = np.linspace(0, 4*np.maximum(self.S, self.K), self.Nx+1)
        self.Xvec = np.linspace(self.Xmin, self.Xmax, self.Nx+1)
        self.Tvec = np.linspace(0, T, self.Nt+1)
        self.grid = np.zeros(shape=(self.Nt+1, self.Nx+1))
        
    def _set_terminal_condition_(self):
        '''终止条件 max(e^x-K,0)'''
        self.grid[-1, :] = np.maximum(self.omega*(np.e ** self.Xvec - self.K), 0)

    def _set_boundary_condition_(self):
        '''边界条件'''
        self.grid[:,  0] = np.maximum(self.omega*(np.e ** self.Xvec[0]  - self.K), 0)
        self.grid[:, -1] = np.maximum(self.omega*(np.e ** self.Xvec[-1] - self.K), 0)
        
    def _set_coefficient__(self):
        self.a = ((self.h + 2) / (4 * self.h**2)) * self.sigma**2 - self.r / (2 * self.h)
        self.b = -(self.r + (self.sigma/self.h)**2)
        self.c = ((2 - self.h) / (4 * self.h**2)) * self.sigma**2 + self.r / (2 * self.h)
        
    def _solve_(self):
        pass
    
    def _interpolate_(self):
        tck = spi.splrep( self.Xvec, self.grid[0, :], k=3 )
        return spi.splev( self.X, tck )
        #return np.interp(self.S, self.Svec, self.grid[:,0])
    
    def price(self):
        self._set_terminal_condition_()
        self._set_boundary_condition_()
        self._set_coefficient__()
        self._set_matrix_()
        self._solve_()
        return self._interpolate_()


class FullyExplicitEu(FiniteDifference):
    '''完全显式格式'''
    
    def _set_matrix_(self):
        self.B = sp.diags([[self.a]*(self.Nx-2), [self.b]*(self.Nx-1), [self.c]*(self.Nx-2)], [-1, 0, 1],  format='csc').toarray()
        self.I = sp.eye(self.Nx-1).toarray()
        self.M = self.I + self.tau * self.B
                                        
    def _solve_(self):
        for j in reversed(np.arange(self.Nt)):
            _tmp = self.M.dot(self.grid[j+1, 1:-1])
            _tmp[0] += self.tau * self.a * self.grid[j+1, 0]
            _tmp[-1] += self.tau * self.c * self.grid[j+1, -1]
            self.grid[j, 1:-1] = _tmp



class FullyImplicitEu(FiniteDifference):
    '''完全隐式格式'''

    def _set_matrix_(self):
        self.B = sp.diags([[self.a]*(self.Nx-2), [self.b]*(self.Nx-1), [self.c]*(self.Nx-2)], [-1, 0, 1],  format='csc').toarray()
        self.I = sp.eye(self.Nx-1).toarray()
        self.M = self.I - self.tau * self.B
    
    def _solve_(self):  
        P, M_lower, M_upper = sla.lu(self.M)

        for j in reversed(np.arange(self.Nt)):
            _tmp = self.grid[j+1, 1:-1]
            _tmp[0] += self.tau * self.a * self.grid[j, 0]
            _tmp[-1] += self.tau * self.c * self.grid[j, -1]
            _tmp = sla.solve_triangular( M_lower, _tmp, lower=True )
            self.grid[j, 1:-1] = sla.solve_triangular( M_upper, _tmp, lower=False )


class CrankNicolsonEu(FiniteDifference):
    '''C-N格式'''
    
    def _set_matrix_(self):
        self.B = sp.diags([[self.a]*(self.Nx-2), [self.b]*(self.Nx-1), [self.c]*(self.Nx-2)], [-1, 0, 1],  format='csc').toarray()
        self.I = sp.eye(self.Nx-1).toarray()
        self.M1 = self.I + 0.5 * self.tau * self.B
        self.M2 = self.I - 0.5 * self.tau * self.B
    
    def _solve_(self):           
        _, M_lower, M_upper = sla.lu(self.M2)        
        for j in reversed(np.arange(self.Nt)):
            
            _tmp = self.M1.dot(self.grid[j+1, 1:-1])
            _tmp[0] += 0.5 * self.tau * self.a * (self.grid[j, 0] + self.grid[j+1, 0])
            _tmp[-1] += 0.5 * self.tau * self.c * (self.grid[j, -1] + self.grid[j+1, -1])
            _tmp = sla.solve_triangular( M_lower, _tmp, lower=True )
            self.grid[j, 1:-1] = sla.solve_triangular( M_upper, _tmp, lower=False )
            
            


blackscholes(S, K, r, T, sigma, option_type)
FullyExplicitEu(S, K, r, T, sigma, option_type, Nx=40, Nt=80).price()
FullyImplicitEu(S, K, r, T, sigma, option_type, Nx=100, Nt=100).price()
CrankNicolsonEu(S, K, r, T, sigma, option_type, Nx=60, Nt=100).price()


optionEx = FullyExplicitEu(S, K, r, T, sigma, option_type, Nx=40, Nt=80)
optionEx.price()

optionIm = FullyImplicitEu(S, K, r, T, sigma, option_type, Nx=100, Nt=100)
optionIm.price()

optionCN = CrankNicolsonEu(S, K, r, T, sigma, option_type, Nx=80, Nt=100)
optionCN.price()

# for nx in [5, 10, 20, 40, 80, 160, 320]:
#     for nt in [5, 10, 20, 40, 80, 160, 320]:



t = optionCN.Tvec
X = optionCN.Xvec
V = optionCN.grid

import matplotlib.pyplot as plt
from matplotlib import cm
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(t, X, V, cmap=cm.viridis)
ax.set_xlabel('Time to Expiration')
ax.set_ylabel('Price of Underlying Asset')
ax.set_zlabel('Option Price')
ax.set_title('Explicit Scheme Solution of an Option')
plt.show()
