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
blackscholes(S, K, r, T, sigma, option_type)



'''有限差分法'''
(Ns, Nt) = (200, 200)

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
        self.Xmin = 0
        self.Xmax = np.log(4*np.maximum(self.S, self.K))
        self.Nx = int(Nx)
        self.Nt = int(Nt)
        self.dX = (self.Xmax-self.Xmin)/self.Nx * 1.0
        self.h = self.dX
        self.dt = T/Nt*1.0
        self.tau = self.dt
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
        self.a = self.r / (2 * self.h) - ((self.h + 2) / (4 * self.h**2)) * self.sigma**2
        self.b = self.r + (self.sigma/self.h)**2
        self.c = -(((2 - self.h) / (4 * self.h**2)) * self.sigma**2 + self.r / (2 * self.h))
        
    def _solve_(self):
        pass
    
    def _interpolate_(self):
        tck = spi.splrep( self.Xvec, self.grid[0, :], k=3 )
        return np.e ** spi.splev( self.X, tck )
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
            # if j < 5:
            print(j)
            print(_tmp)
            # print(pd.DataFrame(self.grid))
            input()


class FullyImplicitEu(FiniteDifference):

    def _set_matrix_(self):
        self.B = sp.diags([[self.a]*(self.Nx-2), [self.b]*(self.Nx-1), [self.c]*(self.Nx-2)], [-1, 0, 1],  format='csc').toarray()
        self.I = sp.eye(self.Nx-1).toarray()
        self.M = self.I - self.tau * self.B
    
    def _solve_(self):  
        _, M_lower, M_upper = sla.lu(self.M)

        for j in reversed(np.arange(self.Nt)):     
            _tmp = self.M.dot(self.grid[j+1, 1:-1])
            _tmp[0] += self.tau * self.a * self.grid[j, 0]
            _tmp[-1] += self.tau * self.c * self.grid[j, -1]
            _tmp = sla.solve_triangular( M_lower, _tmp, lower=True )
            self.grid[j, 1:-1] = sla.solve_triangular( M_upper, _tmp, lower=False )




option = FullyExplicitEu(S, K, r, T, sigma, option_type, Nx=50, Nt=100)


option._set_terminal_condition_()
option._set_boundary_condition_()
# option._set_coefficient__()
# option.a
# option.b
# option.c
# option._set_matrix_()
# pd.DataFrame(option.B)
# pd.DataFrame(option.I)
# pd.DataFrame(option.M)
# option._solve_()
# pd.DataFrame(option.grid)

option.Xmax
option.Xmin

np.log(50)

pd.DataFrame(option.grid)

option.a = option.r / (2 * option.h) - ((option.h + 2) / (4 * option.h**2)) * option.sigma**2
option.b = option.r + (option.sigma/option.h)**2
option.c = -(((2 - option.h) / (4 * option.h**2)) * option.sigma**2 + option.r / (2 * option.h))


option.B = sp.diags([[option.a]*(option.Nx-2), [option.b]*(option.Nx-1), [option.c]*(option.Nx-2)], [-1, 0, 1],  format='csc').toarray()
option.I = sp.eye(option.Nx-1).toarray()
option.M = option.I + option.tau * option.B


for j in reversed(np.arange(option.Nt)):

    _tmp = option.M.dot(option.grid[j+1, 1:-1])
    _tmp[0] += option.tau * option.a * option.grid[j+1, 0]
    _tmp[-1] += option.tau * option.c * option.grid[j+1, -1]
    option.grid[j, 1:-1] = _tmp
    # if j < 5:
    print(j)
    print(_tmp)
    # print(pd.DataFrame(self.grid))
    input()