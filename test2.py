import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.interpolate as spi
import scipy.sparse as sp
import scipy.linalg as sla
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(1031)
dt_hex = '#2B4750'    # dark teal,  RGB = 43,71,80
r_hex = '#DC2624'     # red,        RGB = 220,38,36
g_hex = '#649E7D'     # green,      RGB = 100,158,125
tl_hex = '#45A0A2'    # teal,       RGB = 69,160,162
tn_hex = '#C89F91'    # tan,        RGB = 200,159,145


# /////////////////////////////////////////// 理论 /////////////////////////////////////////////////

def plot_grid_withTCBC( S_arr, t_arr ):
    fig = plt.figure(figsize=(6,6), dpi=200)
    ax = fig.add_subplot(1,1,1)
    
    t_mat, S_mat = np.meshgrid(t_arr, S_arr)
    ax.scatter(S_mat, t_mat, facecolor='grey', s=60, edgecolors=None )

    # 设定标题和纵轴的标注
    ax.set_ylabel('$0$-----------------------$t$--------------------->$T$', size=15, alpha=0.5)
    ax.set_title('$S_{\min}$-----------------$S$-------------------->$S_{\max}$', size=15, alpha=0.5)

    # 去除横轴和纵轴的刻度，去除边框
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())

    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 给边界条件和终止条件的点上颜色，并标注
    ax.scatter(S_mat[[0, -1]], t_mat[[0, 1]], facecolor=dt_hex, s=60, edgecolors=None)
    ax.scatter(S_mat[:, -1], t_mat[:, -1], facecolor=r_hex, s=60, edgecolors=None)

    dt = t_arr[1]-t_arr[0]
    dS = S_arr[1]-S_arr[0]
    ax.text(S_arr.mean(), t_arr.max() + T/30, '终值条件', ha='center', color=r_hex)
    ax.text(S_arr.min() - S/12.5, t_arr.mean(), '边界条件', rotation='vertical', color=dt_hex)
    ax.text(S_arr.max() + S/20, t_arr.mean(), '边界条件', rotation='vertical', color=dt_hex)
    
    return ax
    

(S, K, r, q, T, sigma, omega, option_type) = (40, 50, 0.3, 0, 1, 0.5, 1, 'call')
(Smin, Smax, Tmin, Tmax, Ns, Nt) = (0, np.maximum(S,K), 0, T, 4, 4)
S_arr, t_arr = np.linspace(Smin, Smax, Ns+1), np.linspace(Tmin, Tmax, Nt+1)

# 网格 ///////////////////////////////////////////////////////////////////////////////////////////
ax = plot_grid_withTCBC( S_arr, t_arr )
plt.savefig('paper/1_grid.png')
plt.show()

# 完全显式格式 ///////////////////////////////////////////////////////////////////////////////////////////
ax = plot_grid_withTCBC( S_arr, t_arr )

size=10
dt = t_arr[1]-t_arr[0]
dS = S_arr[1]-S_arr[0]

(i,j) = (2,2)
(ti,Sj) = (t_arr[i], S_arr[j])

ax.plot( [Sj, Sj-dS], [ti-dt, ti], color='k', alpha=0.2, zorder=0 )
ax.plot( [Sj, Sj], [ti-dt, ti], color='k', alpha=0.2, zorder=0 )
ax.plot( [Sj, Sj+dS], [ti-dt, ti], color='k', alpha=0.2, zorder=0 )


ax.scatter(Sj, ti, facecolor='b', s=60, edgecolors=None, zorder=1 )
ax.scatter(Sj-dS, ti, facecolor='b', s=60, edgecolors=None, zorder=1 )
ax.scatter(Sj+dS, ti, facecolor='b', s=60, edgecolors=None, zorder=1 )
ax.scatter(Sj, ti-dt, facecolor=g_hex, s=60, edgecolors=None, zorder=1 )

ax.text( Sj, ti+0.3*dt, '完全显式格式', ha='center', size=14 )
ax.text( Sj, ti+0.1*dt, '$u(S_j, t_i)$', ha='right', color='b', size=size )
ax.text( Sj-dS, ti+0.1*dt, '$u(S_{j-1}, t_i)$', ha='right', color='b', size=size )
ax.text( Sj+dS, ti+0.1*dt, '$u(S_{j+1}, t_i)$', ha='right', color='b', size=size )
ax.text( Sj, ti-1.2*dt, '$u(S_j, t_{i-1})$', ha='right', color='b', size=size )

plt.savefig('paper/2_ExpEu.png')
plt.show()


# 完全隐式格式 ///////////////////////////////////////////////////////////////////////////////////////////
ax = plot_grid_withTCBC( S_arr, t_arr )

size=10
dt = t_arr[1]-t_arr[0]
dS = S_arr[1]-S_arr[0]

(i,j) = (2,2)
(ti,Sj) = (t_arr[i], S_arr[j])

ax.plot( [Sj, Sj-dS], [ti+dt, ti], color='k', alpha=0.2, zorder=0 )
ax.plot( [Sj, Sj], [ti+dt, ti], color='k', alpha=0.2, zorder=0 )
ax.plot( [Sj, Sj+dS], [ti+dt, ti], color='k', alpha=0.2, zorder=0 )


ax.scatter(Sj, ti, facecolor='b', s=60, edgecolors=None, zorder=1 )
ax.scatter(Sj-dS, ti, facecolor='b', s=60, edgecolors=None, zorder=1 )
ax.scatter(Sj+dS, ti, facecolor='b', s=60, edgecolors=None, zorder=1 )
ax.scatter(Sj, ti+dt, facecolor=g_hex, s=60, edgecolors=None, zorder=1 )

ax.text( Sj, ti-0.3*dt, '完全隐式格式', ha='center', size=14 )
ax.text( Sj, ti+0.1*dt, '$u(S_j, t_i)$', ha='right', color='b', size=size )
ax.text( Sj-dS, ti+0.1*dt, '$u(S_{j-1}, t_i)$', ha='right', color='b', size=size )
ax.text( Sj+dS, ti+0.1*dt, '$u(S_{j+1}, t_i)$', ha='right', color='b', size=size )
ax.text( Sj, ti+1.1*dt, '$u(S_j, t_{i+1})$', ha='right', color='b', size=size )

plt.savefig('paper/3_ImpEu.png')
plt.show()


# C-N 格式 ///////////////////////////////////////////////////////////////////////////////////////////
ax = plot_grid_withTCBC( S_arr, t_arr )

size=10
dt = t_arr[1]-t_arr[0]
dS = S_arr[1]-S_arr[0]

(i,j) = (2,2)
(ti,Sj) = (t_arr[i], S_arr[j])

ax.plot( [Sj, Sj], [ti, ti+dt], color='k', alpha=0.2, zorder=0 )
ax.plot( [Sj-dS, Sj+dS], [ti, ti+dt], color='k', alpha=0.2, zorder=0 )
ax.plot( [Sj+dS, Sj-dS], [ti, ti+dt], color='k', alpha=0.2, zorder=0 )

ax.scatter(Sj, ti+0.5*dt, facecolor=g_hex, s=60, edgecolors=None, zorder=1 )

ax.scatter(Sj-dS, ti, facecolor='b', s=60, edgecolors=None, zorder=1 )
ax.scatter(Sj+dS, ti, facecolor='b', s=60, edgecolors=None, zorder=1 )
ax.scatter(Sj, ti, facecolor='b', s=60, edgecolors=None, zorder=1 )
ax.scatter(Sj-dS, ti+dt, facecolor='b', s=60, edgecolors=None, zorder=1 )
ax.scatter(Sj+dS, ti+dt, facecolor='b', s=60, edgecolors=None, zorder=1 )
ax.scatter(Sj, ti+dt, facecolor='b', s=60, edgecolors=None, zorder=1 )

ax.text( Sj, ti-0.5*dt, 'C-N 格式', ha='center', size=14 )

ax.text( Sj-0.45*dS, ti+0.5*dt, '$u(S_{j}, t_{i+0.5})$', ha='center', color=g_hex, size=10, zorder=1 )

ax.text( Sj-dS, ti-0.2*dt, '$u(S_{j-1}, t_i)$', ha='right', color='b', size=size )
ax.text( Sj, ti-0.2*dt, '$u(S_j, t_i)$', ha='right', color='b', size=size )
ax.text( Sj+dS, ti-0.2*dt, '$u(S_{j+1}, t_i)$', ha='right', color='b', size=size )

ax.text( Sj-dS, ti+1.1*dt, '$u(S_{j-1}, t_{i+1})$', ha='right', color='b', size=size )
ax.text( Sj, ti+1.1*dt, '$u(S_j, t_{i+1})$', ha='right', color='b', size=size )
ax.text( Sj+dS, ti+1.1*dt, '$u(S_{j+1}, t_{i+1})$', ha='right', color='b', size=size )

plt.savefig('paper/4_CNEu.png')
plt.show()






# /////////////////////////////////////////// 应用 /////////////////////////////////////////////////


def blackscholes( S0=100, K=100, r=0.01, q=0.01, T=1, sigma=0.2, omega=1 ):
    discount = np.exp(-r*T)
    forward = S0*np.exp((r-q)*T)
    moneyness = np.log(forward/K)
    vol_sqrt_T = sigma*np.sqrt(T)
    
    d1 = moneyness / vol_sqrt_T + 0.5*vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    
    V = omega * discount * (forward*norm.cdf(omega*d1) - K*norm.cdf(omega*d2))
    return V

class OptionPricingMethod():
    
    def __init__(self, S, K, r, q, T, sigma, option_type):
        self.S = S
        self.K = K
        self.r = r
        self.q = q
        self.T = T
        self.sigma = sigma
        self.option_type = option_type
        self.is_call = (option_type[0].lower()=='c')
        self.omega = 1 if self.is_call else -1

class FiniteDifference(OptionPricingMethod):
    
    def __init__(self, S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt):
        super().__init__(S, K, r, q, T, sigma, option_type)
        self.Smin = Smin
        self.Smax = Smax
        self.Ns = int(Ns)
        self.Nt = int(Nt)
        self.dS = (Smax-Smin)/Ns * 1.0
        self.dt = T/Nt*1.0
        self.Svec = np.linspace(Smin, Smax, self.Ns+1)
        self.Tvec = np.linspace(0, T, self.Nt+1)
        self.grid = np.zeros(shape=(self.Ns+1, self.Nt+1))
        
    def _set_terminal_condition_(self):
        self.grid[:, -1] = np.maximum(self.omega*(self.Svec - self.K), 0)
    
    def _set_boundary_condition_(self):
        tau = self.Tvec[-1] - self.Tvec;     
        DFq = np.exp(-q*tau)
        DFr = np.exp(-r*tau);

        self.grid[0,  :] = np.maximum(self.omega*(self.Svec[0]*DFq - self.K*DFr), 0)
        self.grid[-1, :] = np.maximum(self.omega*(self.Svec[-1]*DFq - self.K*DFr), 0)        
        
    def _set_coefficient__(self):
        drift = (self.r-self.q)*self.Svec[1:-1]/self.dS
        diffusion_square = (self.sigma*self.Svec[1:-1]/self.dS)**2
        
        self.l = 0.5*(diffusion_square - drift)
        self.c = -diffusion_square - self.r
        self.u = 0.5*(diffusion_square + drift)
        
    def _solve_(self):
        pass
    
    def _interpolate_(self):
        tck = spi.splrep( self.Svec, self.grid[:,0], k=3 )
        return spi.splev( self.S, tck )
        #return np.interp(self.S, self.Svec, self.grid[:,0])
    
    def price(self):
        self._set_terminal_condition_()
        self._set_boundary_condition_()
        self._set_coefficient__()
        self._set_matrix_()
        self._solve_()
        return self._interpolate_()

class FullyExplicitEu(FiniteDifference):
    
    def _set_matrix_(self):
        self.A = sp.diags([self.l[1:], self.c, self.u[:-1]], [-1, 0, 1],  format='csc')
        self.I = sp.eye(self.Ns-1)
        self.M = self.I + self.dt*self.A
                                        
    def _solve_(self):
        for j in reversed(np.arange(self.Nt)):
            U = self.M.dot(self.grid[1:-1, j+1])
            U[0] += self.l[0]*self.dt*self.grid[0, j+1] 
            U[-1] += self.u[-1]*self.dt*self.grid[-1, j+1] 
            self.grid[1:-1, j] = U

class FullyImplicitEu(FiniteDifference):

    def _set_matrix_(self):
        self.A = sp.diags([self.l[1:], self.c, self.u[:-1]], [-1, 0, 1],  format='csc')
        self.I = sp.eye(self.Ns-1)
        self.M = self.I - self.dt*self.A
    
    def _solve_(self):  
        _, M_lower, M_upper = sla.lu(self.M.toarray())

        for j in reversed(np.arange(self.Nt)):      
            U = self.grid[1:-1, j+1].copy()
            U[0] += self.l[0]*self.dt*self.grid[0, j] 
            U[-1] += self.u[-1]*self.dt*self.grid[-1, j] 
            Ux = sla.solve_triangular( M_lower, U, lower=True )
            self.grid[1:-1, j] = sla.solve_triangular( M_upper, Ux, lower=False )

class CrankNicolsonEu(FiniteDifference):

    theta = 0.5
    
    def _set_matrix_(self):
        self.A = sp.diags([self.l[1:], self.c, self.u[:-1]], [-1, 0, 1],  format='csc')
        self.I = sp.eye(self.Ns-1)
        self.M1 = self.I + (1-self.theta)*self.dt*self.A
        self.M2 = self.I - self.theta*self.dt*self.A
    
    def _solve_(self):           
        _, M_lower, M_upper = sla.lu(self.M2.toarray())        
        for j in reversed(np.arange(self.Nt)):
            
            U = self.M1.dot(self.grid[1:-1, j+1])
            
            U[0] += self.theta*self.l[0]*self.dt*self.grid[0, j] \
                 + (1-self.theta)*self.l[0]*self.dt*self.grid[0, j+1] 
            U[-1] += self.theta*self.u[-1]*self.dt*self.grid[-1, j] \
                  + (1-self.theta)*self.u[-1]*self.dt*self.grid[-1, j+1] 
            
            Ux = sla.solve_triangular( M_lower, U, lower=True )
            self.grid[1:-1, j] = sla.solve_triangular( M_upper, Ux, lower=False )


(Smin, Smax) = (0, 4*np.maximum(S,K))

blackscholes(S, K, r, q, T, sigma, omega)

optionEX = FullyExplicitEu(S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt)
optionIM = FullyImplicitEu(S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt)
optionCN = CrankNicolsonEu(S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt)

# optionEX.price()
# optionIM.price()
# optionCN.price()



def PDE_visualizer( option, step=0 ):
    V0 = option.price()
    (Tvec, Svec, Vmat) = (option.Tvec, option.Svec, option.grid)
    
    (S0, K, r, q, omega) = (option.S, option.K, option.r, option.q, option.omega)
    
    fig = plt.figure(figsize=(6,4), dpi=100)
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    
    t, S = np.meshgrid(Tvec, Svec)
    V = np.zeros(Vmat.shape)
        
    ax.scatter( t, S, V, color='k', alpha=0.2 )
    
    # 设定横轴和纵轴的范围
    ax.set_xlim3d(Tvec[0], Tvec[-1])
    ax.set_ylim3d(Svec[0], Svec[-1])
    ax.set_zlim3d(0, np.max(Vmat[:,-1]))

    # 设定标题和纵轴的标注
    ax.set_xlabel('时间', color=dt_hex)
    ax.set_ylabel('标的', color=r_hex)
    ax.set_zlabel('期权', color=g_hex)

    # 去除横轴和纵轴的刻度  
    ax.set_xticks(Tvec)
    ax.set_xticklabels(['$t_0$', '$t_1$', '$t_2$', '$t_3$', '$t_4$'] )
    ax.set_yticks(Svec)
    ax.set_yticklabels(['$S_0$', '$S_1$', '$S_2$', '$S_3$', '$S_4$'] )
    ax.set_zticks([])
    
    ax.grid(False)
    
    # 给边界条件和终止条件的点上颜色，并标注
    DFq = np.exp(-q*(Tvec[-1]-Tvec))
    DFr = np.exp(-r*(Tvec[-1]-Tvec))
    
    LB = np.maximum(omega*(Svec[0]*DFq - K*DFr), 0)
    UB = np.maximum(omega*(Svec[-1]*DFq - K*DFr), 0)
    payoff = np.maximum(omega*(Svec-K),0)

    ax.plot( Tvec, S[0,:], LB, color=dt_hex )
    ax.scatter( Tvec, S[0,:], LB, color=dt_hex )
    
    ax.plot( Tvec, S[-1,:], UB, color=dt_hex )
    ax.scatter( Tvec, S[-1,:], UB, color=dt_hex )
        
    for i in reversed(np.arange(step,len(Tvec))):
        c = r_hex if i == len(Tvec)-1 else g_hex
        ax.plot( t[:,i], Svec, Vmat[:,i], color=c )
        ax.scatter( t[:,i], Svec, Vmat[:,i], color=c )
    
    if step == 0:
        ax.plot( [0], [S0], [V0], markerfacecolor='b', markeredgecolor='b', marker='o', markersize=8 )

# 期权价格图(迭代4次)
option = optionEX
for i in range(5):
    PDE_visualizer( option, 4-i )
    plt.savefig(f'paper/{i+5}_option_price_surface_4iter_step{i}.png')
    plt.show()





def plot_surface( func, S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt ):
    fig = plt.figure(figsize=(8,4), dpi=240)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    option = func( S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt )    
    V = option.price()
    
    Tgrid, Sgrid = np.meshgrid(option.Tvec, option.Svec)
    ax.plot_surface( Tgrid, Sgrid, option.grid, cmap='coolwarm', antialiased=False, alpha=0.5 )
    ax.plot( [0], [S], [V], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5 )
    ax.set_xlabel('时间')
    ax.set_ylabel('股价')
    ax.set_zlabel('期权价格')
    ax.view_init(elev=30, azim=160)

    plt.show()

plot_surface( FullyImplicitEu, S, K, r, q, T, sigma, option_type, Smin, Smax, 100, 100)
plot_surface( FullyExplicitEu, S, K, r, q, T, sigma, option_type, Smin, Smax, 100, 100)


# 检查结果是否收敛
def convergence_test( func, S, K, r, q, T, sigma, option_type, Smin, Smax, Ns_vec, Nt_vec ):
    numerical = np.array([])
    for Ns, Nt in zip(Ns_vec, Nt_vec):
        option = func( S, K, r, q, T, sigma, option_type, Smin, Smax, Ns, Nt )
        numerical = np.append(numerical, option.price())
        
    omega = 1 if option_type[0].lower()=='c' else -1
    benchmark = blackscholes(S, K, r, q, T, sigma, omega)
    error = np.abs(numerical-benchmark)
    return error, benchmark, numerical


Ns_vec = np.array([100, 200, 400, 800])
Nt_vec = np.array([200, 400, 800, 1600])
errorImp, impPrice, anaPrice = convergence_test( FullyImplicitEu, S, K, r, q, T, sigma, option_type, Smin, Smax, Ns_vec, Nt_vec )
errorCN, CNPrice, anaPrice = convergence_test( CrankNicolsonEu, S, K, r, q, T, sigma, option_type, Smin, Smax, Ns_vec, Nt_vec )



fig = plt.figure(figsize=(8,4), dpi=250)
ax = fig.gca()

ax.plot( np.log(Nt_vec), errorImp, color=dt_hex, alpha=0.5, label='隐式法' )
ax.scatter( np.log(Nt_vec), errorImp, color =dt_hex )
ax.plot( np.log(Nt_vec), errorCN, color=r_hex, alpha=0.5, label='克兰克尼克尔森法' )
ax.scatter( np.log(Nt_vec), errorCN, color =r_hex )

ax.set_xticks(np.log(Nt_vec))
ax.set_xticklabels(['ln(200)', 'ln(400)', 'ln(800)', 'ln(1600)'])
ax.set_xlabel('ln(时段数)')
ax.set_ylabel('绝对差异')
ax.set_title('有限差分数值解 Vs 解析解')
plt.legend()
plt.savefig(f'paper/14_error.png')
plt.show()