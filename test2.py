import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.interpolate as spi
import scipy.sparse as sp
import scipy.linalg as sla
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



(S, K, r, q, T, sigma, omega) = (40, 50, 0.3, 0, 1, 0.5, 1)
(Smin, Smax, Ns, Nt) = (0, 4*np.maximum(S,K), 4, 4)


def blackscholes( S0=100, K=100, r=0.01, q=0.01, T=1, sigma=0.2, omega=1 ):
    discount = np.exp(-r*T)
    forward = S0*np.exp((r-q)*T)
    moneyness = np.log(forward/K)
    vol_sqrt_T = sigma*np.sqrt(T)
    
    d1 = moneyness / vol_sqrt_T + 0.5*vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    
    V = omega * discount * (forward*norm.cdf(omega*d1) - K*norm.cdf(omega*d2))
    return V

blackscholes(S, K, r, q, T, sigma, omega)

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
    ax.text(t_arr.mean(), S_arr.min()-S/10, '边界条件', ha='center', color=dt_hex)
    ax.text(t_arr.mean(), S_arr.max()+S/20, '边界条件', ha='center', color=dt_hex)
    ax.text(t_arr.max()+dt/8, S_arr.mean(), '终值条件', rotation='vertical', color=r_hex)
    
    return ax
    

(Smin, Smax, Tmin, Tmax, Ns, Nt) = (0, np.maximum(S,K), 0, T, 4, 4)
S_arr, t_arr = np.linspace(Smin, Smax, Ns+1), np.linspace(Tmin, Tmax, Nt+1)

ax = plot_grid_withTCBC( S_arr, t_arr )

size=10
dt = t_arr[1]-t_arr[0]
dS = S_arr[1]-S_arr[0]

(i,j) = (2,2)
(ti,Sj) = (t_arr[i], S_arr[j])
ax.plot( [ti, ti+dt], [Sj, Sj-dS], color='k', alpha=0.2, zorder=0 )
ax.plot( [ti, ti+dt], [Sj, Sj], color='k', alpha=0.2, zorder=0 )
ax.plot( [ti, ti+dt], [Sj, Sj+dS], color='k', alpha=0.2, zorder=0 )

ax.scatter(ti+dt, Sj, facecolor=g_hex, s=60, edgecolors=None, zorder=1 )
ax.scatter(ti+dt, Sj-dS, facecolor='b', s=60, edgecolors=None, zorder=1 )
ax.scatter(ti+dt, Sj+dS, facecolor='b', s=60, edgecolors=None, zorder=1 )
ax.scatter(ti, Sj, facecolor='b', s=60, edgecolors=None, zorder=1 )

ax.text( ti+dt, Sj-1.3*dS, '完全显式', ha='center', size=14 )
ax.text( ti-0.1*dt, Sj, '$V(t_{i-1}, S_j)$', ha='right', color='b', size=size )
ax.text( ti+1.1*dt, Sj-dS, '$V(t_{i}, S_{j-1})$', ha='left', color='b',  size=size )
ax.text( ti+1.1*dt, Sj, '$V(t_{i}, S_{j})}$', ha='left', color=g_hex, size=size )
ax.text( ti+1.1*dt, Sj+dS, '$V(t_{i}, S_{j+1})$', ha='left',  color='b', size=size )

plt.show()