from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
fig, ax = plt.subplots(figsize = (6., 4.15))

HI = 1e12 # in GeV
Ni = -80.
Nf = 0.
num = 10000
alpha_i = 8

lmbda = 1e-9
V1 = lambda phi: lmbda * phi ** 4 / 24
dV1 = lambda phi: lmbda * phi ** 3 / 6
ddV1 = lambda phi: lmbda * phi ** 2 / 2
dddV1 = lambda phi: lmbda * phi
phi_i1 = np.sqrt(6 * HI ** 2 * alpha_i / lmbda)
kappa1 = lambda phi: dddV1(phi) * dV1(phi) / ddV1(phi) ** 2

p = 2.1
Lambda = 4.7e11 # in GeV
V2 = lambda phi: Lambda ** (4 - p) * np.abs(phi) ** p
dV2 = lambda phi: np.sign(phi) * p * Lambda ** (4 - p) * np.abs(phi) ** (p - 1)
ddV2 = lambda phi: p * (p - 1) * Lambda ** (4 - p) * np.abs(phi) ** (p - 2)
phi_i2 = (3 * HI ** 2 * alpha_i / p / (p - 1) / Lambda ** (4 - p)) ** (1 / (p - 2))
kappa2 = lambda phi: np.full_like(phi, (p - 2) / (p - 1))

def zero_mode1(y, N):
    phi, dphi = y
    return [dphi, -3 * dphi - dV1(phi) / HI ** 2]

def zero_mode2(y, N):
    phi, dphi = y
    return [dphi, -3 * dphi - dV2(phi) / HI ** 2]

Ns = np.linspace(Ni, Nf, num)
sol1 = integrate.odeint(zero_mode1, [phi_i1, 0], Ns)
sol2 = integrate.odeint(zero_mode2, [phi_i2, 0], Ns)
alpha1 = ddV1(sol1[:,0]) / 3 / HI ** 2
alpha2 = ddV2(sol2[:,0]) / 3 / HI ** 2
sr1 = np.argmin(np.abs(Ns - Ni - 1 / kappa1(sol1[:,0])))
sr2 = np.argmin(np.abs(Ns - Ni - 1 / kappa2(sol2[:,0])))

BLUE = (0.317647, 0.654902, 0.752941)
ORANGE = (1., 0.721569, 0.219608)

ax.plot(Ns, alpha1, color = BLUE, label = r'$p=4$')
ax.plot(Ns[sr1:], 1 / kappa1(sol1[sr1:,0]) / (Ns[sr1:] - Ni), color = BLUE, ls = '--')
ax.plot(Ns, alpha2, color = ORANGE, label = r'$p=2.1$')
ax.plot(Ns[sr2:], 1 / kappa2(sol2[sr2:,0]) / (Ns[sr2:] - Ni), color = ORANGE, ls = '--')

ax.set_yscale('log')
ax.set_xlim(-81, 1)
ax.set_ylim(1e-2, 10)
ax.set_xlabel('$N$')
ax.set_ylabel(r'$\alpha(N)$')
ax.legend(loc = 'upper right')

class CustomTicker(ticker.LogFormatterSciNotation): 
    def __call__(self, x, pos = None):
        if x not in np.concatenate((0.1 * np.arange(1, 10), np.arange(1, 10), 10 * np.arange(1, 10))): 
            return ticker.LogFormatterSciNotation.__call__(self, x, pos = None) 
        else:
            return "{x:g}".format(x = x)

ax.tick_params(which = 'both', direction = 'in')
ax.yaxis.set_major_formatter(CustomTicker())
secxax = ax.secondary_xaxis('top')
secxax.tick_params(which = 'both', direction = 'in')
plt.setp(secxax.get_xticklabels(), visible = False)
secyax = ax.secondary_yaxis('right')
secyax.tick_params(which = 'both', direction = 'in')
plt.setp(secyax.get_yticklabels(), visible = False)

fig.tight_layout()
#fig.show()
fig.savefig('alpha.pdf')
