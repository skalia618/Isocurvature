from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
fig, ax = plt.subplots(figsize = (6., 4.5))

data1 = np.load('Pdel_p4.npz')
HI1 = data1['HI']
ks1 = data1['ks']
Pdel1 = data1['Pdel']
coeff1 = data1['coeff']
exp1 = data1['exp']

data2 = np.load('Pdel_p2.1.npz')
HI2 = data2['HI']
ks2 = data2['ks']
Pdel2 = data2['Pdel']
coeff2 = data2['coeff']
exp2 = data2['exp']

BLUE = (0.317647, 0.654902, 0.752941)
ORANGE = (1., 0.721569, 0.219608)

cmb = np.loadtxt('cmb.txt').T
cmb[1] *= (0.14 / 0.12) ** 2
ax.fill_between(cmb[0], cmb[1], 1000, color = '0.9')
ax.plot(cmb[0], cmb[1], color = '0.75')
ax.text(5e-3, 3e-6, 'CMB', color = '0.2', ha = 'center', va = 'center')
lyman = np.loadtxt('lyman.txt').T
ax.fill_between(lyman[0], lyman[1], 1000, color = '0.9')
ax.plot(lyman[0], lyman[1], color = '0.75')
ax.text(0.85, 3e-4, r'Ly-$\alpha$', color = '0.2', ha = 'center', va = 'center')
ufd = np.loadtxt('ufd.txt').T
ax.fill_between(ufd[0], ufd[1], 1000, color = '0.9')
ax.plot(ufd[0], ufd[1], color = '0.75')
ax.text(150, 5e-2, 'Ultrafaint\nDwarfs', color = '0.2', ha = 'center', va = 'center')

ax.loglog(ks1, Pdel1, color = BLUE, label = r'$p=4$')
ax.loglog(ks1, coeff1 * np.exp(exp1), color = BLUE, ls = '--')
ax.loglog(ks2, Pdel2, color = ORANGE, label = r'$p=2.1$')
ax.loglog(ks2, coeff2 * np.exp(exp2), color = ORANGE, ls = '--')

kmax = 1.21e3 # in Mpc^-1; output of relic.py
ax.axvline(kmax, color = '0.7', ls = ':')

ax.set_xlim(5e-5, 2e6)
ax.set_xticks(np.logspace(-4, 6, 6))
ax.set_ylim(1e-14, 1)
ax.set_yticks(np.logspace(-14, 0, 8))
ax.set_xlabel(r'$k_\mathrm{today}\,[\mathrm{Mpc}^{-1}]$')
ax.set_ylabel(r'$P^\mathrm{iso}_\delta(k)$')
#ax.legend(loc = 'upper left')

class CustomTicker(ticker.LogFormatterSciNotation): 
    def __call__(self, x, pos = None):
        if x not in np.concatenate((0.1 * np.arange(1, 10), np.arange(1, 10), 10 * np.arange(1, 10))): 
            return ticker.LogFormatterSciNotation.__call__(self, x, pos = None) 
        else:
            return "{x:g}".format(x = x)

invMpc_to_GeV = 6.3949e-39
km_per_sec = 3.3356e-6
H0 = 70 * km_per_sec * invMpc_to_GeV # in GeV
omegam = 0.32
aeq = 1 / 3400
gstar_s = 3.9
gstar_rho = 3.4
gstar_rh = 106.75
def k_to_N(k, HI = HI1):
    a_today = np.sqrt(HI / H0) / (omegam * aeq) ** (1 / 4) * gstar_rh ** (1 / 12) * gstar_rho ** (1 / 4) / gstar_s ** (1 / 3)
    return np.log(k * a_today * invMpc_to_GeV / HI)
def N_to_k(N, HI = HI1):
    a_today = np.sqrt(HI / H0) / (omegam * aeq) ** (1 / 4) * gstar_rh ** (1 / 12) * gstar_rho ** (1 / 4) / gstar_s ** (1 / 3)
    return HI * np.exp(N) / a_today / invMpc_to_GeV

ax.tick_params(which = 'both', direction = 'in')
ax.xaxis.set_major_formatter(CustomTicker())
ax.yaxis.set_major_formatter(CustomTicker())
secxax = ax.secondary_xaxis('top', functions = (k_to_N, N_to_k), zorder = 1)
secxax.tick_params(which = 'both', direction = 'in')
secxax.set_xscale('linear')
secxax.set_xlabel(r'$N_*$')
secyax = ax.secondary_yaxis('right')
secyax.tick_params(which = 'both', direction = 'in')
secyax.set_yticks(ax.get_yticks())
plt.setp(secyax.get_yticklabels(), visible = False)

cmbinterp = np.interp(ks1, cmb[0], cmb[1])
iso_factor = np.min(cmbinterp / Pdel1)
min_ind = np.argmin(cmbinterp / Pdel1)
slope = np.log(Pdel1[min_ind + 1] / Pdel1[min_ind - 1]) / np.log(ks1[min_ind + 1] / ks1[min_ind - 1])
print('Quartic isocurvature safe by factor of {:.2f}'.format(iso_factor))
print('Slope at constraining scale: {:.2e}'.format(slope))

fig.tight_layout()
#fig.show()
fig.savefig('Pdel.pdf')
