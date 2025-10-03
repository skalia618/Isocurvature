from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
fig, ax = plt.subplots(figsize = (6., 4.5))

BLUE = np.array([0.317647, 0.654902, 0.752941])
ORANGE = np.array([1., 0.721569, 0.219608])
RED = np.array([0.921569, 0.494118, 0.431373])
PINK = np.array([0.894118, 0.709804, 0.749020])

m_min = 1e-4 # in eV
m_max = 1e6 # in eV
HI1 = 1e10 # in GeV
HI2 = 1e12 # in GeV
masses = np.logspace(np.log10(m_min), np.log10(m_max), 1000)

sidm_bound = 1 # in cm^2/g
ev3_to_gpercm2 = 4.5782e-24
ax.fill_between(masses, np.sqrt(sidm_bound * 64 * np.pi * masses ** 3 * ev3_to_gpercm2), 1, color = '0.95', zorder = 0.5)
ax.loglog(masses, np.sqrt(sidm_bound * 64 * np.pi * masses ** 3 * ev3_to_gpercm2), color = '0.8', zorder = 0.5)
ax.text(0.17, 1.3e-10, 'SIDM', color = '0.2', ha = 'center', va = 'center')

struc_bound = 7.1e-4 # at m = 1 eV; from Sec. 7 of arxiv:2306.12477
ax.fill_between(masses, struc_bound * masses ** 4, 1, color = '0.9', zorder = 0.6)
ax.loglog(masses, struc_bound * masses ** 4, color = '0.75', zorder = 0.6)
ax.text(1e-3, 5e-11, 'Structure\nFormation', color = '0.2', ha = 'center', va = 'center')

Mpl = 2.4353e18 # in GeV
ax.fill_between(masses, 6 * HI1 ** 2 / Mpl ** 2, 1e-30, color = PINK, alpha = 0.2, zorder = 0.7)
ax.loglog(masses, np.full_like(masses, 6 * HI1 ** 2 / Mpl ** 2), color = PINK, zorder = 0.7)
ax.loglog(masses, np.full_like(masses, 6 * HI2 ** 2 / Mpl ** 2), color = PINK, ls = ':', zorder = 0.7)
ax.text(0.2, 4e-17, 'Transplanckian', color = 0.5 * PINK, ha = 'center', va = 'center')

HI_ref = 1e12 # in GeV
lmbda_ref = 1e-9
iso_factor = 2.68 # output of Pdel_plot.py
slope = 9.45e-2 # output of Pdel_plot.py
ax.fill_between(masses, iso_factor * lmbda_ref * (HI1 / HI_ref) ** (slope / 2), 1, color = RED, alpha = 0.2, zorder = 0.8)
ax.loglog(masses, np.full_like(masses, iso_factor * lmbda_ref * (HI1 / HI_ref) ** (slope / 2)), color = RED, zorder = 0.8)
ax.loglog(masses, np.full_like(masses, iso_factor * lmbda_ref * (HI2 / HI_ref) ** (slope / 2)), color = RED, ls = ':', zorder = 0.8)
ax.text(10, 1.4e-8, 'Isocurvature', color = 0.5 * RED, ha = 'center', va = 'center')

invMpc_to_GeV = 6.3949e-39
km_per_sec = 3.3356e-6
GN = 6.7088e-39 # in GeV^-2
H0 = 70 * km_per_sec * invMpc_to_GeV # in GeV
omegam = 0.32
aeq = 1 / 3400
gstar_s = 3.9
gstar_rho = 3.4
gstar_rh = 106.75
Neff = 0.2
TBBN = 1e-3 # in GeV
T_today = 2.35e-13 # in GeV
gstar_BBN = 10.6
rhoBBN_max = np.pi ** 2 / 30 * 7 / 4 * (4 / 11) ** (4 / 3) * Neff * TBBN ** 4

m_sim = 1e17 # in eV
relic_data = np.load('relic.npz')
a = relic_data['a']
rho = relic_data['rho']
a_today1 = np.sqrt(HI1 / H0) / (omegam * aeq) ** (1 / 4) * gstar_rh ** (1 / 12) * gstar_rho ** (1 / 4) / gstar_s ** (1 / 3)
a_today2 = np.sqrt(HI2 / H0) / (omegam * aeq) ** (1 / 4) * gstar_rh ** (1 / 12) * gstar_rho ** (1 / 4) / gstar_s ** (1 / 3)
aBBN1 = a_today1 * T_today / TBBN * (gstar_s / gstar_BBN) ** (1 / 3)
aBBN2 = a_today2 * T_today / TBBN * (gstar_s / gstar_BBN) ** (1 / 3)
BBN_bound1 = lambda m: np.piecewise(m, [aBBN1 * m / m_sim * HI_ref / HI1 < a[0], aBBN1 * m / m_sim * HI_ref / HI1 > a[-1]],
                                    [lmbda_ref * rho[0] * (a[0] / aBBN1 * HI1 / HI_ref) ** 4 / rhoBBN_max,
                                     lambda mass: lmbda_ref * rho[-1] * (mass / m_sim) * (a[-1] / aBBN1 * HI1 / HI_ref) ** 3 / rhoBBN_max,
                                     lambda mass: lmbda_ref * np.interp(aBBN1 * mass / m_sim * HI_ref / HI1, a, rho) * (mass / m_sim) ** 4 / rhoBBN_max])
BBN_bound2 = lambda m: np.piecewise(m, [aBBN2 * m / m_sim * HI_ref / HI2 < a[0], aBBN2 * m / m_sim * HI_ref / HI2 > a[-1]],
                                    [lmbda_ref * rho[0] * (a[0] / aBBN2 * HI2 / HI_ref) ** 4 / rhoBBN_max,
                                     lambda mass: lmbda_ref * rho[-1] * (mass / m_sim) * (a[-1] / aBBN2 * HI2 / HI_ref) ** 3 / rhoBBN_max,
                                     lambda mass: lmbda_ref * np.interp(aBBN2 * mass / m_sim * HI_ref / HI2, a, rho) * (mass / m_sim) ** 4 / rhoBBN_max])
ax.fill_between(masses, BBN_bound1(masses), 1e-30, color = ORANGE, alpha = 0.1, zorder = 0.9)
ax.loglog(masses, BBN_bound1(masses), color = ORANGE, zorder = 0.9)
ax.loglog(masses, BBN_bound2(masses), color = ORANGE, ls = ':', zorder = 0.9)
ax.text(1e4, 4e-18, 'BBN', color = 0.6 * ORANGE, ha = 'center', va = 'center')

m_ref = 66.1 # in eV; output of relic.py
ax.loglog(masses, lmbda_ref * masses / m_ref * (HI1 / HI_ref) ** 1.5, color = BLUE, zorder = 1)
ax.loglog(masses, lmbda_ref * masses / m_ref * (HI2 / HI_ref) ** 1.5, color = BLUE, ls = ':', zorder = 1)

ax.set_xlim(m_min, m_max)
ax.set_xticks(np.logspace(-4, 6, 6))
ax.set_ylim(1e-19, 1e-7)
ax.set_xlabel(r'$m\,[\mathrm{eV}]$')
ax.set_ylabel(r'$\lambda$')

class CustomTicker(ticker.LogFormatterSciNotation): 
    def __call__(self, x, pos = None):
        if x not in np.concatenate((0.1 * np.arange(1, 10), np.arange(1, 10), 10 * np.arange(1, 10))): 
            return ticker.LogFormatterSciNotation.__call__(self, x, pos = None) 
        else:
            return "{x:g}".format(x = x)

ax.tick_params(which = 'both', direction = 'in')
ax.xaxis.set_major_formatter(CustomTicker())
secxax = ax.secondary_xaxis('top')
secxax.tick_params(which = 'both', direction = 'in')
plt.setp(secxax.get_xticklabels(), visible = False)
secyax = ax.secondary_yaxis('right')
secyax.tick_params(which = 'both', direction = 'in')
plt.setp(secyax.get_yticklabels(), visible = False)

fig.tight_layout()
#fig.show()
fig.savefig('quartic.pdf')
