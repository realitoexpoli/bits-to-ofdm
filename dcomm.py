"""dcomm: small helpers shared by the labs (Digital Communications in Python).

Everything here is plain NumPy/SciPy, written to be read. Each lab shows how these are built
before using them, so you can open this file and recognise every line.
"""
import numpy as np
from scipy.special import erfc

# ----------------------------------------------------------------------------- basics
def qfunc(x):
    """Gaussian tail probability Q(x) = P(N(0,1) > x)."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2))


def random_bits(n, seed=None):
    """n random bits (0/1) as a uint8 array."""
    return np.random.default_rng(seed).integers(0, 2, n, dtype=np.uint8)


def bits_to_ints(bits, k):
    """Groups of k bits (MSB first) -> integers 0..2^k-1."""
    bits = np.asarray(bits).reshape(-1, k)
    return bits @ (1 << np.arange(k - 1, -1, -1))


def ints_to_bits(ints, k):
    """Integers -> k bits each (MSB first), flattened."""
    ints = np.asarray(ints)[:, None]
    return ((ints >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).ravel()


def gray(n):
    """Binary-reflected Gray code of n."""
    return n ^ (n >> 1)


# ----------------------------------------------------------------------------- constellations
def pam_levels(M):
    """M-PAM amplitudes ..., -3, -1, 1, 3, ... (not normalised)."""
    return np.arange(-(M - 1), M, 2, dtype=float)


def constellation(name):
    """Unit-average-energy constellation with Gray labelling.

    Returns an array `c` where c[i] is the symbol sent for the integer i (the bits of i, MSB first).
    name: 'bpsk', 'qpsk', '16qam' or '64qam'.
    """
    name = name.lower()
    if name == 'bpsk':
        return np.array([-1.0 + 0j, 1.0 + 0j])
    sizes = {'qpsk': 4, '16qam': 16, '64qam': 64}
    if name not in sizes:
        raise ValueError('unknown constellation: ' + name)
    M = sizes[name]
    m = int(np.sqrt(M)); kk = int(np.log2(m))
    levels = pam_levels(m)
    # Gray label g -> PAM level, separately on I and Q
    level_of_gray = np.empty(m)
    for idx in range(m):
        level_of_gray[gray(idx)] = levels[idx]
    c = np.empty(M, dtype=complex)
    for i in range(M):
        hi, lo = i >> kk, i & (m - 1)            # first half of the bits -> I, second half -> Q
        c[i] = level_of_gray[hi] + 1j * level_of_gray[lo]
    return c / np.sqrt(np.mean(np.abs(c) ** 2))


def modulate(bits, c):
    """Bits -> symbols with constellation c (len(c) a power of 2)."""
    k = int(np.log2(len(c)))
    return c[bits_to_ints(bits, k)]


def demodulate(symbols, c):
    """Minimum-distance (hard) decision: symbols -> bits."""
    k = int(np.log2(len(c)))
    d = np.abs(np.asarray(symbols)[:, None] - c[None, :])
    return ints_to_bits(np.argmin(d, axis=1), k)


# ----------------------------------------------------------------------------- channel
def awgn(symbols, ebn0_db, bits_per_symbol, seed=None, sps=1):
    """Adds complex white Gaussian noise for a given Eb/N0 (dB).

    Assumes unit average symbol energy Es = 1, so Eb = 1/bits_per_symbol, and noise of variance N0/2 per
    real dimension on each sample. That is right for symbol-rate signals and for signals shaped with a
    unit-energy pulse (like dc.rrc) followed by the matched filter.
    sps: use it only for a pulse of `sps` samples of amplitude 1 (rectangular), whose symbol energy is sps.
    """
    rng = np.random.default_rng(seed)
    ebn0 = 10 ** (ebn0_db / 10)
    n0 = 1 / (bits_per_symbol * ebn0)
    sigma = np.sqrt(n0 * sps / 2)
    x = np.asarray(symbols)
    if np.iscomplexobj(x):
        return x + sigma * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return x + sigma * rng.standard_normal(x.shape)  # real (baseband BPSK on one dimension)


# ----------------------------------------------------------------------------- theory
def ber_theory(name, ebn0_db):
    """Bit error rate with Gray coding in AWGN (exact for BPSK/QPSK, very tight for M-QAM)."""
    g = 10 ** (np.asarray(ebn0_db, dtype=float) / 10)
    name = name.lower()
    if name in ('bpsk', 'qpsk'):
        return qfunc(np.sqrt(2 * g))
    M = {'16qam': 16, '64qam': 64}[name]
    k = np.log2(M)
    return 4 / k * (1 - 1 / np.sqrt(M)) * qfunc(np.sqrt(3 * k * g / (M - 1)))


# ----------------------------------------------------------------------------- pulses
def rrc(beta, span, sps):
    """Root-raised-cosine pulse, `span` symbols long, `sps` samples per symbol, unit energy."""
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(4 * beta * ti) - 1) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def raised_cosine(beta, span, sps):
    """Raised-cosine pulse (= RRC * RRC), peak 1 at t = 0, zero at the other symbol instants."""
    h = np.convolve(rrc(beta, span, sps), rrc(beta, span, sps))
    return h / h.max()


def upsample(symbols, sps):
    """Inserts sps-1 zeros after each symbol."""
    out = np.zeros(len(symbols) * sps, dtype=np.asarray(symbols).dtype)
    out[::sps] = symbols
    return out
