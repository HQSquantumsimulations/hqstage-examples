from struqture_py.spins import PauliHamiltonian
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector

from itertools import combinations
import numpy as np
from typing import Tuple, Any
import re


def struqture_hamiltonian_to_qiskit_op(
    ham_struqture: PauliHamiltonian,
    n_qubits: int,
    reverse_qubit_order: bool = True,
) -> SparsePauliOp:
    """
    Convert a struqture PauliHamiltonian into a Qiskit SparsePauliOp.

    This parses keys such as '0Z', '0X1X', or '10Y11Z' emitted by struqture and
    builds the corresponding Qiskit Pauli labels of length ``n_qubits``. The
    mapping between struqture site indices and Qiskit qubit positions is
    controlled by ``reverse_qubit_order``.

    Parameters
    ----------
    ham_struqture : PauliHamiltonian
        Input Hamiltonian from ``struqture_py.spins`` whose keys encode which
        Pauli operators act on which spin indices.
    n_qubits : int
        Number of spins/qubits. Determines the label length and the placement of
        identity operators on unaffected qubits.
    reverse_qubit_order : bool, optional
        Endianness toggle for mapping site index ``i`` to a Qiskit label
        character. When ``True`` (default), struqture site index 0 maps to the
        rightmost character of the Qiskit label (little-endian), i.e.,
        ``q = n_qubits - 1 - i``. When ``False``, index ``i`` maps to the ``i``-th
        character (big-endian).

        Example for ``n_qubits = 2``:
        - Key '0X1Y' -> label 'YX' if ``reverse_qubit_order=True``.
        - Key '0X1Y' -> label 'XY' if ``reverse_qubit_order=False``.

    Returns
    -------
    SparsePauliOp
        The assembled operator with Pauli labels and complex coefficients. If a
        key 'I' is present in ``ham_struqture``, it is mapped to the identity on
        all qubits (label of all 'I').

    Raises
    ------
    IndexError
        If a site index in a key maps outside the range ``0 .. n_qubits-1``.
    """
    labels = []
    coeffs = []
    token_re = re.compile(r"(\d+)([XYZ])")

    for key, val in zip(ham_struqture.keys(), ham_struqture.values()):
        s = str(key)  # e.g., '0Z', '0X1X', '10X11X'
        pauli = ["I"] * n_qubits
        if s != "I":
            for m in token_re.finditer(s):
                idx = int(m.group(1))  # site index (can be multi-digit)
                op = m.group(2)  # 'X', 'Y', or 'Z'
                q = (n_qubits - 1 - idx) if reverse_qubit_order else idx
                if not (0 <= q < n_qubits):
                    raise IndexError(
                        f"Site index {idx} (mapped to qubit {q}) out of range 0..{n_qubits-1}"
                    )
                pauli[q] = op
        labels.append("".join(pauli))
        coeffs.append(complex(val))

    return SparsePauliOp(labels, coeffs)


def xsum_eigenstate_template(n_qubits: int):
    """
    Build a parameterized circuit preparing product eigenstates of Σ_i X_i.

    The template prepares, per qubit, |+> (eigenstate of X with eigenvalue +1)
    followed by an RZ rotation whose parameter toggles between |+> and |->.
    Setting phi[i] = 0 keeps qubit i in |+>, while phi[i] = π yields |-> up to
    a global (physically irrelevant) phase.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the register.

    Returns
    -------
    QuantumCircuit, ParameterVector
        Circuit with RY(π/2) then RZ(phi[i]) on each qubit, and the associated
        parameter vector ``phi`` of length ``n_qubits``. Bind ``phi`` with 0 or
        π to select the desired X-basis product eigenstate per qubit.

    Notes
    -----
    - Convention: X-basis coding uses 0 → |+>, π → |-> in the returned
      ``ParameterVector``. The overall global phase introduced by RZ is irrelevant
      for expectation values.
    """
    phi = ParameterVector("phi", n_qubits)
    qc = QuantumCircuit(n_qubits, name="XsumEigen")
    for q in range(n_qubits):
        qc.ry(np.pi / 2, q)  # |0> -> |+>
        qc.rz(phi[q], q)  # 0 keeps |+>, pi -> |-> (global phase irrelevant)
    return qc, phi


# 1) Helper: X-magnetization sign for a bitstring (0->|+>, 1->|->)
def magnetization_sign(bits):
    """
    Compute the total σ_x magnetization sign for an X-basis bitstring.

    Bit convention: 0 encodes |+> (eigenvalue +1 under X), 1 encodes |->
    (eigenvalue -1 under X). The returned value is the sum of ±1 contributions
    over all qubits, i.e., Σ_i s_i with s_i ∈ {+1, -1}. For spin-1/2 operators,
    the physical I_x total magnetization is related by I_x^tot = (1/2) Σ_i σ_x^i.

    Parameters
    ----------
    bits : sequence of int
        Tuple/list of 0/1 values of length n_qubits representing an X-basis
        product state.

    Returns
    -------
    int
        The integer magnetization in the Pauli-σ convention (range −n..+n).
    """
    # Using Pauli-σ convention for obs_x: each qubit contributes +1 (|+>) or -1 (|->)
    return sum(1 if b == 0 else -1 for b in bits)


# 2) Generate only positive-magnetization bitstrings
def positive_magnetization_bitstrings(n: int):
    """
    Yield X-basis product states with strictly positive total σ_x magnetization.

    For odd ``n``, this yields exactly half of all bitstrings (2^(n-1)). For even
    ``n``, bitstrings with zero total σ_x (k = n/2 ones) are omitted. The function
    yields tuples of length ``n`` with entries 0 (|+>) or 1 (|->). Order is
    deterministic: ascending number of ones (|->) and lexicographic within each
    Hamming weight.

    Parameters
    ----------
    n : int
        Number of qubits.

    Yields
    ------
    tuple[int, ...]
        Bitstrings in the X basis with strictly positive Σ_i σ_x^i.
    """
    max_ones = (n - 1) // 2  # <= floor((n-1)/2)
    for k in range(0, max_ones + 1):  # number of |-> entries ("1"s)
        for ones in combinations(range(n), k):
            bits = [0] * n
            for j in ones:
                bits[j] = 1
            yield tuple(bits)


def time_signal_to_spectrum(
    c_xx: np.ndarray,
    c_yx: np.ndarray,
    trotter_timestep: float,
    number_of_trottersteps: int,
    Hz_to_ppm_conversion: float,
    chem_shift_hz: float,
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Compute an NMR spectrum from causal time-domain correlation functions.

    The frequency-domain signal is assembled from the real part of the causal
    Fourier transform of C_xx(t) and the imaginary part of the causal transform
    of C_yx(t), as suggested by linear response relations. A rotating-frame
    offset ``chem_shift_hz`` (e.g., mean Zeeman term divided by 2π) is subtracted
    during time evolution and added back to the frequency axis here. The current
    implementation uses a "2 × offset" correction consistent with the construction
    of C_xx and C_yx in this notebook.

    Parameters
    ----------
    c_xx : np.ndarray
        Discrete samples of C_xx(t) for t ≥ 0 on a uniform grid.
    c_yx : np.ndarray
        Discrete samples of C_yx(t) for t ≥ 0 on a uniform grid.
    trotter_timestep : float
        Uniform time step Δt used to sample the correlation functions (seconds).
    number_of_trottersteps : int
        Number of time steps (length of the time signals).
    Hz_to_ppm_conversion : float
        Conversion factor from Hz to ppm at the chosen field, typically
        1e6 / ν0, where ν0 is the Larmor frequency of the reference nucleus
        (Hz).
    chem_shift_hz : float
        Rotating-frame offset in Hz. This routine shifts the frequency axis by
        2 × ``chem_shift_hz`` to undo the rotating-frame subtraction used earlier
        in the workflow.

    Returns
    -------
    spectrum_full : np.ndarray
        The normalized magnitude spectrum on the ppm axis (1D array, length
        ``number_of_trottersteps``).
    ppm : np.ndarray
        The chemical shift axis (ppm) corresponding to ``spectrum_full``.

    Notes
    -----
    - Spectral normalization divides by max absolute value to stabilize plotting
      and comparison; set or remove externally if a different normalization is
      desired.
    - The causal FFT is used via ``fft_causal`` to avoid implicit negative-time
      contributions.
    """
    # causal FFT of the signals
    fft_xx = fft_causal(c_xx)
    fft_yx = fft_causal(c_yx)

    frequencies = np.fft.fftshift(np.fft.fftfreq(number_of_trottersteps, d=trotter_timestep))

    # Convert to ppm; shift by twice the rotating-frame offset (see construction above)
    ppm = (frequencies - 2 * chem_shift_hz) * Hz_to_ppm_conversion

    spectrum_full = fft_xx.real + fft_yx.imag

    # Normalize robustly
    max_abs = np.max(np.abs(spectrum_full)) or 1.0
    spectrum_full /= max_abs

    return np.abs(spectrum_full), np.asarray(ppm, dtype=float)


def fft_causal(f: np.ndarray) -> np.ndarray:
    """
    Causal Fourier transform via even/odd symmetrization and fftshift.

    Given samples of a causal sequence f(t ≥ 0), this constructs even and odd
    extensions to negative times, computes FFTs with orthogonal normalization,
    combines them, and downsamples to recover an FFT of length equal to the
    original signal. The result is fftshifted to align zero frequency at the
    center.

    Parameters
    ----------
    f : np.ndarray
        1D array of real or complex samples of the causal signal.

    Returns
    -------
    np.ndarray
        Complex frequency-domain array of length len(f), fftshifted.

    Notes
    -----
    - The even/odd construction reduces artifacts due to the implicit extension
      of the signal to t < 0.
    - Uses ``norm="ortho"`` in numpy FFT for numerical stability and a consistent
      scale, which is absorbed by subsequent normalization in this workflow.
    """
    L = len(f)

    # Even part (symmetrize)
    fe = np.zeros(2 * L - 1, dtype=f.dtype)
    fe[:L] = f
    fe[L:] = f[:-1][::-1]
    fe /= 2

    # Odd part (antisymmetrize)
    fo = np.zeros(2 * L - 1, dtype=f.dtype)
    fo[:L] = f
    fo[L:] = -f[:-1][::-1]
    fo /= 2
    fo[0] = 0  # avoid jump discontinuity at t=0

    # FFT with orthogonal normalization and shift
    fft_even = np.fft.fftshift(np.fft.fft(fe, norm="ortho"))
    fft_odd = np.fft.fftshift(np.fft.fft(fo, norm="ortho"))

    # Combine and downsample to original length
    combined_fft = (fft_even + fft_odd)[::2]

    return combined_fft


def spectrum_rmse(
    spec_a: np.ndarray,
    freq_a: np.ndarray,
    spec_b: np.ndarray,
    freq_b: np.ndarray,
) -> float:
    """
    Compute RMSE between two 1D spectra that may live on different axes.

    The routine finds the overlap of the two frequency axes, interpolates both
    spectra onto a common uniform grid over the overlap, and returns the root-
    mean-square error between the aligned arrays. It assumes inputs are already
    normalized if desired (no normalization is applied internally).

    Parameters
    ----------
    spec_a : np.ndarray
        Amplitudes of spectrum A (1D array).
    freq_a : np.ndarray
        Frequency axis for spectrum A (same length as ``spec_a``; Hz or ppm).
    spec_b : np.ndarray
        Amplitudes of spectrum B (1D array).
    freq_b : np.ndarray
        Frequency axis for spectrum B (same length as ``spec_b``; Hz or ppm).

    Returns
    -------
    float
        Root-mean-square error computed on the common axis.

    Raises
    ------
    ValueError
        If the input axes do not overlap or contain fewer than two points in the
        overlap region to define an interpolation grid.
    """
    # determine overlap
    left = max(freq_a[0], freq_b[0])
    right = min(freq_a[-1], freq_b[-1])
    if not (left < right):
        raise ValueError("No overlapping frequency range between the two spectra.")

    # masks inside overlap
    mask_a = (freq_a >= left) & (freq_a <= right)
    mask_b = (freq_b >= left) & (freq_b <= right)
    xa, ya = freq_a[mask_a], spec_a[mask_a]
    xb, yb = freq_b[mask_b], spec_b[mask_b]

    if len(xa) < 2 or len(xb) < 2:
        raise ValueError("Not enough points in the overlap to compare (need at least 2).")

    step = min(np.median(np.diff(xa)), np.median(np.diff(xb)))
    # ensure at least 2 points
    npts = max(2, int(np.floor((right - left) / step)) + 1)
    x_common = np.linspace(left, right, npts)
    A = np.interp(x_common, xa, ya, left=ya[0], right=ya[-1])
    B = np.interp(x_common, xb, yb, left=yb[0], right=yb[-1])

    rmse = float(np.sqrt(np.mean((A - B) ** 2)))

    return rmse
