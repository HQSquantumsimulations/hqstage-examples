# HQS use cases and HQStage examples

This repository provides examples for HQS software modules with strong focus on solving quantum mechanical systems.
To execute the code in (most of) the examples you will need access to HQStage.
To learn more about licensing options and free trial visit our [cloud website](https://cloud.quantumsimulations.de).
You can find more information on the available modules and HQStage as such in our [documentation](https://docs.cloud.quantumsimulations.de).

## Use cases

The HQStage Modules are centered around solving quantum mechanical systems.
HQStage provides access to a wide range of example systems and Hamiltonians.
There is great interest in solving these systems and Hamiltonains on quantum computers.

HQStage examples and modules will be available for **three use cases**:

- Nuclear magnetic resonance
- Electron spectroscopy
- Magnetic response

As of now we have published the nuclear magnetic resonance (NMR) and Electron spectroscopy use case.
Magnetic resonance use cases will be published later in 2025.
[Read more ...](https://docs.cloud.quantumsimulations.de/use_cases.html)

### Nuclear magnetic resonance

Expert users in NMR spectroscopy should use our HQStage modules [`HQS Spectrum Tools`](https://docs.dev.cloud.quantumsimulations.de/licensing_and_modules/modules/hqs_spectrum_tools.html).
To get a feeling for NMR spectroscopy [try out](https://cloud.quantumsimulations.de/hqspectrum/trial) our end-to-end
product [HQSpectrum](https://quantumsimulations.de/hqspectrum).

You find a set of examples for NMR simulations using `HQS Spectrum Tools` on conventional and quantum computers in the [hqs_spectrum_tools](https://github.com/HQSquantumsimulations/hqstage-examples/tree/main/hqs_spectrum_tools) folder of this repository.
Learn more about our NMR examples in the [user documentation](https://docs.dev.cloud.quantumsimulations.de/hqs-spectrum-tools/examples.html).

### Electron spectroscopy

Electron spectroscopy measures excited states of molecules. Depending on the setup, it can probe purely electronic excitations, vibrational structure on top of them, or their coupling. A particularly important quantity that these measurements access is the singlet–triplet energy splitting ΔEST between the lowest singlet (S1) and triplet (T1) states. In what follows we focus on accurately predicting ΔEST for realistic molecules.

You find a set of examples for Electron spectroscopy simulations using [`HQS Quantum Solver`](https://docs.cloud.quantumsimulations.de/hqs-quantum-solvers/index.html) on conventional computers in the [hqs_quantum solver](https://github.com/HQSquantumsimulations/hqstage-examples/tree/main/hqs_quantum_solver) folder of this repository.
Learn more about our quantum solver and electron spectroscopy examples in the [user documentation](https://docs.cloud.quantumsimulations.de/hqs-quantum-solvers/index.html).

## Try HQStage in the cloud

The fastest way to explore HQStage (and the full HQS software stack) is to run it directly in your browser on our managed [JupyterLab](https://jupyter.org/) service.

- Sign in at <https://cloud.quantumsimulations.de>
- Get free credits (`HQ$`) at <https://cloud.quantumsimulations.de/account/credits>
- Launch JupyterLab at <https://cloud.quantumsimulations.de/notebook>
- Start using HQS modules with HQStage — no local setup required. We have pre-installed all eligible HQS Modules and common packages like numpy into your Python environment.
- Explore our use-cases and examples — downloaded and ready to use in your Notebooks
- You can start on our free plan; see [Licensing for details](licensing_and_modules/licensing.md)

Copy and paste the following code into your first jupyter notebook to produce a simple NMR spectrum.

<details>

<summary>Code for a simple NMR example</summary>

```python
from hqs_nmr_parameters import examples
from hqs_nmr import NMRCalculationParameters, calculate_spectrum
from matplotlib import pyplot as plt
print(f"Available moleucules: {examples.molecules.keys}")
molecule_name = '1,2,4-trichlorobenzene'
molecule_parameters = examples.molecules[molecule_name]
fig, ax = plt.subplots(figsize=(10, 6))
for field_T in [1, 5, 10]:
    result = calculate_spectrum(molecule_parameters, NMRCalculationParameters(field_T=field_T))
    ax.plot(result.spectrum.omegas_ppm, result.spectrum.intensity, label=f"B={field_T}T")
plt.legend()
ax.set_xlabel(r"$\delta$ [ppm]")
ax.set_ylabel("Intensity, arb. units")
ax.set_title(f"Spectrum of {molecule_name}")
plt.show()
```

</details>
