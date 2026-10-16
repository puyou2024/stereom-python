# stereom-python

**High-performance Python refactoring of the MATLAB toolbox
[Quantitative-3D-structural-analysis-of-the-cellular-microstructures](https://github.com/Ziling-Wu/Quantitative-3D-structural-analysis-of-the-cellular-microstructures)
(T. Yang, Z. Wu, H. Chen, Y. Zhu, L. Li — Virginia Tech).**

> **This is a refactoring of an existing program, not an original method.**
> The analysis methodology — skeleton-based network registration of a
> binarized 3D volume and the node / branch / ring / chain / orientation /
> Fourier descriptors computed from it — was developed by the original
> authors and is described in their papers (see [Citation](#citation)).
> This repository re-implements that pipeline in Python for speed and
> HPC use, fixes a number of defects found in the original code, and adds
> a few extensions (listed below). No MATLAB code from the original
> repository is included here. If you use this code, **please cite the
> original papers.**

`stereom` takes a binary 3D volume of a cellular solid (a TIFF stack, a
NumPy array, or a region cut from an STL mesh) and extracts its skeleton
network: nodes, branches (length, thickness, tortuosity, orientation),
rings, chains, inter-branch angles, node planarity, branch profile fits
and the Fourier spectrum of the node arrangement. It was written for sea
urchin stereom but works on any strut-like cellular structure
(lattices, gyroids, spinodoids, foams).

## What changed relative to the original MATLAB toolbox

| | Original (MATLAB) | This refactoring (Python) |
|---|---|---|
| Language / stack | MATLAB | NumPy, SciPy, scikit-image, networkx, joblib |
| Demo volume (386×386×308) | hours (thickness stage dominant) | **~28 s** full pipeline, laptop CPU |
| Thickness | full-volume temporary per centerline voxel | one Euclidean distance transform |
| Node merging | windowed linear scan | KD-tree + union-find (exact, order-independent) |
| Ring search | serial, numeric-hash dedup | parallel, exact tuple dedup |
| Volume storage | `double` | `bool` (64× smaller) |
| Parameters | partly hard-coded | all in one `StereomConfig` (JSON/YAML) |
| Tests | — | 64 pytest tests (synthetic phantoms, scaling & rotation invariance) |

**Defects fixed** (IDs refer to the defect register in
[`docs/stereom_implementation_spec.md`](docs/stereom_implementation_spec.md) §5):
ring under-counting from hash collisions (D4); only 4 of 6 angle pairs
measured at degree-4 nodes (D6); wrong orientation for axis-aligned
branches (D8); merged node placed at an arbitrary voxel instead of the
cluster centroid (D9); missed merge neighbours (D10); unclamped `acos`
(D11); unchecked bounds in the thickness search (D13); chord instead of
local tangent as branch axis (D14); the `open` flag silently overriding
the merge distance (D17); and the thickness-stage performance bottleneck
(D18).

**Additions not in the original:** the Paper II quartic profile fit and
ring flatness filter, STL-mesh region extraction (`stereom
extract-regions`, exact per-slice cross-sectioning with automatic
outlier-slice detection), MATLAB export (`results.mat`) with plotting
scripts in [`matlab_viz/`](matlab_viz), batch processing for many
volumes, and SLURM examples.

Some results differ from the published numbers because of these fixes
(e.g. node-degree distribution and ring counts). This is expected and is
explained in [`docs/RUN_INSTRUCTIONS.md`](docs/RUN_INSTRUCTIONS.md) §12.

## Install

Python ≥ 3.10.

```bash
git clone https://github.com/puyou2024/stereom-python.git
cd stereom-python
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[mesh,viz,dev]"
pytest -q                          # check the install
```

Extras: `mesh` (STL support), `viz` (matplotlib), `yaml`, `parquet`,
`hpc` (mpi4py), `dev` (pytest). Plain `pip install -e .` installs the core
pipeline only.

## Quick start

Binary TIFF stack (or `.npy` / `.npz`):

```bash
stereom run --input path/to/tiff_folder --output results --voxel-size 0.65 --export-mat
```

`--voxel-size` is in µm. The demo dataset from the original repository
(`Code/Data/Binary.zip`, 0.65 µm/voxel) works directly as input.

Region of an STL mesh (bbox and pitch in the STL's own units, e.g. mm):

```bash
stereom extract-regions --stl sample.stl --bbox 19.5 24.5 4 9 0 7.5 \
    --name my_region --pitch 0.02 --stl-units-to-um 1000 --n-jobs -1 --output-dir regions_out
stereom run --input regions_out/my_region.npz --output results/my_region --export-mat
```

Choose `--pitch` so the thinnest strut is at least ~8 voxels across.

Python API:

```python
from stereom import io, pipeline
from stereom.config import StereomConfig

volume = io.load_volume("path/to/tiff_folder", voxel_size_um=0.65)
result = pipeline.run_full_pipeline(volume, StereomConfig(voxel_size_um=0.65))
result.branches_df.describe()
```

Outputs (CSV tables, `graph.npz`, `run_summary.json`, `config_used.json`,
optional `results.mat`), every parameter, the orientation-axis
convention, HPC usage and validation against the demo volume are
documented in **[`docs/RUN_INSTRUCTIONS.md`](docs/RUN_INSTRUCTIONS.md)**.

## Repository layout

```
src/stereom/   the package (each module's docstring documents its fixes vs. the MATLAB code)
tests/         pytest suite + synthetic phantom generators
matlab_viz/    MATLAB scripts to plot results.mat
docs/          RUN_INSTRUCTIONS.md (user guide), stereom_implementation_spec.md (design / defect analysis)
```

## Limitations

- Assumes thin, strut-like structures. Dense or plate-like regions
  skeletonize into surfaces rather than lines.
- A single volume must fit in one machine's memory (many volumes in
  parallel are supported via `pipeline.run_batch`).
- The finite-element / beam-model export from Paper II was never released
  in the original code and is not implemented here.

## Citation

The methodology is that of the original authors. If you use this code,
please cite:

1. T. Yang, Z. Wu, H. Chen, Y. Zhu, L. Li, *Quantitative 3D structural
   analysis of the cellular microstructure of sea urchin spines (I):
   Methodology*, Acta Biomaterialia 107 (2020) 204–217.
   <https://doi.org/10.1016/j.actbio.2020.02.034>
2. H. Chen, T. Yang, Z. Wu, Z. Deng, Y. Zhu, L. Li, *Quantitative 3D
   structural analysis of the cellular microstructure of sea urchin spines
   (II): Large-volume structural analysis*, Acta Biomaterialia 107 (2020)
   218–231. <https://doi.org/10.1016/j.actbio.2020.03.006>

and link to the original MATLAB repository:
<https://github.com/Ziling-Wu/Quantitative-3D-structural-analysis-of-the-cellular-microstructures>.

## License

No license has been chosen yet. The original MATLAB repository does not
carry a license either. Please contact the maintainer before reusing or
redistributing this code beyond academic use and citation.

## Acknowledgements

Original MATLAB toolbox and methodology: Ting Yang, Ziling Wu, Hongshun
Chen, Zhifei Deng, Yunhui Zhu and Ling Li (Virginia Tech; see above). Python refactoring: Pu You ([@puyou2024](https://github.com/puyou2024)).
