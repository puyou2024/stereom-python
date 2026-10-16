"""Command-line entry point: ``stereom run ...`` / ``stereom init-config ...``.

See ``RUN_INSTRUCTIONS.md`` at the repository root for full usage,
including HPC/SLURM examples.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from stereom import io as stereom_io
from stereom import tables
from stereom.config import StereomConfig
from stereom.pipeline import run_full_pipeline
from stereom.volume import Volume


def _load_input_volume(args: argparse.Namespace, config: StereomConfig) -> Volume:
    input_path = Path(args.input)
    if input_path.is_dir():
        return stereom_io.load_volume(
            input_path,
            start_page=args.start_page,
            end_page=args.end_page,
            sig_digits=args.sig_digits,
            image_name=args.image_name,
            style=args.style,
            voxel_size_um=config.voxel_size_um,
            axes=config.axes,
        )
    if input_path.suffix == ".npz":
        return stereom_io.load_volume_npz(input_path)
    if input_path.suffix == ".npy":
        arr = np.load(input_path)
        return stereom_io.load_volume_array(arr, voxel_size_um=config.voxel_size_um, axes=config.axes)
    raise ValueError(f"Unrecognized input {input_path}: expected a TIFF-stack directory, .npz or .npy")


def cmd_run(args: argparse.Namespace) -> None:
    config = StereomConfig.load(args.config) if args.config else StereomConfig()
    if args.voxel_size is not None:
        config.voxel_size_um = args.voxel_size
    if args.n_workers is not None:
        config.n_workers = args.n_workers

    t0 = time.time()
    volume = _load_input_volume(args, config)
    print(f"[stereom] loaded volume {volume.shape}, solid fraction {volume.solid_fraction:.4f}", file=sys.stderr)

    result = run_full_pipeline(
        volume,
        config,
        compute_rings=not args.skip_rings,
        compute_chains=not args.skip_chains,
        compute_fft=not args.skip_fft,
    )
    elapsed = time.time() - t0
    print(
        f"[stereom] pipeline done in {elapsed:.1f}s: "
        f"{len(result.nodes_df)} nodes, {len(result.branches_df)} branches, "
        f"merge_d_m={result.merge_d_m_um:.2f} um",
        file=sys.stderr,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables.export_tables(result.nodes_df, result.branches_df, out_dir, fmt=args.table_format)
    tables.save_graph_npz(result.graph, out_dir / "graph.npz")

    if result.rings_df is not None:
        result.rings_df.to_csv(out_dir / "rings.csv", index=False)
    if result.chains_df is not None:
        result.chains_df.to_csv(out_dir / "chains.csv", index=False)
    if result.angles_df is not None:
        result.angles_df.to_csv(out_dir / "inter_branch_angles.csv", index=False)
    if result.planarity_df is not None:
        result.planarity_df.to_csv(out_dir / "node_planarity.csv", index=False)
    if result.fft_spectrum is not None:
        np.savez_compressed(out_dir / "fft_spectrum.npz", spectrum=result.fft_spectrum)

    fit_rows = [
        {
            "branch_id": r.branch_id,
            "accepted": r.accepted,
            "reject_reason": r.reject_reason,
            "a0": r.a0,
            "a1": r.a1,
            "b": r.b,
            "r2": r.r2,
            "t_min_um": r.t_min_um,
            "length_um": r.length_um,
            "quartic_a": r.quartic_a,
        }
        for r in result.profile_fits
    ]
    import pandas as pd

    pd.DataFrame(fit_rows).to_csv(out_dir / "profile_fits.csv", index=False)

    build_stats_dict = {
        "n_candidate_nodes": result.build_stats.n_candidate_nodes,
        "n_segments": result.build_stats.n_segments,
        "n_branches_kept": result.build_stats.n_branches_kept,
        "n_dropped_dangling": result.build_stats.n_dropped_dangling,
        "n_dropped_ambiguous": result.build_stats.n_dropped_ambiguous,
        "n_self_loops_dropped": result.build_stats.n_self_loops_dropped,
        "merge_d_m_um": result.merge_d_m_um,
        "elapsed_seconds": elapsed,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(build_stats_dict, indent=2))
    config.save(out_dir / "config_used.json")

    if args.export_mat:
        import shutil

        from stereom.matlab_export import export_matlab

        input_path = Path(args.input)
        alignment_path = input_path.with_name(input_path.stem + "_stl_alignment.json")
        if alignment_path.exists():
            alignment = json.loads(alignment_path.read_text())
            stl_src = input_path.parent / alignment["stl_file"]
            if stl_src.exists():
                shutil.copy2(stl_src, out_dir / alignment["stl_file"])
            else:
                print(f"[stereom] WARNING: {alignment_path} references {stl_src} but it doesn't exist; skipping STL overlay data", file=sys.stderr)
                alignment_path = None
        else:
            alignment_path = None

        export_matlab(result.graph, out_dir / "results.mat", pipeline_result=result, stl_alignment_path=alignment_path)
        print(f"[stereom] wrote {out_dir / 'results.mat'}", file=sys.stderr)

    print(f"[stereom] wrote results to {out_dir}", file=sys.stderr)


def cmd_init_config(args: argparse.Namespace) -> None:
    StereomConfig().save(args.output)
    print(f"[stereom] wrote default config to {args.output}", file=sys.stderr)


def cmd_extract_regions(args: argparse.Namespace) -> None:
    from stereom.mesh_regions import RegionSpec, extract_regions, load_regions_json

    if args.spec:
        specs = load_regions_json(args.spec)
    else:
        if not (args.stl and args.bbox and args.name and args.pitch is not None):
            raise SystemExit(
                "Either --spec, or all of --stl/--bbox/--name/--pitch, are required."
            )
        xmin, xmax, ymin, ymax, zmin, zmax = args.bbox
        specs = [
            RegionSpec(
                name=args.name,
                stl_path=args.stl,
                bbox_min=(xmin, ymin, zmin),
                bbox_max=(xmax, ymax, zmax),
                pitch_stl_units=args.pitch,
                stl_units_to_um=args.stl_units_to_um,
            )
        ]
    extract_regions(specs, args.output_dir, n_jobs=args.n_jobs, export_stl=not args.no_export_stl, verbose=True)


def cmd_export_mat(args: argparse.Namespace) -> None:
    import shutil

    from stereom import tables as _tables
    from stereom.matlab_export import export_matlab

    results_dir = Path(args.results_dir)
    graph = _tables.load_graph_npz(results_dir / "graph.npz")

    alignment_path = None
    if args.stl_alignment:
        alignment_path = Path(args.stl_alignment)
        alignment = json.loads(alignment_path.read_text())
        stl_src = alignment_path.parent / alignment["stl_file"]
        if stl_src.exists() and stl_src.resolve() != (results_dir / alignment["stl_file"]).resolve():
            shutil.copy2(stl_src, results_dir / alignment["stl_file"])

    export_matlab(
        graph,
        results_dir / "results.mat",
        extra_csv_dir=results_dir,
        stl_alignment_path=alignment_path,
    )
    print(f"[stereom] wrote {results_dir / 'results.mat'}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stereom", description="Quantitative 3D stereom network analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the full pipeline on one volume")
    p_run.add_argument("--input", required=True, help="TIFF-stack directory, or a .npz/.npy volume file")
    p_run.add_argument("--output", required=True, help="Output directory")
    p_run.add_argument("--config", default=None, help="StereomConfig JSON/YAML file (default: built-in defaults)")
    p_run.add_argument("--voxel-size", type=float, default=None, help="Override voxel size (um), isotropic")
    p_run.add_argument("--n-workers", type=int, default=None, help="Override config.n_workers")
    p_run.add_argument("--start-page", type=int, default=0)
    p_run.add_argument("--end-page", type=int, default=None)
    p_run.add_argument("--sig-digits", type=int, default=4)
    p_run.add_argument("--image-name", default="")
    p_run.add_argument("--style", default=".tif")
    p_run.add_argument("--table-format", choices=["csv", "parquet"], default="csv")
    p_run.add_argument("--skip-rings", action="store_true")
    p_run.add_argument("--skip-chains", action="store_true")
    p_run.add_argument("--skip-fft", action="store_true")
    p_run.add_argument(
        "--export-mat", action="store_true", help="Also write results.mat for the MATLAB visualization scripts"
    )
    p_run.set_defaults(func=cmd_run)

    p_init = sub.add_parser("init-config", help="Write a default config file")
    p_init.add_argument("--output", default="stereom_config.json")
    p_init.set_defaults(func=cmd_init_config)

    p_regions = sub.add_parser(
        "extract-regions", help="Crop bounding-box regions out of STL mesh(es) into pipeline-ready volumes"
    )
    p_regions.add_argument("--spec", default=None, help="JSON file listing one or more regions (see RUN_INSTRUCTIONS.md)")
    p_regions.add_argument("--stl", default=None, help="Single-region mode: path to the STL file")
    p_regions.add_argument(
        "--bbox",
        type=float,
        nargs=6,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Single-region mode: bounding box in the STL file's own coordinate units",
    )
    p_regions.add_argument("--name", default=None, help="Single-region mode: output name (writes {name}.npz)")
    p_regions.add_argument(
        "--pitch", type=float, default=None, help="Single-region mode: voxel edge length, in STL file units"
    )
    p_regions.add_argument(
        "--stl-units-to-um",
        type=float,
        default=1000.0,
        help="Single-region mode: scale factor from STL units to micrometres (default 1000 = STL is in mm)",
    )
    p_regions.add_argument("--output-dir", required=True)
    p_regions.add_argument(
        "--n-jobs", type=int, default=1, help="Parallel workers across Z-slices during voxelization (-1 = all cores)"
    )
    p_regions.add_argument(
        "--no-export-stl", action="store_true", help="Skip writing the companion cropped .stl (for the MATLAB overlay)"
    )
    p_regions.set_defaults(func=cmd_extract_regions)

    p_mat = sub.add_parser("export-mat", help="Write results.mat for the MATLAB visualization scripts from an existing results directory")
    p_mat.add_argument("--results-dir", required=True, help="Directory previously written by `stereom run`")
    p_mat.add_argument(
        "--stl-alignment",
        default=None,
        help="Path to a {name}_stl_alignment.json sidecar (written by `stereom extract-regions`) to include STL overlay data",
    )
    p_mat.set_defaults(func=cmd_export_mat)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
