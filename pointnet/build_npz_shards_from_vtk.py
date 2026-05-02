from pathlib import Path
import re
import numpy as np

try:
    import pyvista as pv
except Exception as e:
    raise RuntimeError("pyvista is required. Install with: pip install pyvista vtk") from e


def _default_output_dir(vtk_dir: Path) -> Path:
    return vtk_dir.parent / f"{vtk_dir.name}_npz"


def _read_xyz_from_vtk(vtk_path: Path) -> np.ndarray:
    mesh = pv.read(str(vtk_path))
    pts = np.asarray(mesh.points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Unexpected point shape {pts.shape} in {vtk_path}")
    return pts


def _canonical_geom_name(raw_name: str) -> str:
    parts = raw_name.split("_")
    last = parts[-1] if parts else raw_name
    if last.isdigit():
        return f"geom_{int(last):03d}"
    return raw_name


def _extract_case_id(raw_name: str) -> int | None:
    match = re.search(r"(\d+)$", raw_name)
    if not match:
        return None
    return int(match.group(1))


def _load_case_to_geom_map(case_data_path: str | Path) -> dict[int, str]:
    """
    Parse case_data.dat and return mapping: numeric case id -> canonical geom name.
    """
    path = Path(case_data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"case_data.dat not found: {path}")

    case_to_geom: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tokens = s.split()
            if len(tokens) < 2:
                continue
            case_name, geom_name = tokens[0], tokens[1]
            case_id = _extract_case_id(case_name)
            if case_id is None:
                raise ValueError(f"Invalid case name at {path}:{line_num}: {case_name!r}")
            case_to_geom[case_id] = _canonical_geom_name(geom_name)

    if not case_to_geom:
        raise RuntimeError(f"No case->geom mappings found in: {path}")
    return case_to_geom


def build_npz_shards_from_vtk_folder(
    vtk_folder: str | Path,
    output_folder: str | Path | None = None,
    case_data_path: str | Path | None = None,
) -> Path:
    """
    Convert all .vtk files in `vtk_folder` into one .npz shard per file.

    Parameters
    ----------
    vtk_folder:
        Directory containing input .vtk files.
    output_folder:
        Directory for output .npz files. If None, uses sibling folder
        <vtk_folder_name>_npz.
    case_data_path:
        Path to case_data.dat used to map case number -> geometry name.
        If None, defaults to <repo_root>/test/case_data.dat.

    Returns
    -------
    Path
        Output directory path.
    """
    vtk_dir = Path(vtk_folder).expanduser().resolve()
    if not vtk_dir.exists() or not vtk_dir.is_dir():
        raise FileNotFoundError(f"Input VTK folder not found: {vtk_dir}")

    resolved_case_data_path = (
        Path(case_data_path).expanduser().resolve()
        if case_data_path
        else Path(__file__).resolve().parents[2] / "test" / "case_data.dat"
    )
    case_to_geom = _load_case_to_geom_map(resolved_case_data_path)

    out_dir = Path(output_folder).expanduser().resolve() if output_folder else _default_output_dir(vtk_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vtk_files = sorted(vtk_dir.glob("*.vtk"))
    if not vtk_files:
        raise FileNotFoundError(f"No .vtk files found in: {vtk_dir}")

    converted = 0
    for vtk_file in vtk_files:
        source_name = vtk_file.stem
        case_id = _extract_case_id(source_name)
        if case_id is None:
            raise ValueError(f"Could not extract case number from VTK filename: {vtk_file.name}")
        if case_id not in case_to_geom:
            continue
        geo_name = case_to_geom[case_id]
        xyz = _read_xyz_from_vtk(vtk_file)

        if (xyz.shape)[-1] == 3:

            # Keep one shard per CFD case to avoid filename collisions when
            # multiple cases share the same geometry.
            npz_path = out_dir / f"{source_name}.npz"
            np.savez_compressed(
                npz_path,
                geo_name=np.array(geo_name),
                geom_name=np.array(geo_name),
                source_name=np.array(source_name),
                xyz=xyz,
            )
        else:
            print(xyz.shape)
        converted += 1

    print(f"Converted {converted} VTK files -> {out_dir}")
    return out_dir


VTK_INPUT_FOLDER = r"C:\Users\Eladio\Documents\Intro to Deep Learning\BlendedNet Dataset - Released\test\vtk"
OUTPUT_FOLDER = None  # e.g. r"C:\path\to\custom_output" or None for sibling folder
CASE_DATA_PATH = r"C:\Users\Eladio\Documents\Intro to Deep Learning\BlendedNet Dataset - Released\test\case_data.dat"


if __name__ == "__main__":
    build_npz_shards_from_vtk_folder(VTK_INPUT_FOLDER, OUTPUT_FOLDER, CASE_DATA_PATH)
