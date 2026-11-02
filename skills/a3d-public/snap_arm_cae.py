"""Opt-in strain check for a cantilever snap arm, solved with CalculiX.

Takes the STEP of one manufactured part and a small ``caeIntent``, solves the
arm on three meshes with cad-cae-copilot (Gmsh C3D10 + CalculiX), scales the
linear result to the requested catch height and compares the root strain with
the material's allowable strain. A result that is not a clean pass is a
*warning*: this check never blocks a commit.

The pure helpers below need only the standard library. The solver stack runs
in a separate Python (``A3D_CAE_PYTHON``) with a pinned cad-cae-copilot clone
(``A3D_CAE_REPO``); when either is unset the report says ``skipped``.

Limits: linear elastic, one load case, no fatigue or creep, no contact, FDM
anisotropy only through the declared derates. Not a certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "a3d-snap-arm-cae/v1"
PINNED_COMMIT = "b98b4d50aa147b897eae9a56a70684412d8d9c3d"
FACE_TOLERANCE_MM = 1e-3
CONVERGED_FINEST_PAIR_PCT = 2.0   # this check's own threshold, not a standard
MARGINAL_FRACTION = 0.8           # this check's own threshold, not a standard
MESH_FACTORS = (0.4, 0.28, 0.2)   # mesh size = factor * arm thickness, coarse to fine

# Generic values. Moduli are typical-order numbers, NOT datasheet values: pass
# material.E_mpa from the filament datasheet when it matters. Allowable strains
# are guidance for printed parts, low end of the quoted range where one is given.
# ABS 7 %, PLA 4-8 %, nylon 4-15 %: Fictiv, "How to Design Snap Fit Components".
# Nylon 4 % is also the value BASF's Snap-Fit Design Manual uses for nylon 6 in
# the mating-force formula. PETG: no source found, so the user must supply it.
MATERIALS: dict[str, dict[str, Any]] = {
    "ABS": {"E_mpa": 2200.0, "nu": 0.35, "allowable_strain_pct": 7.0},
    "PLA": {"E_mpa": 3500.0, "nu": 0.36, "allowable_strain_pct": 4.0},
    "PETG": {"E_mpa": 2100.0, "nu": 0.38, "allowable_strain_pct": None},
    "NYLON": {"E_mpa": 1500.0, "nu": 0.40, "allowable_strain_pct": 4.0},
}
ACROSS_LAYERS_FACTOR = 0.5   # Fictiv: halve allowable stress/strain for Z-axis cantilevers
REPEATED_USE_FACTOR = 0.6    # Covestro snap-fit guide: 60 % of the values for frequent rejoining

LIMITATIONS = (
    "Linear elastic, one load case; no fatigue, creep, stress relaxation or contact.",
    "Peak strain is axial stress at the arm root divided by E; it is a screening number.",
    "FDM anisotropy is handled only through the declared derates.",
    "Moduli in the built-in table are generic, not datasheet values.",
    "Screening evidence, not certification or a production-safety claim.",
)


class SnapArmError(ValueError):
    """Raised when the intent or the geometry cannot be checked."""


# ---------------------------------------------------------------- pure helpers

def _vec(value: Any, name: str) -> tuple[float, float, float]:
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
        raise SnapArmError(f"{name} must be three finite numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def _unit(value: Any, name: str) -> tuple[float, float, float]:
    x, y, z = _vec(value, name)
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-9:
        raise SnapArmError(f"{name} must not be the zero vector")
    return (x / n, y / n, z / n)


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise SnapArmError(f"{name} must be a positive number")
    return float(value)


def validate_intent(intent: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalised copy of the intent or raise SnapArmError."""
    if not isinstance(intent, Mapping):
        raise SnapArmError("intent must be an object")
    arm = intent.get("arm")
    if not isinstance(arm, Mapping):
        raise SnapArmError("arm must be an object")
    out = {
        "part": str(intent.get("part") or ""),
        "fixed_point": _vec(intent.get("fixed_point"), "fixed_point"),
        "load_point": _vec(intent.get("load_point"), "load_point"),
        "load_direction": _unit(intent.get("load_direction"), "load_direction"),
        "catch_mm": _positive(intent.get("catch_mm"), "catch_mm"),
        "arm": {
            "axis": _unit(arm.get("axis"), "arm.axis"),
            "root_point": _vec(arm.get("root_point"), "arm.root_point"),
            "thickness_mm": _positive(arm.get("thickness_mm"), "arm.thickness_mm"),
            "width_mm": _positive(arm.get("width_mm"), "arm.width_mm"),
            "length_to_load_mm": _positive(arm.get("length_to_load_mm"), "arm.length_to_load_mm"),
        },
        "material": intent.get("material"),
        "arm_across_layers": bool(intent.get("arm_across_layers", False)),
        "repeated_use": bool(intent.get("repeated_use", False)),
    }
    ax, ld = out["arm"]["axis"], out["load_direction"]
    if abs(sum(a * b for a, b in zip(ax, ld))) > 0.1:
        raise SnapArmError("load_direction must be perpendicular to arm.axis")
    sizes = intent.get("mesh_sizes_mm")
    if sizes is not None:
        if not isinstance(sizes, Sequence) or len(sizes) != 3:
            raise SnapArmError("mesh_sizes_mm must list exactly three sizes")
        out["mesh_sizes_mm"] = [_positive(s, "mesh_sizes_mm") for s in sizes]
        if not (out["mesh_sizes_mm"][0] > out["mesh_sizes_mm"][1] > out["mesh_sizes_mm"][2]):
            raise SnapArmError("mesh_sizes_mm must go from coarse to fine")
    else:
        out["mesh_sizes_mm"] = [round(f * out["arm"]["thickness_mm"], 4) for f in MESH_FACTORS]
    return out


def resolve_material(spec: Any, *, arm_across_layers: bool = False, repeated_use: bool = False) -> dict[str, Any]:
    """Resolve a table name or an override object into E, nu and the derated allowable strain."""
    if isinstance(spec, str):
        base = MATERIALS.get(spec.strip().upper())
        if base is None:
            raise SnapArmError(f"unknown material {spec!r}; known: {sorted(MATERIALS)} or pass an object")
        name, values, origin = spec.strip().upper(), dict(base), "table"
    elif isinstance(spec, Mapping):
        name = str(spec.get("name") or "CUSTOM")
        values = dict(MATERIALS.get(name.upper(), {}))
        for key in ("E_mpa", "nu", "allowable_strain_pct"):
            if key in spec:
                values[key] = spec[key]
        origin = "override" if name.upper() not in MATERIALS else "table+override"
    else:
        raise SnapArmError("material must be a table name or an object")
    e, nu = values.get("E_mpa"), values.get("nu")
    _positive(e, "material.E_mpa")
    if isinstance(nu, bool) or not isinstance(nu, (int, float)) or not 0 <= nu < 0.5:
        raise SnapArmError("material.nu must be in [0, 0.5)")
    allowable = values.get("allowable_strain_pct")
    notes: list[str] = []
    derates: list[dict[str, Any]] = []
    if allowable is not None:
        allowable = _positive(allowable, "material.allowable_strain_pct")
        for flag, factor, label in ((arm_across_layers, ACROSS_LAYERS_FACTOR, "arm_across_layers"),
                                    (repeated_use, REPEATED_USE_FACTOR, "repeated_use")):
            if flag:
                allowable *= factor
                derates.append({"reason": label, "factor": factor})
    else:
        notes.append(f"{name} has no allowable strain in the table; pass material.allowable_strain_pct")
    if origin != "override" and "E_mpa" not in (spec if isinstance(spec, Mapping) else {}):
        notes.append("E_mpa is a generic typical value, not a datasheet value")
    return {"name": name, "origin": origin, "E_mpa": float(e), "nu": float(nu),
            "allowable_strain_pct": allowable, "derates": derates, "notes": notes}


def hand_calc(e_mpa: float, arm: Mapping[str, Any], catch_mm: float) -> dict[str, float]:
    """Euler-Bernoulli cantilever with a rigid root: force and root strain for a tip deflection."""
    t, w, a = arm["thickness_mm"], arm["width_mm"], arm["length_to_load_mm"]
    stiffness = e_mpa * w * t**3 / (4 * a**3)            # N/mm
    return {"stiffness_n_per_mm": stiffness, "force_n": stiffness * catch_mm,
            "root_strain_pct": 1.5 * t * catch_mm / a**2 * 100}


def face_at(faces: Sequence[Mapping[str, Any]], point: Sequence[float], tol: float = FACE_TOLERANCE_MM) -> Mapping[str, Any]:
    """The single planar face whose plane contains the point and whose box covers it."""
    hits = []
    for face in faces:
        if face.get("surface_type") != "plane":
            continue
        n, c, bb = face["normal"], face["center"], face["bounding_box"]
        if abs(sum(n[i] * (point[i] - c[i]) for i in range(3))) > tol:
            continue
        if all(bb[i] - tol <= point[i] <= bb[i + 3] + tol for i in range(3)):
            hits.append(face)
    if len(hits) != 1:
        raise SnapArmError(f"point {tuple(point)} matches {len(hits)} planar faces {[h['id'] for h in hits]}; "
                           "it must lie on exactly one")
    return hits[0]


def axial_stress(tensor: Sequence[float], axis: Sequence[float]) -> float:
    """axis^T S axis for a CalculiX stress record [sxx, syy, szz, sxy, syz, szx]."""
    sxx, syy, szz, sxy, syz, szx = tensor[:6]
    x, y, z = axis
    return (sxx * x * x + syy * y * y + szz * z * z + 2 * sxy * x * y + 2 * syz * y * z + 2 * szx * z * x)


def root_region_nodes(nodes: Mapping[Any, Sequence[float]], arm: Mapping[str, Any], load_dir: Sequence[float]) -> list[Any]:
    """Nodes within 0.6 mm behind to 1.5 mm ahead of the root plane, near the arm's thickness band."""
    axis, root = arm["axis"], arm["root_point"]
    width_dir = (axis[1] * load_dir[2] - axis[2] * load_dir[1],
                 axis[2] * load_dir[0] - axis[0] * load_dir[2],
                 axis[0] * load_dir[1] - axis[1] * load_dir[0])
    half_t, half_w = arm["thickness_mm"] / 2 + 1.0, arm["width_mm"] / 2 + 0.05
    picked = []
    for key, q in nodes.items():
        d = (q[0] - root[0], q[1] - root[1], q[2] - root[2])
        s = sum(d[i] * axis[i] for i in range(3))
        h = sum(d[i] * load_dir[i] for i in range(3))
        w = sum(d[i] * width_dir[i] for i in range(3))
        if -0.6 <= s <= 1.5 and abs(h) <= half_t and abs(w) <= half_w:
            picked.append(key)
    return picked


def convergence(values_coarse_to_fine: Sequence[float], sizes_coarse_to_fine: Sequence[float]) -> dict[str, Any]:
    """Finest-pair change, plus the ASME V&V-20 / Celik three-grid GCI when convergence is monotone."""
    f3, f2, f1 = values_coarse_to_fine
    h3, h2, h1 = sizes_coarse_to_fine
    change = abs(f1 - f2) / abs(f1) * 100 if f1 else float("inf")
    out: dict[str, Any] = {"values": [f3, f2, f1], "finest_pair_change_pct": change,
                           "converged": change <= CONVERGED_FINEST_PAIR_PCT, "gci_fine_pct": None, "order": None}
    e21, e32 = f2 - f1, f3 - f2
    if e21 != 0 and e32 != 0 and e32 / e21 > 0:
        r = ((h2 / h1) * (h3 / h2)) ** 0.5
        p = abs(math.log(abs(e32 / e21))) / math.log(r)
        if p > 0:
            out["order"] = p
            out["gci_fine_pct"] = 1.25 * abs((f1 - f2) / f1) / (r**p - 1) * 100
    return out


def judge(*, strain_pct: float, material: Mapping[str, Any], bands: Sequence[Any], conv: Mapping[str, Any]) -> dict[str, Any]:
    """Status is ``pass`` or ``warn``; a warning never blocks."""
    reasons: list[str] = []
    allowable = material["allowable_strain_pct"]
    ratio = None
    if allowable is None:
        reasons.append("needs_allowable: no allowable strain for this material")
    else:
        ratio = strain_pct / allowable
        if ratio > 1:
            reasons.append(f"exceeds_allowable: {strain_pct:.2f} % > {allowable:.2f} %")
        elif ratio > MARGINAL_FRACTION:
            reasons.append(f"marginal: {ratio * 100:.0f} % of the allowable strain")
    if any(b != "reliable" for b in bands):
        reasons.append(f"mesh_band: {sorted({str(b) for b in bands})}")
    if not conv["converged"]:
        reasons.append(f"not_converged: finest pair differs {conv['finest_pair_change_pct']:.1f} %")
    return {"status": "pass" if not reasons else "warn", "blocking": False,
            "strain_over_allowable": ratio, "reasons": reasons}


def repair_hints(strain_pct: float, material: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    """First-order changes that would bring the root strain to MARGINAL_FRACTION of the allowable.

    Strain at the root scales with thickness / length^2 / and linearly with the catch, so these are the
    beam-formula scalings applied to the FE result. They are starting points: re-run the check to confirm.
    """
    allowable = material["allowable_strain_pct"]
    if allowable is None or strain_pct <= 0:
        return {}
    target = allowable * MARGINAL_FRACTION
    if strain_pct <= target:
        return {}
    arm = spec["arm"]
    return {
        "target_strain_pct": round(target, 3),
        "max_catch_mm": round(spec["catch_mm"] * target / strain_pct, 3),
        "arm_thickness_mm": round(arm["thickness_mm"] * target / strain_pct, 3),
        "length_to_load_mm": round(arm["length_to_load_mm"] * math.sqrt(strain_pct / target), 3),
        "basis": "first-order beam scaling of the FE result: strain ~ catch, ~ thickness, ~ 1/length^2; "
                 "a thinner or longer arm is also softer; re-run to confirm",
    }


def summarize(report: Mapping[str, Any]) -> str:
    """One line for the agent: status, the measured numbers, and the reasons."""
    status = str(report.get("status"))
    if status in {"skipped", "error"}:
        return f"snap-arm check {status}: " + "; ".join(report.get("reasons") or [])
    result = report["result"]
    allowable = result.get("allowable_strain_pct")
    text = (f"snap-arm check {status}: root strain {result['root_peak_strain_pct']:.2f} % at a "
            f"{result['catch_mm']:g} mm catch (allowable "
            f"{'unknown' if allowable is None else f'{allowable:.2f} %'}), push force {result['force_n']:.2f} N")
    if report.get("reasons"):
        text += "; " + "; ".join(report["reasons"])
    hints = report.get("repairHints") or {}
    if hints:
        text += (f". To reach {hints['target_strain_pct']:g} %: catch <= {hints['max_catch_mm']:g} mm, or arm "
                 f"thickness {hints['arm_thickness_mm']:g} mm, or length to load {hints['length_to_load_mm']:g} mm")
    return text


def ensure_inside(root: Path, *paths: Path) -> None:
    """Every path must resolve (symlinks included) inside root."""
    base = root.resolve()
    for path in paths:
        resolved = path.resolve()
        if resolved != base and base not in resolved.parents:
            raise SnapArmError(f"{path} is outside the session workspace")


def skipped_report(reason: str) -> dict[str, Any]:
    return {"schema": SCHEMA, "status": "skipped", "blocking": False, "reasons": [reason],
            "limitations": list(LIMITATIONS)}


# ---------------------------------------------------------------- solver stack

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_step_code(path: Path) -> str:
    """build123d source that imports one STEP; the path is embedded as an escaped literal (repr)."""
    return ("from build123d import *\n"
            f"part = import_step({str(path)!r})\n"
            "part.label = 'part'\nresult = Compound(children=[part])\n")


def run_study(step_path: Path, intent: Mapping[str, Any], out_dir: Path, repo: Path) -> dict[str, Any]:
    """Solve on three meshes and build the report. Needs the CAE environment."""
    import shutil
    import tempfile
    import zipfile

    spec = validate_intent(intent)
    material = resolve_material(spec["material"], arm_across_layers=spec["arm_across_layers"],
                                repeated_use=spec["repeated_use"])
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()

    sys.path.insert(0, str(repo / "aieng-ui" / "backend"))
    previous = Path.cwd()
    os.chdir(work)  # the tool keeps a CWD-relative cache
    try:
        from app import runtime
        from app.app_factory import create_app
        from app.config import Settings
        from app.main import default_project, save_project
        from app.project_io import project_dir
        from aieng.simulation.cae_setup_writer import authored_setup_document
        from aieng.simulation.frd_result_extractor import parse_frd_steps
        from aieng.simulation.surface_loads import parse_mesh

        settings = Settings(platform_root=work / "platform", workspace_root=work / "workspace",
                            data_root=work / "data", aieng_root=repo / "aieng",
                            sample_step=work / "workspace" / "sample.step")
        create_app(settings)
        tool = runtime.invoke_tool

        pid = save_project(settings, default_project("snap_arm"))["id"]
        staged = work / "input.step"          # fixed name: the caller's path never reaches the generated code
        shutil.copyfile(step_path, staged)
        built = tool("cad.execute_build123d", {"project_id": pid, "timeout": 300, "thumbnail": False,
                                               "code": import_step_code(staged)})
        if built.get("status") != "ok":
            raise SnapArmError(f"STEP import failed: {json.dumps(built)[:400]}")
        pkg = project_dir(settings, pid) / f"{pid}.aieng"
        topology = json.loads(zipfile.ZipFile(pkg).read("geometry/topology_map.json"))
        faces = [e for e in topology["entities"] if e.get("type") == "face"]
        fixed, load = face_at(faces, spec["fixed_point"]), face_at(faces, spec["load_point"])
        if fixed["id"] == load["id"]:
            raise SnapArmError("fixed_point and load_point lie on the same face")

        patches = [
            {"action_type": "create_file", "path": "simulation/solver_settings.json",
             "content": {"solver": "CalculiX", "analysis_type": "static"}},
            {"action_type": "create_file", "path": "simulation/cae_imports/parsed_materials.json",
             "content": authored_setup_document("materials", [
                 {"name": material["name"], "elastic": {"youngs_modulus": material["E_mpa"],
                                                        "poisson_ratio": material["nu"]}}],
                 authored_by="snap_arm_cae")},
            {"action_type": "create_file", "path": "simulation/cae_imports/parsed_boundary_conditions.json",
             "content": authored_setup_document("boundary_conditions", [
                 {"id": "bc_fixed", "type": "fixed", "target": f"@face:{fixed['id']}",
                  "dof_start": 1, "dof_end": 3, "value": 0}], authored_by="snap_arm_cae")},
            {"action_type": "create_file", "path": "simulation/cae_imports/parsed_loads.json",
             "content": authored_setup_document("loads", [
                 {"id": "push", "type": "force", "target": f"@face:{load['id']}", "value_n": 1.0,
                  "direction": list(spec["load_direction"])}], authored_by="snap_arm_cae")},
        ]
        applied = tool("cae.apply_setup_patch", {"project_id": pid, "patches": patches})
        if applied.get("status") != "ok":
            raise SnapArmError(f"CAE setup failed: {json.dumps(applied)[:400]}")

        ld, lb = spec["load_direction"], load["bounding_box"]
        rows: list[dict[str, Any]] = []
        for index, size in enumerate(spec["mesh_sizes_mm"]):
            run_id = f"run_{index + 1:03d}"
            meshed = tool("cae.generate_mesh", {"project_id": pid, "mesh_size_mm": size})
            deck = tool("cae.generate_solver_input", {"project_id": pid, "run_id": run_id})
            solved = tool("cae.run_solver", {"project_id": pid, "timeout_seconds": 900,
                                             "input_deck_path": f"simulation/runs/{run_id}/solver_input.inp"})
            if not (meshed.get("status") == "completed" and deck.get("status") == "completed"
                    and solved.get("status") == "completed"):
                raise SnapArmError(f"{run_id} failed: {solved.get('errors') or solved.get('diagnosis') or deck.get('warnings')}")
            with zipfile.ZipFile(pkg) as zf:
                frd_name = next(n for n in zf.namelist()
                                if n.startswith(f"simulation/runs/{run_id}/") and n.endswith(".frd"))
                frd = Path(tempfile.mkdtemp()) / "result.frd"
                frd.write_bytes(zf.read(frd_name))
                nodes = parse_mesh(zf.read("simulation/mesh/mesh.inp").decode())[0]
            fields = {k: v[0]["node_data"] for k, v in parse_frd_steps(frd).items()}
            stress = fields.get("STRESS") or fields["S"]
            on_load = [k for k, q in nodes.items()
                       if abs(sum(load["normal"][i] * (q[i] - load["center"][i]) for i in range(3))) < 1e-4
                       and all(lb[i] - 1e-6 <= q[i] <= lb[i + 3] + 1e-6 for i in range(3))]
            if not on_load:
                raise SnapArmError("no mesh nodes found on the load face")
            defl = sum(abs(sum(fields["DISP"][k][i] * ld[i] for i in range(3))) for k in on_load) / len(on_load)
            region = root_region_nodes(nodes, spec["arm"], ld)
            if not region:
                raise SnapArmError("no mesh nodes found in the root region; check arm.root_point")
            peak = max(axial_stress(stress[k], spec["arm"]["axis"]) for k in region)
            scale = spec["catch_mm"] / defl
            rows.append({"mesh_mm": size, "nodes": len(nodes),
                         "accuracy_band": meshed.get("accuracy", {}).get("band"),
                         "deflection_at_1n_mm": defl, "force_for_catch_n": scale,
                         "root_peak_strain_pct": peak * scale / material["E_mpa"] * 100})

        shutil.copy(pkg, out_dir / "snap_arm_cae.aieng")
    finally:
        os.chdir(previous)
        shutil.rmtree(work, ignore_errors=True)   # keep only the report and the evidence package

    strain =[r["root_peak_strain_pct"] for r in rows]
    force = [r["force_for_catch_n"] for r in rows]
    sizes = spec["mesh_sizes_mm"]
    conv_strain, conv_force = convergence(strain, sizes), convergence(force, sizes)
    hand = hand_calc(material["E_mpa"], spec["arm"], spec["catch_mm"])
    verdict = judge(strain_pct=strain[-1], material=material,
                    bands=[r["accuracy_band"] for r in rows], conv=conv_strain)
    return {
        "schema": SCHEMA, **verdict,
        "inputs": {"step": str(step_path), "step_sha256": _sha256(step_path), "intent": spec,
                   "fixed_face": fixed["id"], "load_face": load["id"]},
        "material": material,
        "result": {"catch_mm": spec["catch_mm"], "force_n": force[-1], "root_peak_strain_pct": strain[-1],
                   "allowable_strain_pct": material["allowable_strain_pct"]},
        "meshes": rows,
        "convergence": {"root_peak_strain": conv_strain, "force_for_catch": conv_force},
        "hand_calc": {**hand, "note": "rigid root, no fillet, no hook; the FE arm is expected to be softer and "
                                      "to peak higher",
                      "fe_force_over_hand": force[-1] / hand["force_n"],
                      "fe_strain_over_hand": strain[-1] / hand["root_strain_pct"]},
        "repairHints": repair_hints(strain[-1], material, spec) if verdict["status"] == "warn" else {},
        "versions": versions(repo),
        "limitations": list(LIMITATIONS),
    }


def versions(repo: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        text = subprocess.run([os.environ.get("AIENG_CCX_CMD", "ccx"), "-v"], capture_output=True, text=True,
                              timeout=30).stdout
        out["calculix"] = next((l.strip() for l in text.splitlines() if "Version" in l), text.strip())
    except (OSError, subprocess.SubprocessError):
        out["calculix"] = None
    for module in ("gmsh", "build123d"):
        try:
            out[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001
            out[module] = None
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out["cad_cae_copilot_commit"] = commit or None
    out["cad_cae_copilot_pinned_commit"] = PINNED_COMMIT
    out["cad_cae_copilot_matches_pin"] = commit == PINNED_COMMIT
    return out


# ------------------------------------------------------------------------ CLI

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--step", required=True, type=Path)
    parser.add_argument("--intent", required=True, type=Path, help="caeIntent JSON")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--workspace", type=Path, help="session workspace; every path must stay inside it")
    args = parser.parse_args(argv)
    if args.workspace is not None:
        try:
            ensure_inside(args.workspace, args.step, args.intent, args.out)
        except SnapArmError as exc:
            print(json.dumps({"status": "error", "reasons": [str(exc)]}))
            return 2
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "cae-report.json"

    python, repo = os.environ.get("A3D_CAE_PYTHON"), os.environ.get("A3D_CAE_REPO")
    if not python or not repo:
        report = skipped_report("A3D_CAE_PYTHON and A3D_CAE_REPO are not both set")
    elif not Path(python).exists() or not (Path(repo) / "aieng-ui" / "backend").is_dir():
        report = skipped_report("A3D_CAE_PYTHON or A3D_CAE_REPO does not point at a CAE environment")
    elif os.environ.get("A3D_SNAP_CAE_CHILD") != "1":
        env = {**os.environ, "A3D_SNAP_CAE_CHILD": "1", "AIENG_CCX_CMD": os.environ.get("AIENG_CCX_CMD", "ccx"),
               "PATH": f"{Path(python).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        done = subprocess.run([python, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
                              env=env)
        return done.returncode
    else:
        try:
            report = run_study(args.step.resolve(), json.loads(args.intent.read_text()), args.out.resolve(), Path(repo))
        except (SnapArmError, ValueError, OSError) as exc:    # includes unreadable or malformed intent JSON
            report = {"schema": SCHEMA, "status": "error", "blocking": False, "reasons": [str(exc)],
                      "limitations": list(LIMITATIONS)}
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"status": report["status"], "summary": summarize(report), "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
