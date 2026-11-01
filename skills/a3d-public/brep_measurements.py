"""Measure final BRep sections, independently of source construction parameters.

``measure_section(part, Plane.XY.offset(95))`` measures the actual z=95 section.
Its width/depth are along the plane's explicit u/v directions; its center is the
outer bounds midpoint, not the material centroid. Compare parallel sections in
the same frame to inspect mouth/foot widths and center displacement. No section
is selected implicitly, and these measurements do not establish global minima.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from uuid import uuid4

from build123d import Face, Plane, Shape, import_step, section
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps


_NOTES = [
    "Coordinates are the input part coordinates; no print-pose transform is applied.",
    "BRep intersections and bounds retain kernel precision; areas use adaptive integration with relative target 1e-9, not mesh sampling.",
    "Width u and depth v follow the reported plane axes; center is the outer bounds midpoint, not a centroid.",
    "Material islands are section faces per solid; holes are their bounded inner loops, not proven through passages.",
    "Areas are summed per solid without unioning overlaps; overlapping solids can double count material.",
    "Empty means no positive-area section returned; a tangent plane may still touch an edge or point. Inspect nearby specified planes when needed.",
    "Finite sections do not prove global wall thickness, connectivity, or satisfaction of unsampled requirements.",
]
_SUMMARY_SECTION_LIMIT = 8
_LENGTH_REPORT_RESOLUTION_MM = 0.01


def _xyz(value) -> list[float]:
    return [float(v) for v in value]


def _bounds(shape: Shape) -> dict:
    # Analytic BRep bounds, without using or clearing a cached triangulation and
    # without inflating by stored shape tolerances. Kernel precision still applies.
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped, box, False, False)
    if box.IsVoid():
        raise ValueError("cannot bound an empty BRep")
    values = box.Get()
    low, high = list(values[:3]), list(values[3:])
    return {"min": low, "max": high, "size": [b - a for a, b in zip(low, high)]}


def _envelope(bounds: dict, plane: Plane) -> dict:
    low, high = bounds["min"][:2], bounds["max"][:2]
    center = [(a + b) / 2 for a, b in zip(low, high)]
    return {
        "min_uv_mm": low, "max_uv_mm": high,
        "width_u_mm": high[0] - low[0], "depth_v_mm": high[1] - low[1],
        "center_uv_mm": center,
        "center_world_mm": _xyz(plane.from_local_coords((*center, 0))),
    }


def _surface_properties(face: Face) -> tuple[float, list[float], float]:
    # Default non-adaptive quadrature can lose area on trimmed B-spline curves.
    properties = GProp_GProps()
    error = BRepGProp.SurfaceProperties_s(face.wrapped, properties, 1e-9, False)
    return float(properties.Mass()), list(properties.CentreOfMass().Coord()), float(error)


def _boundary(wire, plane: Plane) -> dict:
    if not wire.is_closed:
        raise ValueError("section returned an open boundary for a material face")
    area, _, error = _surface_properties(Face(wire))
    return {
        "envelope": _envelope(_bounds(plane.to_local_coords(wire)), plane),
        "enclosed_area_mm2": area, "relative_area_error_estimate": error,
        "perimeter_mm": float(wire.length),
    }


def _solids(shape: Shape) -> list:
    if not isinstance(shape, Shape) or not shape.is_valid:
        raise ValueError("measurement requires a valid BRep containing solids")
    solids = list(shape.solids())
    if not solids:
        raise ValueError("measurement requires at least one solid")
    return solids


def _component_solids(shape: Shape) -> tuple[list, int]:
    """Valid solids of a maker assembly, checked one by one, and the excluded count.

    Supplier STEPs often carry loose surfaces (labels, silkscreen) or a defective
    solid that invalidate the whole compound. Those never become material here.
    """
    if not isinstance(shape, Shape):
        raise ValueError("component measurement requires a BRep")
    solids = list(shape.solids())
    valid = [solid for solid in solids if solid.is_valid]
    if not valid:
        raise ValueError("component measurement found no valid solids")
    return valid, len(solids) - len(valid)


def _crosses(solid: Shape, plane: Plane) -> bool:
    box = _bounds(solid)
    heights = [plane.to_local_coords((x, y, z)).Z
               for x in (box["min"][0], box["max"][0])
               for y in (box["min"][1], box["max"][1])
               for z in (box["min"][2], box["max"][2])]
    return min(heights) <= 0 <= max(heights)


def _section_preview(faces: list, plane: Plane, path: Path, label: str | None) -> dict:
    """Render the already measured faces; display sampling never feeds measurements."""
    from PIL import Image, ImageDraw
    polygons = []
    for face in faces:
        boundaries = []
        for wire in [face.outer_wire(), *face.inner_wires()]:
            points = []
            for edge in wire.order_edges():
                # Arclength-sampled outlines are illustrative, not acceptance evidence.
                count = max(8, min(256, math.ceil(float(edge.length) * 2)))
                points.extend((point.X, point.Y) for point in
                              (plane.to_local_coords(edge.position_at(i / count)) for i in range(count + 1)))
            boundaries.append(points)
        polygons.append(boundaries)
    points = [point for boundaries in polygons for loop in boundaries for point in loop]
    low = [min(point[i] for point in points) for i in range(2)] if points else [0, 0]
    high = [max(point[i] for point in points) for i in range(2)] if points else [1, 1]
    scale = min(720 / max(high[0] - low[0], 1e-6), 520 / max(high[1] - low[1], 1e-6))
    def pixel(point):
        return (40 + (point[0] - low[0]) * scale, 580 - (point[1] - low[1]) * scale)
    image = Image.new("RGB", (800, 640), "#f4f6f8")
    for boundaries in polygons:
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        for index, loop in enumerate(boundaries):
            if len(loop) >= 3:
                draw.polygon([pixel(point) for point in loop], fill=255 if index == 0 else 0)
        image.paste("#527d9e", mask=mask)
    draw = ImageDraw.Draw(image)
    draw.text((24, 16), "Specified section - input-part coordinates - mm", fill="#172b3a")
    draw.text((24, 610), "u ->    v up; sampled display only; dimensions are in the bound report", fill="#172b3a")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return {"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest(),
            "scope": "same measured faces and plane; sampled outline display, not global thickness proof"}


def measure_section(shape: Shape, plane: Plane, *, label: str | None = None,
                    preview: str | Path | None = None, component: bool = False) -> dict:
    """Measure a specified plane in the final part's coordinates, in mm/mm².

    Each solid's positive-area section faces remain separate material islands.
    ``outer_envelope`` encloses all islands (including intervening empty space);
    it is null for an empty section. A hole belongs to its particular island.
    Coplanar end faces count as sections when returned by the kernel. No plane
    offset, tolerance-based pass/fail, source station, or mesh sampling is used.
    Optional preview writes an illustrative image from these same section faces.
    """
    if not isinstance(plane, Plane) or not all(
        math.isfinite(v) for axis in (plane.origin, plane.x_dir, plane.y_dir, plane.z_dir) for v in axis
    ):
        raise ValueError("plane must have finite origin and axes")
    # section() may clean its input's triangulation cache; isolate the caller.
    solids = _component_solids(deepcopy(shape))[0] if component else _solids(deepcopy(shape))
    islands = []
    measured_faces = []
    for solid_index, solid in enumerate(solids):
        if component and not _crosses(solid, plane):
            continue
        # build123d uses a finite cutting face. Recenter that face on the solid's
        # projection, so an equivalent plane with a distant in-plane origin
        # cannot miss the solid. Measurement coordinates still use the input plane.
        box = _bounds(solid)
        center = tuple((a + b) / 2 for a, b in zip(box["min"], box["max"]))
        local = plane.to_local_coords(center)
        cutting_plane = Plane(origin=plane.from_local_coords((local.X, local.Y, 0)),
                              x_dir=plane.x_dir, z_dir=plane.z_dir)
        cut = section(solid, section_by=cutting_plane)
        for face in cut.faces():
            area, centroid, error = _surface_properties(face)
            if area <= 0:
                continue
            measured_faces.append(face)
            islands.append({
                "solid_index": solid_index,
                "island_index": len(islands),
                "material_area_mm2": area,
                "material_centroid_world_mm": centroid,
                "relative_area_error_estimate": error,
                "outer": _boundary(face.outer_wire(), plane),
                "holes": [_boundary(wire, plane) for wire in face.inner_wires()],
            })
    envelope = None
    if islands:
        boxes = [island["outer"]["envelope"] for island in islands]
        envelope = _envelope({
            "min": [min(box["min_uv_mm"][i] for box in boxes) for i in range(2)],
            "max": [max(box["max_uv_mm"][i] for box in boxes) for i in range(2)],
        }, plane)
    result = {
        "schema": "brep-section/v1", "label": label,
        "units": {"length": "mm", "area": "mm2"},
        "coordinate_frame": "input-part",
        "plane": {"origin_mm": _xyz(plane.origin), "u_dir": _xyz(plane.x_dir),
                  "v_dir": _xyz(plane.y_dir), "normal_dir": _xyz(plane.z_dir)},
        "status": "material" if islands else "empty",
        "outer_envelope": envelope,
        "material_island_count": len(islands),
        "hole_count": sum(len(island["holes"]) for island in islands),
        "sum_material_area_mm2": sum(island["material_area_mm2"] for island in islands),
        "material_islands": islands,
        "notes": list(_NOTES),
    }
    if preview is not None:
        result["preview"] = _section_preview(measured_faces, plane, Path(preview), label)
    return result


def measure_step(path: str | Path, sections: Mapping[str, Plane] | None = None, *,
                 preview_dir: str | Path | None = None, component: bool = False) -> dict:
    """Read a final STEP; bind its SHA-256, world bounds and named exact sections.

    ``sections={"foot": Plane.XY, "mouth": Plane.XY.offset(95)}`` preserves the
    caller's chosen planes and labels. STEP is imported in millimetres. Python
    build source and intent parameters are never read or executed.
    ``component=True`` reads a maker's assembly STEP to locate features: solids are
    validated one by one and only valid solids are measured. It is never
    manufacturing acceptance for a printed part.
    """
    # The session sandbox can allow this file while denying metadata on its
    # ancestors. Match cad_compile's non-strict resolution, then check the file.
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"measurement input is not an existing file: {path}")
    if path.suffix.lower() not in {".step", ".stp"}:
        raise ValueError("measurement input must be a STEP/STP file")
    if sections is not None and not isinstance(sections, Mapping):
        raise ValueError("sections must map labels to Plane objects")
    input_hash = sha256(path.read_bytes()).hexdigest()
    shape = import_step(path)
    if component:
        solids, excluded = _component_solids(shape)
        boxes = [_bounds(solid) for solid in solids]
        low = [min(box["min"][i] for box in boxes) for i in range(3)]
        high = [max(box["max"][i] for box in boxes) for i in range(3)]
        bounds = {"min": low, "max": high, "size": [b - a for a, b in zip(low, high)]}
    else:
        solids, bounds = _solids(shape), _bounds(shape)
    result = {
        "schema": "brep-measurements/v1",
        "input": {"path": str(path), "sha256": input_hash, "format": "STEP"},
        "units": {"length": "mm", "area": "mm2", "volume": "mm3"},
        "coordinate_frame": "input-part",
        "world_bounds_mm": bounds, "solid_count": len(solids),
        "sum_solid_volume_mm3": sum(float(solid.volume) for solid in solids),
        "sections": [measure_section(shape, plane, label=label, component=component,
                     preview=Path(preview_dir) / f"section-{index}.png" if preview_dir is not None else None)
                     for index, (label, plane) in enumerate((sections or {}).items())],
        "notes": list(_NOTES),
    }
    if component:
        result["mode"] = "component"
        result["excluded_invalid_solid_count"] = excluded
        result["notes"].append(
            "Component mode: a maker assembly measured to locate features. Solids are validated one by "
            "one; loose surfaces and invalid solids are never material, and bounds cover valid solids "
            "only. This is not manufacturing acceptance for a printed part."
        )
    if sha256(path.read_bytes()).hexdigest() != input_hash:
        raise ValueError("STEP changed during measurement; result discarded")
    return result


def _length_report(result: dict) -> dict:
    """Project measured lengths for the CLI without altering geometry or planes."""
    report = deepcopy(result)

    def length(value):
        if isinstance(value, list):
            return [length(item) for item in value]
        return round(value, 2) or 0.0

    def envelope(value):
        if value is not None:
            for key in ("min_uv_mm", "max_uv_mm", "width_u_mm", "depth_v_mm",
                        "center_uv_mm", "center_world_mm"):
                value[key] = length(value[key])

    for key in ("min", "max", "size"):
        report["world_bounds_mm"][key] = length(report["world_bounds_mm"][key])
    for cut in report["sections"]:
        envelope(cut["outer_envelope"])
        for island in cut["material_islands"]:
            island["material_centroid_world_mm"] = length(island["material_centroid_world_mm"])
            for boundary in (island["outer"], *island["holes"]):
                envelope(boundary["envelope"])
                boundary["perimeter_mm"] = length(boundary["perimeter_mm"])
    report["length_report_resolution_mm"] = _LENGTH_REPORT_RESOLUTION_MM
    report["notes"].append(
        "Reported measured lengths use 0.01 mm resolution. Plane origins/directions and non-length "
        "quantities retain their precision; Python measurement helpers return unrounded values. "
        "Rounded readings are not canonical geometry or the raw values used for acceptance."
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Write and summarize final section sizes at 0.01 mm length resolution."""
    from cad_compile import _workspace_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    for axis in "xyz":
        parser.add_argument(f"--section-{axis}", action="append", type=float, default=[],
                            metavar="MM", help=f"Repeat for exact world {axis} coordinates in mm")
    parser.add_argument("--out", type=Path,
                        help="Full report path (default: <model-stem>_measurements.json in workspace)")
    parser.add_argument("--preview", action="store_true", help="Render only the explicitly requested sections")
    parser.add_argument("--component", action="store_true",
                        help="Maker assembly STEP: validate solids one by one; locates features, not acceptance")
    args = parser.parse_args(argv)
    try:
        workspace = args.workspace.resolve()
        model = _workspace_path(workspace, args.model, "model", must_exist=True)
        output = _workspace_path(workspace, args.out or Path(f"{model.stem}_measurements.json"),
                                 "output", must_exist=False)
        if output == model or (output.exists() and output.samefile(model)):
            raise ValueError("output must not overwrite the STEP input")
        planes = {}
        for axis in "xyz":
            for value in getattr(args, f"section_{axis}"):
                if not math.isfinite(value):
                    raise ValueError("section coordinates must be finite")
                origin = tuple(value if a == axis else 0 for a in "xyz")
                # y cuts use u=+X, v=+Z, normal=-Y for a right-handed frame.
                normal = {"x": (1, 0, 0), "y": (0, -1, 0), "z": (0, 0, 1)}[axis]
                u_dir = (0, 1, 0) if axis == "x" else (1, 0, 0)
                planes[f"{axis}={value!r}"] = Plane(origin=origin, x_dir=u_dir, z_dir=normal)
        if args.preview and not planes:
            raise ValueError("--preview needs at least one explicit section")
        preview_dir = (_workspace_path(workspace, Path(".amagine3d-measurements") / str(uuid4()),
                                        "section previews", must_exist=False) if args.preview else None)
        result = _length_report(measure_step(model, planes, preview_dir=preview_dir,
                                             component=args.component))
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        output.write_text(encoded, encoding="utf-8")
        fields = ("label", "plane", "status", "outer_envelope", "material_island_count",
                  "hole_count", "sum_material_area_mm2")
        summaries = [{key: cut[key] for key in (*fields, "preview") if key in cut}
                     for cut in result["sections"][:_SUMMARY_SECTION_LIMIT]]
        print(json.dumps({
            "schema": "brep-measurement-summary/v1", "input": result["input"],
            "fullResult": {"path": str(output), "sha256": sha256(encoded.encode("utf-8")).hexdigest()},
            "units": result["units"], "coordinate_frame": result["coordinate_frame"],
            "length_report_resolution_mm": result["length_report_resolution_mm"],
            "world_bounds_mm": result["world_bounds_mm"], "solid_count": result["solid_count"],
            **{key: result[key] for key in ("mode", "excluded_invalid_solid_count") if key in result},
            "sections": summaries, "section_count": len(result["sections"]),
            "returned_section_count": len(summaries),
            "omitted_section_count": len(result["sections"]) - len(summaries),
            "scope": "Specified sections only, not a global proof. Remaining sections, material islands, holes and limitations are in fullResult.",
        }, allow_nan=False))
    except (ValueError, OSError) as error:
        parser.exit(2, f"measurement error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
