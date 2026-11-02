# Snap-arm strain check (optional, advisory)

`a3d snapcheck` solves one cantilever snap arm with CalculiX and compares the
root strain with the material's allowable strain. Use it after the geometry is
stable and a `snap-fit` interface exists. It is not part of `a3d compile`, it
never blocks, and a missing solver stack gives `skipped`, not a failure.

## Run it

```bash
a3d snapcheck --step clip.step --intent cae_intent.json --out snapcheck
```

`--step` is the STEP of the one part that carries the arm (the per-part STEP
from `a3d draft` or `a3d compile`, not the whole assembly). All three paths must
stay inside the session workspace. The command prints one JSON line with
`status`, a one-sentence `summary` and the report path. Read the summary first;
open `snapcheck/cae-report.json` for the numbers.

Runtime: about one minute for three meshes. The output folder keeps the report and
`snap_arm_cae.aieng` (about 16 MB, the CalculiX evidence package).

## caeIntent

Coordinates are millimetres in the part's own STEP frame.

| Field | Meaning |
|---|---|
| `material` | `"ABS"`, `"PLA"`, `"PETG"`, `"NYLON"`, or `{name, E_mpa, nu, allowable_strain_pct}` |
| `fixed_point` | A point ON the face that is held fixed (the face behind the arm root) |
| `load_point` | A point ON the face that is pushed (the hook's flat top) |
| `load_direction` | Direction of the push, perpendicular to the arm axis |
| `catch_mm` | How far the load face must move (the hook undercut) |
| `arm.axis` | Unit direction from root to tip |
| `arm.root_point` | A point at the middle of the arm root |
| `arm.thickness_mm`, `arm.width_mm` | Arm section |
| `arm.length_to_load_mm` | Root to the load point, along the axis |
| `arm_across_layers` | `true` if the arm bends across print layers (halves the allowable) |
| `repeated_use` | `true` if the clip is opened and closed often (60 % of the allowable) |
| `mesh_sizes_mm` | Optional, coarse to fine. Default: 0.4, 0.28 and 0.2 times the arm thickness |

Feature IDs such as `clip-hook` do not survive in the STEP, so faces are chosen
by point. Each point must lie on exactly one planar face. The check stops with an
error when it matches none or several.

## Reading the report

- `status`: `pass`, `warn`, `skipped` or `error`. `blocking` is always `false`.
- `result.root_peak_strain_pct` against `result.allowable_strain_pct`.
- `reasons` (a warning when any is present):
  - `exceeds_allowable`: strain is above the allowable.
  - `marginal`: strain is above 80 % of the allowable (this check's own threshold).
  - `needs_allowable`: the material has no allowable strain. Pass one.
  - `mesh_band`: a mesh was not rated `reliable` by cad-cae-copilot.
  - `not_converged`: the two finest meshes differ by more than 2 % (this check's own threshold).
- `repairHints`: first-order catch height, arm thickness or length that would bring
  the strain to 80 % of the allowable. Change one, rebuild and run the check again.
- `hand_calc`: rigid-root beam formulas. The FE arm is normally softer and peaks higher
  because the wall flexes and the fillet concentrates strain. Do not expect them to match.
- `versions`: CalculiX, Gmsh, build123d and the cad-cae-copilot commit. Keep these with the design.

Strain is read at the arm root, so a sharp inside corner never converges. Model the
root fillet, or read `not_converged` as a request for one.

## Materials

| Name | Allowable strain | Source |
|---|---|---|
| ABS | 7 % | Fictiv, "How to Design Snap Fit Components" (printed ABS) |
| PLA | 4 % | Same article, low end of 4 to 8 % |
| NYLON | 4 % | Same article, low end of 4 to 15 %. BASF Snap-Fit Design Manual also uses 4 % for nylon 6 |
| PETG | none | No source found. Pass `allowable_strain_pct` |

The modulus in the table is a typical value, not a datasheet value. Pass `E_mpa`
from the filament datasheet when the result matters. Derates: 0.5 for an arm that
bends across layers (Fictiv), 0.6 for frequent re-joining (Covestro, "Snap-Fit
Joints for Plastics"). Injection-molded limits from BASF (ABS 6 to 7 %) are not
the same as printed limits.

## Set up the solver stack

The check runs in a separate Python with CalculiX and Gmsh. Nothing is installed
into the Amagine3D environment.

```bash
brew install micromamba
micromamba create -y -n cae -c conda-forge python=3.11 calculix=2.23 gmsh python-gmsh
git clone https://github.com/armpro24-blip/cad-cae-copilot && cd cad-cae-copilot
git checkout b98b4d50aa147b897eae9a56a70684412d8d9c3d
micromamba run -n cae pip install -e './aieng[dev]' -e './aieng-ui/backend[dev,cad,verification]'
export A3D_CAE_PYTHON=$(micromamba run -n cae python -c 'import sys; print(sys.executable)')
export A3D_CAE_REPO=$PWD
```

cad-cae-copilot is alpha software. The commit above is the one this check was
verified against; the report records `cad_cae_copilot_matches_pin`.

Why not its own `cae.mesh_convergence` tool: at that commit it re-reads the material
through a path that ignores the nested `elastic` form and silently uses aluminium
(armpro24-blip/cad-cae-copilot#563). This check computes convergence from its own solves.

## Limits

Linear elastic, one load case. No fatigue, creep, stress relaxation or contact. The
peak strain is axial stress divided by E, a screening number. Printed-part anisotropy
is handled only through the two derates above. The result is review evidence, not
certification or a production-safety claim.
