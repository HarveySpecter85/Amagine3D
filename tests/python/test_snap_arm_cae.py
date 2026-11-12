from pathlib import Path
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SKILL = Path(__file__).resolve().parents[2] / "skills" / "a3d-public"
sys.path.insert(0, str(SKILL))
import snap_arm_cae as cae
from snap_arm_cae import SnapArmError

ARM = {"axis": [1, 0, 0], "root_point": [40, 3, 1], "thickness_mm": 2.0, "width_mm": 6.0, "length_to_load_mm": 17.5}


def intent(**changes):
    base = {
        "part": "clip", "material": "ABS", "fixed_point": [36, 3, 1], "load_point": [57.5, 3, 3],
        "load_direction": [0, 0, -1], "catch_mm": 1.0, "arm": dict(ARM),
    }
    base.update(changes)
    return base


def face(face_id, normal, center, bbox, surface="plane"):
    return {"id": face_id, "surface_type": surface, "normal": normal, "center": center, "bounding_box": bbox}


class MaterialTests(unittest.TestCase):
    def test_table_values_and_derates_multiply(self):
        self.assertEqual(cae.resolve_material("abs")["allowable_strain_pct"], 7.0)
        across = cae.resolve_material("ABS", arm_across_layers=True)
        self.assertAlmostEqual(across["allowable_strain_pct"], 3.5)
        both = cae.resolve_material("ABS", arm_across_layers=True, repeated_use=True)
        self.assertAlmostEqual(both["allowable_strain_pct"], 2.1)
        self.assertEqual([d["reason"] for d in both["derates"]], ["arm_across_layers", "repeated_use"])

    def test_petg_has_no_invented_allowable(self):
        petg = cae.resolve_material("PETG")
        self.assertIsNone(petg["allowable_strain_pct"])
        self.assertTrue(any("allowable" in note for note in petg["notes"]))
        self.assertEqual(cae.resolve_material({"name": "PETG", "allowable_strain_pct": 5})["allowable_strain_pct"], 5.0)

    def test_override_and_errors(self):
        custom = cae.resolve_material({"name": "MY", "E_mpa": 1800, "nu": 0.4, "allowable_strain_pct": 3})
        self.assertEqual((custom["E_mpa"], custom["origin"]), (1800.0, "override"))
        with self.assertRaises(SnapArmError):
            cae.resolve_material("UNOBTAINIUM")
        with self.assertRaises(SnapArmError):
            cae.resolve_material({"name": "MY", "nu": 0.4, "allowable_strain_pct": 3})  # no E


class IntentTests(unittest.TestCase):
    def test_defaults_scale_with_arm_thickness(self):
        spec = cae.validate_intent(intent())
        self.assertEqual(spec["mesh_sizes_mm"], [0.8, 0.56, 0.4])
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in spec["load_direction"])), 1.0)

    def test_rejects_bad_intent(self):
        for changes in ({"catch_mm": 0}, {"load_direction": [1, 0, 0]}, {"fixed_point": [1, 2]},
                        {"mesh_sizes_mm": [0.4, 0.8, 0.5]}, {"arm": {**ARM, "width_mm": -1}}):
            with self.subTest(changes=changes), self.assertRaises(SnapArmError):
                cae.validate_intent(intent(**changes))


class MathTests(unittest.TestCase):
    def test_hand_calc_matches_the_snap_fit_formulas(self):
        hand = cae.hand_calc(2200.0, ARM, 1.0)
        self.assertAlmostEqual(hand["force_n"], 4.9259, places=3)       # P = E w t^3 Y / (4 a^3)
        self.assertAlmostEqual(hand["root_strain_pct"], 0.97959, places=4)  # 1.5 t Y / a^2

    def test_axial_stress_projects_the_tensor(self):
        self.assertAlmostEqual(cae.axial_stress([5, 1, 2, 0, 0, 0], (1, 0, 0)), 5.0)
        s = 1 / math.sqrt(2)
        self.assertAlmostEqual(cae.axial_stress([0, 0, 0, 4, 0, 0], (s, s, 0)), 4.0)  # pure shear at 45 deg

    def test_convergence_flags_oscillation_and_reports_gci_when_monotone(self):
        sizes = [0.8, 0.56, 0.4]
        osc = cae.convergence([1.109, 1.209, 1.200], sizes)
        self.assertTrue(osc["converged"])
        self.assertIsNone(osc["gci_fine_pct"])
        mono = cae.convergence([1.0, 1.10, 1.12], sizes)
        self.assertGreater(mono["gci_fine_pct"], 0)
        self.assertFalse(cae.convergence([1.0, 1.2, 1.4], sizes)["converged"])


class FaceTests(unittest.TestCase):
    FACES = [
        face("wall", [-1, 0, 0], [36, 3, 1], [36, 0, -3, 36, 6, 5]),
        face("hook_top", [0, 0, 1], [57.5, 3, 3], [57, 0, 3, 58, 6, 3]),
        face("fillet", [0, 0, 1], [40, 3, 2], [39.5, 0, 1.5, 40.5, 6, 2.5], surface="cylinder"),
    ]

    def test_picks_the_only_face_containing_the_point(self):
        self.assertEqual(cae.face_at(self.FACES, [36, 3, 1])["id"], "wall")
        self.assertEqual(cae.face_at(self.FACES, [57.5, 3, 3])["id"], "hook_top")

    def test_rejects_no_match_and_ambiguity(self):
        with self.assertRaises(SnapArmError):
            cae.face_at(self.FACES, [50, 3, 1])
        with self.assertRaises(SnapArmError):
            cae.face_at(self.FACES + [face("twin", [-1, 0, 0], [36, 3, 1], [36, 0, -3, 36, 6, 5])], [36, 3, 1])
        with self.assertRaises(SnapArmError):
            cae.face_at(self.FACES, [40, 3, 2])  # only a curved face is there

    def test_root_region_keeps_nodes_near_the_root_and_thickness_band(self):
        nodes = {1: (40, 3, 1), 2: (41, 5.9, 2), 3: (45, 3, 1), 4: (40, 3, 6), 5: (39.5, 0.1, 0)}
        picked = cae.root_region_nodes(nodes, cae.validate_intent(intent())["arm"], (0, 0, -1))
        self.assertEqual(sorted(picked), [1, 2, 5])


class JudgeTests(unittest.TestCase):
    ABS = cae.resolve_material("ABS")
    OK = {"converged": True, "finest_pair_change_pct": 0.5}

    def judge(self, strain, material=None, bands=("reliable",) * 3, conv=None):
        return cae.judge(strain_pct=strain, material=material or self.ABS, bands=bands, conv=conv or self.OK)

    def test_pass_is_clean(self):
        verdict = self.judge(1.2)
        self.assertEqual((verdict["status"], verdict["reasons"], verdict["blocking"]), ("pass", [], False))

    def test_everything_else_warns_and_never_blocks(self):
        cases = {
            "marginal": self.judge(6.0),
            "exceeds_allowable": self.judge(7.5),
            "needs_allowable": self.judge(1.0, material=cae.resolve_material("PETG")),
            "mesh_band": self.judge(1.0, bands=("reliable", "marginal", "reliable")),
            "not_converged": self.judge(1.0, conv={"converged": False, "finest_pair_change_pct": 9.0}),
        }
        for reason, verdict in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual((verdict["status"], verdict["blocking"]), ("warn", False))
                self.assertTrue(any(r.startswith(reason) for r in verdict["reasons"]), verdict["reasons"])


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "snap_arm_cae_warn_report.json"


class StoredReportTests(unittest.TestCase):
    """Real CalculiX output stored once (PLA, across layers, repeated use): the verdict logic must reproduce it."""

    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(FIXTURE.read_text())

    def test_report_shape(self):
        r = self.report
        self.assertEqual((r["schema"], r["status"], r["blocking"]), (cae.SCHEMA, "warn", False))
        self.assertEqual(len(r["meshes"]), 3)
        self.assertTrue(r["versions"]["cad_cae_copilot_matches_pin"])
        self.assertEqual(len(r["inputs"]["step_sha256"]), 64)
        self.assertTrue(r["limitations"])

    def test_verdict_is_reproduced_from_the_stored_numbers(self):
        r = self.report
        material = cae.resolve_material("PLA", arm_across_layers=True, repeated_use=True)
        self.assertEqual(material["allowable_strain_pct"], r["material"]["allowable_strain_pct"])
        strain = [m["root_peak_strain_pct"] for m in r["meshes"]]
        conv = cae.convergence(strain, [m["mesh_mm"] for m in r["meshes"]])
        again = cae.judge(strain_pct=strain[-1], material=material,
                          bands=[m["accuracy_band"] for m in r["meshes"]], conv=conv)
        self.assertEqual((again["status"], again["reasons"]), (r["status"], r["reasons"]))

    def test_hand_calc_and_hints_are_reproduced(self):
        r = self.report
        spec = cae.validate_intent(r["inputs"]["intent"])
        hand = cae.hand_calc(r["material"]["E_mpa"], spec["arm"], spec["catch_mm"])
        self.assertAlmostEqual(hand["force_n"], r["hand_calc"]["force_n"], places=6)
        hints = cae.repair_hints(r["result"]["root_peak_strain_pct"], r["material"], spec)
        self.assertEqual(hints, r["repairHints"])

    def test_summary_names_the_measured_numbers_and_the_fixes(self):
        text = cae.summarize(self.report)
        for expected in ("warn", "1.21 %", "1.20 %", "exceeds_allowable", "catch <=", "arm thickness", "length to load"):
            self.assertIn(expected, text)


class HintTests(unittest.TestCase):
    SPEC = cae.validate_intent(intent())

    def test_no_hints_when_inside_the_margin_or_without_an_allowable(self):
        self.assertEqual(cae.repair_hints(1.2, cae.resolve_material("ABS"), self.SPEC), {})
        self.assertEqual(cae.repair_hints(9.0, cae.resolve_material("PETG"), self.SPEC), {})

    def test_hints_scale_as_the_beam_formula_says_and_land_on_the_target(self):
        material = cae.resolve_material("ABS")                      # allowable 7 %, target 5.6 %
        hints = cae.repair_hints(8.4, material, self.SPEC)          # 1.5 x too much
        self.assertAlmostEqual(hints["target_strain_pct"], 5.6)
        self.assertAlmostEqual(hints["max_catch_mm"], 1.0 * 5.6 / 8.4, places=3)
        self.assertAlmostEqual(hints["arm_thickness_mm"], 2.0 * 5.6 / 8.4, places=3)
        self.assertAlmostEqual(hints["length_to_load_mm"], 17.5 * math.sqrt(8.4 / 5.6), places=3)

    def test_summary_for_skipped_and_error_reports(self):
        self.assertIn("skipped", cae.summarize(cae.skipped_report("no env")))
        self.assertIn("boom", cae.summarize({"status": "error", "reasons": ["boom"]}))


class WorkspaceTests(unittest.TestCase):
    def test_paths_must_stay_inside_the_workspace_including_via_symlink(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            root = Path(tmp)
            cae.ensure_inside(root, root / "a.step", root / "out" / "x")
            for bad in (Path(other) / "a.step", root / ".." / "x.step"):
                with self.subTest(path=str(bad)), self.assertRaises(SnapArmError):
                    cae.ensure_inside(root, bad)
            (root / "link").symlink_to(other)
            with self.assertRaises(SnapArmError):
                cae.ensure_inside(root, root / "link" / "secret.step")

    def test_existing_links_in_the_output_folder_are_refused_and_never_followed(self):
        env = {k: v for k, v in os.environ.items() if k not in {"A3D_CAE_PYTHON", "A3D_CAE_REPO"}}
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other, \
                mock.patch.dict(os.environ, env, clear=True):
            root, victim = Path(tmp), Path(other) / "victim.txt"
            victim.write_text("keep")
            inside = root / "inside.json"
            inside.write_text("{}")
            for name, target in (("cae-report.json", victim), ("snap_arm_cae.aieng", victim),
                                 ("cae-report.json", inside)):          # outside target, then a link that stays inside
                with self.subTest(link=name, target=target.name):
                    out = root / f"out-{name}-{target.name}"
                    out.mkdir()
                    (out / name).symlink_to(target)
                    for flags in (["--workspace", str(root)], []):
                        code = cae.main(["--step", str(root / "a.step"), "--intent", str(root / "i.json"),
                                         "--out", str(out), *flags])
                        self.assertEqual(code, 2)
                    self.assertEqual(victim.read_text(), "keep")
                    self.assertEqual(inside.read_text(), "{}")

    def test_write_atomic_replaces_a_link_instead_of_following_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            victim, dest, source = root / "victim.txt", root / "dest.json", root / "src.bin"
            victim.write_text("keep")
            source.write_bytes(b"payload")
            dest.symlink_to(victim)
            cae.write_atomic(dest, b"{}")
            self.assertFalse(dest.is_symlink())
            self.assertEqual((dest.read_bytes(), victim.read_text()), (b"{}", "keep"))
            dest.unlink()
            cae.write_atomic(dest, source=source)
            self.assertEqual(dest.read_bytes(), b"payload")
            self.assertEqual([p.name for p in root.iterdir() if p.name.startswith(".snap-arm-")], [])

    def test_cli_refuses_outside_paths_and_reports_malformed_intent(self):
        env = {k: v for k, v in os.environ.items() if k not in {"A3D_CAE_PYTHON", "A3D_CAE_REPO"}}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env, clear=True):
            root = Path(tmp)
            self.assertEqual(cae.main(["--step", "/etc/hosts", "--intent", str(root / "i.json"),
                                       "--out", str(root / "out"), "--workspace", str(root)]), 2)
            self.assertFalse((root / "out").exists())


class ImportCodeTests(unittest.TestCase):
    def test_path_is_a_single_escaped_literal_whatever_it_contains(self):
        import ast
        for name in ("plain.step", "it's.step", 'say "hi".step', "a\nb.step", "back\\slash.step",
                     "x'); __import__('os').system('true'); ('.step"):
            with self.subTest(name=name):
                tree = ast.parse(cae.import_step_code(Path("/tmp") / name))
                calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
                self.assertEqual([c.func.id for c in calls if isinstance(c.func, ast.Name)],
                                 ["import_step", "Compound"])           # no extra calls were injected
                literal = next(c.args[0].value for c in calls if getattr(c.func, "id", "") == "import_step")
                self.assertEqual(literal, str(Path("/tmp") / name))


class CliTests(unittest.TestCase):
    def test_skips_with_a_recorded_reason_when_the_cae_environment_is_unset(self):
        env = {k: v for k, v in os.environ.items() if k not in {"A3D_CAE_PYTHON", "A3D_CAE_REPO"}}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env, clear=True):
            out = Path(tmp) / "out"
            self.assertEqual(cae.main(["--step", "x.step", "--intent", "i.json", "--out", str(out)]), 0)
            report = json.loads((out / "cae-report.json").read_text())
        self.assertEqual((report["status"], report["blocking"]), ("skipped", False))
        self.assertIn("A3D_CAE_PYTHON", report["reasons"][0])


@unittest.skipUnless(os.environ.get("A3D_CAE_PYTHON") and os.environ.get("A3D_CAE_REPO"),
                     "set A3D_CAE_PYTHON and A3D_CAE_REPO to run the real CalculiX check")
class RealSolverTests(unittest.TestCase):
    def test_snap_clip_from_step_matches_the_trial_numbers(self):
        from build123d import Align, Axis, Box, Plane, Polyline, Pos, export_step, extrude, fillet, make_face
        mn = (Align.MIN, Align.MIN, Align.MIN)
        x0, t, w, undercut = 40.0, 2.0, 6.0, 1.0
        profile = make_face(Polyline((17, t), (17, t + undercut), (18, t + undercut), (20, t), close=True))
        clip = (Pos(x0 - 4, 0, -3) * Box(4, w, 8, align=mn) + Pos(x0, 0, 0) * Box(20, w, t, align=mn)
                + Pos(x0, 0, 0) * extrude(Plane.XZ * profile, amount=-w))
        roots = [e for e in clip.edges().filter_by(Axis.Y)
                 if abs(e.center().X - x0) < 1e-6 and min(abs(e.center().Z), abs(e.center().Z - t)) < 1e-6]
        clip = fillet(roots, radius=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            step, spec, out = Path(tmp) / "clip.step", Path(tmp) / "intent.json", Path(tmp) / "out"
            export_step(clip, str(step))
            spec.write_text(json.dumps(intent()))
            done = subprocess.run([sys.executable, str(SKILL / "snap_arm_cae.py"), "--step", str(step),
                                   "--intent", str(spec), "--out", str(out)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            report = json.loads((out / "cae-report.json").read_text())
            self.assertTrue((out / "snap_arm_cae.aieng").is_file())
        self.assertEqual(report["status"], "pass", report.get("reasons"))
        self.assertEqual(len(report["meshes"]), 3)
        self.assertAlmostEqual(report["result"]["root_peak_strain_pct"], 1.20, delta=0.08)
        self.assertAlmostEqual(report["result"]["force_n"], 4.60, delta=0.1)
        self.assertTrue(report["versions"]["calculix"])
        self.assertEqual(len(report["inputs"]["step_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
