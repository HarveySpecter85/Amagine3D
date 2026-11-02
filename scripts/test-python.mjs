#!/usr/bin/env node

import { spawn, spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { availableParallelism } from 'node:os';
import { basename, delimiter, join, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const python =
  process.platform === 'win32'
    ? join(root, '.venv', 'Scripts', 'python.exe')
    : join(root, '.venv', 'bin', 'python');

const moduleName = (name) => `tests.python.${name}`;
const methodName = (name) =>
  `${moduleName('test_authoring_example')}.PublicAuthoringExampleTests.${name}`;

const unitA = [
  'test_brep_measurements',
  'test_build_manifest',
  'test_cad_compile',
  'test_cad_draft',
  'test_display_normals',
  'test_step_preview',
  'test_interface_recipes',
  'test_mesh_topology',
  'test_render_preview_materials',
  'test_source_preflight',
  'test_text_a3d_printability',
  'test_text_a3d_scene_contract',
];
const unitB = [
  'test_agent_api',
  'test_design_language',
  'test_design_roles',
  'test_design_connections',
  'test_assembly_orientation',
  'test_brep_envelope',
  'test_brep_stl_normalization',
  'test_brep_tessellation',
  'test_build_session',
  'test_cad_diagnostics',
  'test_capability_manifest',
  'test_display_components',
  'test_freshness_check',
  'test_installation_audit',
  'test_installation_check',
  'test_intent_revisions',
  'test_interface_geometry',
  'test_mesh_normalization',
  'test_opening_placement',
  'test_plate_layout',
  'test_print_orientation',
  'test_print_plates',
  'test_self_tapping_connections',
  'test_skill_shared_files',
  'test_snap_arm_cae',
  'test_source_diagnostics',
  'test_text_a3d_authoring',
  'test_text_a3d_color_guardrails',
  'test_visual_compare',
];

const suites = {
  'cad-basic': [
    methodName('test_example_compiles_with_measured_interface_and_five_views'),
    methodName('test_module_resize_keeps_locator_in_material_and_screws_at_corners'),
    methodName('test_single_part_starter_exports_a_real_blind_pocket'),
  ],
  'cad-module': [
    methodName('test_cover_thickness_edit_keeps_through_holes_and_interface_on_current_control'),
    methodName('test_installation_controls_import_without_geometry_and_do_not_rewrite_targets'),
    methodName('test_installed_module_exports_two_parts_with_installation_and_screw_proofs'),
  ],
  'cad-surface': [
    methodName('test_surface_shell_compiles_with_measured_width_and_open_cavity'),
    methodName('test_surface_shell_finishing_edit_previews_but_fails_final_acceptance'),
    methodName('test_surface_shell_recompiles_changed_walls_without_rewriting_intent'),
    methodName('test_surface_shell_rejects_semantic_envelope_drift'),
  ],
  'cpu-renderer': [moduleName('test_cpu_z_buffer'), moduleName('test_preview_resource_limits')],
  'unit-a': unitA.map(moduleName),
  'unit-b': unitB.map(moduleName),
};

function validateCoverage() {
  const files = readdirSync(join(root, 'tests', 'python'))
    .filter((name) => /^test_.*\.py$/u.test(name))
    .map((name) => basename(name, '.py'))
    .sort();
  const assigned = [...unitA, ...unitB, 'test_authoring_example', 'test_cpu_z_buffer', 'test_preview_resource_limits'];
  const duplicates = assigned.filter((name, index) => assigned.indexOf(name) !== index);
  const missing = files.filter((name) => !assigned.includes(name));
  const unknown = assigned.filter((name) => !files.includes(name));
  if (duplicates.length || missing.length || unknown.length) {
    throw new Error(
      `Python suite manifest is invalid: ${JSON.stringify({ duplicates, missing, unknown })}`,
    );
  }
}

validateCoverage();

const usage = `Usage: node scripts/test-python.mjs [--suite ${Object.keys(suites).sort().join('|')}] [--workers N]`;

function parseWorkers(value) {
  const workers = Number.parseInt(value ?? '', 10);
  if (!Number.isFinite(workers) || workers < 1) return null;
  return Math.min(workers, availableParallelism());
}

const args = process.argv.slice(2);
let suiteName = null;
let workers = null;
for (let index = 0; index < args.length; index += 1) {
  const arg = args[index];
  if (arg === '--list-suites' && args.length === 1) {
    console.log(Object.keys(suites).sort().join('\n'));
    process.exit(0);
  } else if (arg === '--suite' && index + 1 < args.length) {
    suiteName = args[index + 1];
    index += 1;
  } else if (arg === '--workers' && index + 1 < args.length) {
    workers = parseWorkers(args[index + 1]);
    if (workers === null) {
      console.error(usage);
      process.exit(2);
    }
    index += 1;
  } else {
    console.error(usage);
    process.exit(2);
  }
}
if (suiteName !== null && !suites[suiteName]) {
  console.error(usage);
  process.exit(2);
}
if (workers === null) {
  workers = parseWorkers(process.env.A3D_PYTHON_WORKERS) ?? 1;
}

function discoverTestIds() {
  const discoverScript = `
import sys, unittest
sys.path.insert(0, ${JSON.stringify(root)})


def walk(node):
    if isinstance(node, unittest.TestSuite):
        for child in node:
            yield from walk(child)
    else:
        yield node.id()


suite = unittest.TestLoader().discover(${JSON.stringify(join(root, 'tests', 'python'))}, pattern='test_*.py')
for name in walk(suite):
    print(name)
`;
  const found = spawnSync(python, ['-c', discoverScript], {
    cwd: root,
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  });
  if (found.error) {
    console.error(`Python tests could not start: ${found.error.message}`);
    process.exit(1);
  }
  if (found.status !== 0) {
    process.stderr.write(found.stderr ?? '');
    process.exit(found.status ?? 1);
  }
  return (found.stdout ?? '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
}

const running = new Set();
const workerEnv = {
  ...process.env,
  PYTHONPATH: [join(root, 'tests', 'python'), process.env.PYTHONPATH].filter(Boolean).join(delimiter),
};

function runCases(ids) {
  return new Promise((settle) => {
    const child = spawn(python, ['-m', 'unittest', ...ids], { cwd: root, env: workerEnv });
    running.add(child);
    let output = '';
    child.stdout.on('data', (chunk) => {
      output += chunk;
    });
    child.stderr.on('data', (chunk) => {
      output += chunk;
    });
    child.once('error', (error) => {
      running.delete(child);
      settle({ code: 1, output: `${output}\nPython tests could not start: ${error.message}` });
    });
    child.once('exit', (code) => {
      running.delete(child);
      settle({ code: code ?? 1, output });
    });
  });
}

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    for (const child of running) child.kill(signal);
  });
}

if (workers === 1) {
  const pythonArgs =
    suiteName === null
      ? ['-m', 'unittest', 'discover', '-s', 'tests/python', '-p', 'test_*.py']
      : ['-m', 'unittest', ...suites[suiteName]];
  const single = spawnSync(python, pythonArgs, { cwd: root, stdio: 'inherit' });
  if (single.error) {
    console.error(`Python tests could not start: ${single.error.message}`);
    process.exit(1);
  }
  process.exit(single.status ?? 1);
}

const ids = suiteName === null ? discoverTestIds() : suites[suiteName];
if (ids.length === 0) {
  console.error('No Python tests were discovered.');
  process.exit(1);
}
const shards = Array.from({ length: Math.min(workers, ids.length) }, () => []);
ids.forEach((id, index) => shards[index % shards.length].push(id));
console.log(`Python tests: ${ids.length} cases across ${shards.length} workers`);
const results = await Promise.all(
  shards.map(async (shard, index) => ({
    index,
    size: shard.length,
    ...(await runCases(shard)),
  })),
);
let failures = 0;
for (const result of results) {
  const label = `[worker ${result.index + 1}] ${result.size} cases`;
  if (result.code === 0) {
    const summary = result.output
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.startsWith('Ran ') || line.startsWith('OK'));
    console.log(`${label} — ${summary.join(' ')}`);
    continue;
  }
  failures += 1;
  console.error(`${label} — FAILED\n${result.output}`);
}
process.exit(failures === 0 ? 0 : 1);
