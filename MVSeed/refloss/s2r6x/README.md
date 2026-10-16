# S2-R6 extension / generalisation design (2026-10-16) — list items 3 and 4

docs/stage2-R7-prereg.md (item 3: S2-R6 decisions into the c_ini tracker memory) and docs/stage2-R6H-prereg.md
(item 4a: the frozen rule on MUVOD's other annotated views; 4b template for independent data) summarise these.
No MUVOD-GT-scored stage-2 run was made; GPU probes read only the frame-0 prompt.

| path | what |
|---|---|
| `common.py`, `anat1.py`, `item3_ref.py`, `item3_nonref.py`, `sens.py`, `pix.py`, `memres.py`, `pseudo_count.py` | S2-R6 anatomy (where the +0.365 came from), item-3 no-run bounds, GT-free snap counts for the held-out views |
| `reg/` | GT pixel-identity to frame 0 (c_ini 48.6 %, other views 44.4 %) |
| `mem/` | the item-3 memory probes: `probe_mem.py` + `run_probe.sh` (inject_mem / memptr / cond / observe modes), Blocks and CoffeeMartini results, tau binding check |
| `svdata/` | the single-camera runner path checked on Fencing v4 (byte-identical, 168/168) |
| `tools/`, `STATIC_DEF.txt`, `davis_*.json` | static-camera screen (fixed before any MOSE/MOVi frame was seen) and the DAVIS 2017 check (gate passes 0/205 objects) |
| `judge1/`, `judge2/` | the judges' re-checks (census, crop-kept areas, MOSE / MOVi-D metadata) |
| `design_workflow_result.json` | the whole workflow result (maps, designs, judges, synthesis) |
