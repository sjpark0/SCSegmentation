"""GeoQuery arm names and the GeoMem-2 arm each one inherits (torch-free; imported by host-side code)."""
QUERY_ARMS = ("geoquery_ft", "geoquery_nobias_ft", "geoquery_ptr_ft")   # geobias_only path + object-query hub
PTR_ARMS = ("geoquery_ptr_ft",)                        # query injected on the pointer tokens, g_cross 0.1, ObjBias from GeoBias
CONTINUE_ARMS = ("geobias_only_ft2",)                  # geobias_only_ft continued for the same budget, no hub
DEC_ARMS = ("geoset_dec_ft", "geoset_decb_ft")        # S1-E9: geoset_pred_ft recipe + the SAM mask decoder trained (docs/stage1-E9-prereg.md)
BAND_ARMS = ("geoset_decb_ft",)                        # S1-E9 D2: + boundary-band BCE on the target masks
# S1-E12 (docs/stage1-E12-prereg.md): geoset_pred_ft recipe + per-object half-plane dropout of the reference memory tokens;
# cut probability per arm (0.0 = the zero-dose control C0', the same code path; 1.0 = M+)
CUT_ARMS = ("geoset_cut0_ft", "geoset_cutall_ft")
CUT_PROB = {"geoset_cut0_ft": 0.0, "geoset_cutall_ft": 1.0}
SET2_ARMS = ("geoset_only_ft", "geoset_pred_ft") + DEC_ARMS + CUT_ARMS   # GeoSet-2: memory set on the geobias_only base (GT / predicted memory)
# S1-E16 (docs/stage1-E16-prereg.md): the geometry-free control -- the geoset_pred_ft recipe (predicted second memory, same
# steps / lr / seeds) with neither the 3D attention bias nor the 3D embedding; at decode the neighbours come from the rig
# index or SAM 3 features (run.py --nbr) and the gate is `obj`.  Not a bias arm, not a GeoSet-2 arm.
NOGEO_ARMS = ("nogeo_pred_ft",)
PRED_MEM_ARMS = ("geoset_pred_ft",) + DEC_ARMS + CUT_ARMS + NOGEO_ARMS   # second memory = the model's own prediction of a random camera
                                                       # (does NOT derive from SET2_ARMS: every predicted-memory arm is listed here)
FUSE_ARMS = ("geofuse_ft", "geofuse2_ft")              # GeoFuse: geoset_pred_ft recipe + aligned multi-view fusion (B);
                                                       # geofuse2_ft = two-sided alignment validity + own view excluded (discriminating rerun)
ALIGN_ARMS = ("geoalign_ft",)                          # GeoAlign: geoset_only_ft recipe, second memory = aligned view + GT masks
NO_EMBED_ARMS = ("geobias_only_ft",) + SET2_ARMS + FUSE_ARMS + ALIGN_ARMS + NOGEO_ARMS   # arms without the 3D embedding (bias only; NG: nothing)


def base_arm(arm):
    """The GeoMem-2 arm whose freezing/flags a GeoQuery arm inherits (spec 2.6)."""
    return "geobias_only_ft" if arm in QUERY_ARMS + CONTINUE_ARMS else arm
