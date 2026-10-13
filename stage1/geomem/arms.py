"""GeoQuery arm names and the GeoMem-2 arm each one inherits (torch-free; imported by host-side code)."""
QUERY_ARMS = ("geoquery_ft", "geoquery_nobias_ft", "geoquery_ptr_ft")   # geobias_only path + object-query hub
PTR_ARMS = ("geoquery_ptr_ft",)                        # query injected on the pointer tokens, g_cross 0.1, ObjBias from GeoBias
CONTINUE_ARMS = ("geobias_only_ft2",)                  # geobias_only_ft continued for the same budget, no hub
SET2_ARMS = ("geoset_only_ft", "geoset_pred_ft")      # GeoSet-2: memory set on the geobias_only base (GT / predicted memory)
FUSE_ARMS = ("geofuse_ft", "geofuse2_ft")              # GeoFuse: geoset_pred_ft recipe + aligned multi-view fusion (B);
                                                       # geofuse2_ft = two-sided alignment validity + own view excluded (discriminating rerun)
ALIGN_ARMS = ("geoalign_ft",)                          # GeoAlign: geoset_only_ft recipe, second memory = aligned view + GT masks
NO_EMBED_ARMS = ("geobias_only_ft",) + SET2_ARMS + FUSE_ARMS + ALIGN_ARMS   # arms without the 3D embedding (bias only)


def base_arm(arm):
    """The GeoMem-2 arm whose freezing/flags a GeoQuery arm inherits (spec 2.6)."""
    return "geobias_only_ft" if arm in QUERY_ARMS + CONTINUE_ARMS else arm
