# MVSeg2D report
## Headline (scene-mean J&F over the scored cameras)

| method | J | F | J&F | scenes | pairs | halluc. | total s | peak GPU MB |
|---|---|---|---|---|---|---|---|---|
| geoset_pred_ft_rep_e8 | 0.877 | 0.951 | 0.914 | 17 | 664 | 13 | 444 | 20828 |
| geoset_pred_ft_rep_gf_e10 | 0.867 | 0.950 | 0.909 | 17 | 664 | 13 | 793 | 0 |
| geoset_pred_ft_rep_z2_e10 | 0.882 | 0.951 | 0.916 | 17 | 664 | 13 | 1585 | 20717 |
| geoset_pred_ft_rep_z2r_e10 | 0.882 | 0.951 | 0.917 | 17 | 664 | 13 | 1595 | 20717 |

## Per scene (pair-mean J&F)

| scene | geoset_pred_ft_rep_e8 | geoset_pred_ft_rep_gf_e10 | geoset_pred_ft_rep_z2_e10 | geoset_pred_ft_rep_z2r_e10 |
|---|---|---|---|---|
| AlexaMeadeExhibit | 0.909 | 0.902 | 0.920 | 0.920 |
| AlexaMeadeFacePaint | 0.919 | 0.913 | 0.915 | 0.915 |
| Barn | 0.906 | 0.905 | 0.915 | 0.916 |
| Blocks | 0.922 | 0.921 | 0.925 | 0.923 |
| Breakfast | 0.857 | 0.853 | 0.867 | 0.867 |
| CBABasketball | 0.889 | 0.877 | 0.885 | 0.885 |
| Carpark | 0.950 | 0.944 | 0.944 | 0.945 |
| CoffeeMartini | 0.936 | 0.935 | 0.942 | 0.943 |
| Dog | 0.858 | 0.851 | 0.858 | 0.858 |
| Fencing | 0.944 | 0.926 | 0.948 | 0.943 |
| FlameSteak | 0.913 | 0.910 | 0.920 | 0.921 |
| Frog | 0.983 | 0.984 | 0.983 | 0.983 |
| MATF | 0.867 | 0.857 | 0.873 | 0.873 |
| MartialArts | 0.944 | 0.937 | 0.945 | 0.946 |
| Painter | 0.909 | 0.908 | 0.910 | 0.910 |
| PoznanStreet | 0.925 | 0.913 | 0.918 | 0.921 |
| Welder | 0.910 | 0.914 | 0.912 | 0.913 |

## Object size (target GT px), pair-mean J&F

| size_bin | geoset_pred_ft_rep_e8 | geoset_pred_ft_rep_gf_e10 | geoset_pred_ft_rep_z2_e10 | geoset_pred_ft_rep_z2r_e10 |
|---|---|---|---|---|
| <2000 | 0.801 (n=59) | 0.783 (n=59) | 0.823 (n=59) | 0.823 (n=59) |
| 2000-10000 | 0.892 (n=138) | 0.882 (n=138) | 0.899 (n=138) | 0.899 (n=138) |
| >=10000 | 0.933 (n=467) | 0.931 (n=467) | 0.933 (n=467) | 0.933 (n=467) |

## Rig, pair-mean J&F

| rig | geoset_pred_ft_rep_e8 | geoset_pred_ft_rep_gf_e10 | geoset_pred_ft_rep_z2_e10 | geoset_pred_ft_rep_z2r_e10 |
|---|---|---|---|---|
| pinhole | 0.914 (n=542) | 0.909 (n=542) | 0.917 (n=542) | 0.918 (n=542) |
| hemisphere | 0.905 (n=122) | 0.902 (n=122) | 0.911 (n=122) | 0.911 (n=122) |

## View distance from the reference camera (camera-index distance; not spatial on hemisphere rigs), pair-mean J&F

| view_dist_bin | geoset_pred_ft_rep_e8 | geoset_pred_ft_rep_gf_e10 | geoset_pred_ft_rep_z2_e10 | geoset_pred_ft_rep_z2r_e10 |
|---|---|---|---|---|
| 1 | 0.924 (n=16) | 0.920 (n=16) | 0.922 (n=16) | 0.922 (n=16) |
| 2-3 | 0.891 (n=219) | 0.887 (n=219) | 0.898 (n=219) | 0.898 (n=219) |
| >=4 | 0.923 (n=429) | 0.918 (n=429) | 0.925 (n=429) | 0.926 (n=429) |

