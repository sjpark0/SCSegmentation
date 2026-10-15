# MVSeg2D report
## Headline (scene-mean J&F over the scored cameras)

| method | J | F | J&F | scenes | pairs | halluc. | total s | peak GPU MB |
|---|---|---|---|---|---|---|---|---|
| geoset_cut0_ft_rep_x_e12 | 0.879 | 0.953 | 0.916 | 17 | 664 | 13 | 457 | 20717 |
| geoset_cutall_ft_rep_x_e12 | 0.876 | 0.951 | 0.913 | 17 | 664 | 14 | 458 | 20717 |
| geoset_pred_ft_rep_x_e11 | 0.878 | 0.952 | 0.915 | 17 | 664 | 13 | 465 | 20717 |

## Per scene (pair-mean J&F)

| scene | geoset_cut0_ft_rep_x_e12 | geoset_cutall_ft_rep_x_e12 | geoset_pred_ft_rep_x_e11 |
|---|---|---|---|
| AlexaMeadeExhibit | 0.910 | 0.910 | 0.910 |
| AlexaMeadeFacePaint | 0.922 | 0.916 | 0.921 |
| Barn | 0.904 | 0.904 | 0.906 |
| Blocks | 0.923 | 0.921 | 0.923 |
| Breakfast | 0.869 | 0.867 | 0.858 |
| CBABasketball | 0.892 | 0.869 | 0.893 |
| Carpark | 0.950 | 0.949 | 0.949 |
| CoffeeMartini | 0.938 | 0.937 | 0.938 |
| Dog | 0.859 | 0.856 | 0.857 |
| Fencing | 0.943 | 0.940 | 0.942 |
| FlameSteak | 0.914 | 0.914 | 0.914 |
| Frog | 0.984 | 0.980 | 0.984 |
| MATF | 0.873 | 0.868 | 0.869 |
| MartialArts | 0.944 | 0.942 | 0.944 |
| Painter | 0.910 | 0.910 | 0.909 |
| PoznanStreet | 0.925 | 0.924 | 0.925 |
| Welder | 0.915 | 0.919 | 0.912 |

## Object size (target GT px), pair-mean J&F

| size_bin | geoset_cut0_ft_rep_x_e12 | geoset_cutall_ft_rep_x_e12 | geoset_pred_ft_rep_x_e11 |
|---|---|---|---|
| <2000 | 0.814 (n=59) | 0.814 (n=59) | 0.803 (n=59) |
| 2000-10000 | 0.894 (n=138) | 0.893 (n=138) | 0.894 (n=138) |
| >=10000 | 0.933 (n=467) | 0.931 (n=467) | 0.933 (n=467) |

## Rig, pair-mean J&F

| rig | geoset_cut0_ft_rep_x_e12 | geoset_cutall_ft_rep_x_e12 | geoset_pred_ft_rep_x_e11 |
|---|---|---|---|
| pinhole | 0.916 (n=542) | 0.914 (n=542) | 0.915 (n=542) |
| hemisphere | 0.907 (n=122) | 0.907 (n=122) | 0.906 (n=122) |

## View distance from the reference camera (camera-index distance; not spatial on hemisphere rigs), pair-mean J&F

| view_dist_bin | geoset_cut0_ft_rep_x_e12 | geoset_cutall_ft_rep_x_e12 | geoset_pred_ft_rep_x_e11 |
|---|---|---|---|
| 1 | 0.926 (n=16) | 0.920 (n=16) | 0.924 (n=16) |
| 2-3 | 0.895 (n=219) | 0.894 (n=219) | 0.892 (n=219) |
| >=4 | 0.924 (n=429) | 0.922 (n=429) | 0.924 (n=429) |

