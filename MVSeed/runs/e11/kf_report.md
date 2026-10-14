# MVSeg2D report
## Headline (scene-mean J&F over the scored cameras)

| method | J | F | J&F | scenes | pairs | halluc. | total s | peak GPU MB |
|---|---|---|---|---|---|---|---|---|
| geoset_pred_ft_rep_e8 | 0.877 | 0.951 | 0.914 | 17 | 664 | 13 | 444 | 20828 |
| geoset_pred_ft_rep_x_e11 | 0.878 | 0.952 | 0.915 | 17 | 664 | 13 | 465 | 20717 |
| geoset_pred_ft_rep_xm_e11 | 0.878 | 0.952 | 0.915 | 17 | 664 | 13 | 468 | 20717 |
| geoset_pred_ft_rep_xu_e11 | 0.876 | 0.950 | 0.913 | 17 | 664 | 12 | 467 | 20717 |

## Per scene (pair-mean J&F)

| scene | geoset_pred_ft_rep_e8 | geoset_pred_ft_rep_x_e11 | geoset_pred_ft_rep_xm_e11 | geoset_pred_ft_rep_xu_e11 |
|---|---|---|---|---|
| AlexaMeadeExhibit | 0.909 | 0.910 | 0.909 | 0.910 |
| AlexaMeadeFacePaint | 0.919 | 0.921 | 0.922 | 0.921 |
| Barn | 0.906 | 0.906 | 0.906 | 0.906 |
| Blocks | 0.922 | 0.923 | 0.923 | 0.923 |
| Breakfast | 0.857 | 0.858 | 0.858 | 0.858 |
| CBABasketball | 0.889 | 0.893 | 0.892 | 0.893 |
| Carpark | 0.950 | 0.949 | 0.949 | 0.949 |
| CoffeeMartini | 0.936 | 0.938 | 0.938 | 0.938 |
| Dog | 0.858 | 0.857 | 0.857 | 0.857 |
| Fencing | 0.944 | 0.942 | 0.942 | 0.942 |
| FlameSteak | 0.913 | 0.914 | 0.912 | 0.914 |
| Frog | 0.983 | 0.984 | 0.984 | 0.984 |
| MATF | 0.867 | 0.869 | 0.869 | 0.853 |
| MartialArts | 0.944 | 0.944 | 0.942 | 0.944 |
| Painter | 0.909 | 0.909 | 0.911 | 0.909 |
| PoznanStreet | 0.925 | 0.925 | 0.925 | 0.925 |
| Welder | 0.910 | 0.912 | 0.912 | 0.912 |

## Object size (target GT px), pair-mean J&F

| size_bin | geoset_pred_ft_rep_e8 | geoset_pred_ft_rep_x_e11 | geoset_pred_ft_rep_xm_e11 | geoset_pred_ft_rep_xu_e11 |
|---|---|---|---|---|
| <2000 | 0.801 (n=59) | 0.803 (n=59) | 0.804 (n=59) | 0.794 (n=59) |
| 2000-10000 | 0.892 (n=138) | 0.894 (n=138) | 0.894 (n=138) | 0.894 (n=138) |
| >=10000 | 0.933 (n=467) | 0.933 (n=467) | 0.933 (n=467) | 0.933 (n=467) |

## Rig, pair-mean J&F

| rig | geoset_pred_ft_rep_e8 | geoset_pred_ft_rep_x_e11 | geoset_pred_ft_rep_xm_e11 | geoset_pred_ft_rep_xu_e11 |
|---|---|---|---|---|
| pinhole | 0.914 (n=542) | 0.915 (n=542) | 0.915 (n=542) | 0.914 (n=542) |
| hemisphere | 0.905 (n=122) | 0.906 (n=122) | 0.906 (n=122) | 0.906 (n=122) |

## View distance from the reference camera (camera-index distance; not spatial on hemisphere rigs), pair-mean J&F

| view_dist_bin | geoset_pred_ft_rep_e8 | geoset_pred_ft_rep_x_e11 | geoset_pred_ft_rep_xm_e11 | geoset_pred_ft_rep_xu_e11 |
|---|---|---|---|---|
| 1 | 0.924 (n=16) | 0.924 (n=16) | 0.925 (n=16) | 0.924 (n=16) |
| 2-3 | 0.891 (n=219) | 0.892 (n=219) | 0.892 (n=219) | 0.892 (n=219) |
| >=4 | 0.923 (n=429) | 0.924 (n=429) | 0.924 (n=429) | 0.923 (n=429) |

