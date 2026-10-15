# MVSeg2D report
## Headline (scene-mean J&F over the scored cameras)

| method | J | F | J&F | scenes | pairs | halluc. | total s | peak GPU MB |
|---|---|---|---|---|---|---|---|---|
| geoset_pred_ft_rep_x_e11 | 0.878 | 0.952 | 0.915 | 17 | 664 | 13 | 465 | 20717 |
| geoset_pred_ft_rep_xs_e14 | 0.881 | 0.954 | 0.917 | 17 | 664 | 13 | 465 | 20717 |

## Per scene (pair-mean J&F)

| scene | geoset_pred_ft_rep_x_e11 | geoset_pred_ft_rep_xs_e14 |
|---|---|---|
| AlexaMeadeExhibit | 0.910 | 0.912 |
| AlexaMeadeFacePaint | 0.921 | 0.922 |
| Barn | 0.906 | 0.909 |
| Blocks | 0.923 | 0.924 |
| Breakfast | 0.858 | 0.861 |
| CBABasketball | 0.893 | 0.898 |
| Carpark | 0.949 | 0.953 |
| CoffeeMartini | 0.938 | 0.938 |
| Dog | 0.857 | 0.852 |
| Fencing | 0.942 | 0.944 |
| FlameSteak | 0.914 | 0.916 |
| Frog | 0.984 | 0.985 |
| MATF | 0.869 | 0.873 |
| MartialArts | 0.944 | 0.944 |
| Painter | 0.909 | 0.912 |
| PoznanStreet | 0.925 | 0.927 |
| Welder | 0.912 | 0.926 |

## Object size (target GT px), pair-mean J&F

| size_bin | geoset_pred_ft_rep_x_e11 | geoset_pred_ft_rep_xs_e14 |
|---|---|---|
| <2000 | 0.803 (n=59) | 0.804 (n=59) |
| 2000-10000 | 0.894 (n=138) | 0.894 (n=138) |
| >=10000 | 0.933 (n=467) | 0.937 (n=467) |

## Rig, pair-mean J&F

| rig | geoset_pred_ft_rep_x_e11 | geoset_pred_ft_rep_xs_e14 |
|---|---|---|
| pinhole | 0.915 (n=542) | 0.917 (n=542) |
| hemisphere | 0.906 (n=122) | 0.911 (n=122) |

## View distance from the reference camera (camera-index distance; not spatial on hemisphere rigs), pair-mean J&F

| view_dist_bin | geoset_pred_ft_rep_x_e11 | geoset_pred_ft_rep_xs_e14 |
|---|---|---|
| 1 | 0.924 (n=16) | 0.921 (n=16) |
| 2-3 | 0.892 (n=219) | 0.896 (n=219) |
| >=4 | 0.924 (n=429) | 0.926 (n=429) |

