# MVSeg2D report
## Headline (scene-mean J&F over the scored cameras)

| method | J | F | J&F | scenes | pairs | halluc. | total s | peak GPU MB |
|---|---|---|---|---|---|---|---|---|
| geoset_dec_ft_rep_e9 | 0.877 | 0.950 | 0.914 | 17 | 664 | 12 | 455 | 20860 |
| geoset_decb_ft_rep_e9 | 0.878 | 0.946 | 0.912 | 17 | 664 | 12 | 454 | 20860 |
| geoset_pred_ft_rep_e8 | 0.877 | 0.951 | 0.914 | 17 | 664 | 13 | 444 | 20828 |

## Per scene (pair-mean J&F)

| scene | geoset_dec_ft_rep_e9 | geoset_decb_ft_rep_e9 | geoset_pred_ft_rep_e8 |
|---|---|---|---|
| AlexaMeadeExhibit | 0.913 | 0.909 | 0.909 |
| AlexaMeadeFacePaint | 0.923 | 0.918 | 0.919 |
| Barn | 0.901 | 0.897 | 0.906 |
| Blocks | 0.922 | 0.915 | 0.922 |
| Breakfast | 0.849 | 0.858 | 0.857 |
| CBABasketball | 0.876 | 0.880 | 0.889 |
| Carpark | 0.951 | 0.953 | 0.950 |
| CoffeeMartini | 0.936 | 0.931 | 0.936 |
| Dog | 0.857 | 0.867 | 0.858 |
| Fencing | 0.946 | 0.952 | 0.944 |
| FlameSteak | 0.908 | 0.904 | 0.913 |
| Frog | 0.982 | 0.979 | 0.983 |
| MATF | 0.872 | 0.857 | 0.867 |
| MartialArts | 0.943 | 0.930 | 0.944 |
| Painter | 0.910 | 0.911 | 0.909 |
| PoznanStreet | 0.928 | 0.923 | 0.925 |
| Welder | 0.914 | 0.909 | 0.910 |

## Object size (target GT px), pair-mean J&F

| size_bin | geoset_dec_ft_rep_e9 | geoset_decb_ft_rep_e9 | geoset_pred_ft_rep_e8 |
|---|---|---|---|
| <2000 | 0.807 (n=59) | 0.794 (n=59) | 0.801 (n=59) |
| 2000-10000 | 0.894 (n=138) | 0.887 (n=138) | 0.892 (n=138) |
| >=10000 | 0.930 (n=467) | 0.930 (n=467) | 0.933 (n=467) |

## Rig, pair-mean J&F

| rig | geoset_dec_ft_rep_e9 | geoset_decb_ft_rep_e9 | geoset_pred_ft_rep_e8 |
|---|---|---|---|
| pinhole | 0.912 (n=542) | 0.910 (n=542) | 0.914 (n=542) |
| hemisphere | 0.909 (n=122) | 0.906 (n=122) | 0.905 (n=122) |

## View distance from the reference camera (camera-index distance; not spatial on hemisphere rigs), pair-mean J&F

| view_dist_bin | geoset_dec_ft_rep_e9 | geoset_decb_ft_rep_e9 | geoset_pred_ft_rep_e8 |
|---|---|---|---|
| 1 | 0.927 (n=16) | 0.927 (n=16) | 0.924 (n=16) |
| 2-3 | 0.891 (n=219) | 0.888 (n=219) | 0.891 (n=219) |
| >=4 | 0.922 (n=429) | 0.919 (n=429) | 0.923 (n=429) |

