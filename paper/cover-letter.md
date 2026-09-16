# Cover letter — IEEE TCSVT Special Issue on *Towards Spatial Intelligence: 3D Understanding, Generation, and Reasoning from Real-World Videos*

> 투고 화면(Atypon Rex)의 cover letter 칸에 붙여 넣을 영문 초안. 편집자 이름·날짜·저자 정보는 투고 때 채울 것.

Dear Guest Editors,

We submit the manuscript **"Propagating a Single-View Mask Across a Static Camera Rig: Attribution and Error Budget of a Foundation-Model Pipeline for Multi-View Video Object Segmentation"** for consideration in the Special Issue on *Towards Spatial Intelligence: 3D Understanding, Generation, and Reasoning from Real-World Videos*.

## Why this paper belongs in the special issue

The Call for Papers asks for work at the intersection of video understanding, 3D modelling, and spatial reasoning on **real-world videos**, and names *maintaining temporal and geometric consistency*, *foundation models*, and *new benchmarks for 3D understanding* among its challenges and topics. Our paper addresses these directly, from the input side of 3D content creation:

1. **Real-world multi-view video as the object of study.** A static camera rig — a light-field array, a volumetric-capture studio, an instrumented arena — is the raw input of volumetric capture and of every per-object 3D reconstruction or generation pipeline downstream of it. Those pipelines consume multi-view masks. We study whether a 2D segmentation foundation model, given one object mask in one camera at one instant, maintains a geometrically and temporally consistent notion of that object across 9–46 real cameras and 21 frames — with no calibration, no category label, no per-scene optimisation — on MUVOD, the only public rig benchmark (17 scenes, 459 instances, four hemispherical fisheye rigs).

2. **Foundation models for 3D reasoning, measured rather than claimed.** The pipeline reaches 91.0 J&F³ against 79.4 for the published baseline. We decompose the margin (87 % is the foundation model inside an unchanged two-stage structure; 1.5 points are our design), bound the selection optimism of the one tuned element by leave-one-scene-out (0.045 points), and give an error budget for the remaining 9 points, including a model-free measurement of the tracker's mask-grid ceiling.

3. **Explicit 3D on real-world rigs — a negative result with a mechanism.** Section VII tests, under pre-registered gates, whether explicit 3D reasoning can repair the cross-view failures: structure-from-motion with dense stereo, feed-forward point maps (π³), and a per-scene 3D Gaussian Splatting. None passes. The 3DGS finding is of independent interest to 3D reconstruction from real-world video: on rigs of 9–30 wide-baseline cameras the rendered depth is biased towards the camera on every one of 27 cameras with a reference depth, and the depth error is *anti-correlated* with training-view PSNR — the reconstruction memorises the images and the surface dissolves into floaters.

4. **Benchmark analysis.** We reconstruct the benchmark's unpublished reference-camera protocol from three independent lines of evidence, report the grid ceiling as a benchmark property that applies to any tracker with the same output grid, and identify a 1:15 repair-to-protect asymmetry that any future seed-repair method on rigs must be designed against.

Mapped to the CFP's topics of interest, the paper speaks to *Foundation models for 3D reconstruction/generation/reasoning*, *New benchmarks for 3D understanding*, *Advanced 3D representations (3DGS)* through the negative result, *Human-in-the-loop 3D content generation* (one human annotation replaces per-camera labelling), and *Large-scale scene reconstruction* as the consumer of the masks we produce.

## What the paper does not do

We state this plainly so that the fit is judged on the right terms. The method builds no 3D representation; its claim is about what a 2D foundation model can and cannot do spatially on real rigs, and about where explicit 3D fails. There is one baseline and one dataset, because no other method has been evaluated on MUVOD; the paper's limitations section lists this and six further caveats.

## Statements

- The manuscript is original, not under consideration elsewhere, and all experiments were pre-registered in a public repository before being run (23 registration documents).
- Code, registration documents, and the per-scene results will be released with the paper.
- The authors have no conflicts of interest with the Guest Editors. *(확인 후 유지)*

Thank you for your consideration.

Sincerely,
Seongjin Park
Electronics and Telecommunications Research Institute (ETRI), Daejeon, Republic of Korea
seongjin.park@etri.re.kr
