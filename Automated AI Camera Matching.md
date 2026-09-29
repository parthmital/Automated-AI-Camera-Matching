# Automated Single-Image Camera Matching and Inverse Rendering for 3D Production

> **Note:** Numeric values marked _[missing]_ were lost when this document was exported (they were embedded as equation images). Verify them against the cited sources before relying on them.

## Contents

- [Evolution from Classical Vanishing Geometry to Deep Invariant Representations](#evolution-from-classical-vanishing-geometry-to-deep-invariant-representations)
- [Deep Architectural Approaches to Monocular Calibration and Geometry](#deep-architectural-approaches-to-monocular-calibration-and-geometry)
  - [GeoCalib: Second-Order Geometric Optimisation on Neural Perspective Fields](#geocalib-second-order-geometric-optimisation-on-neural-perspective-fields)
  - [MoGe Series (MoGe-2 and MoGe-3): Monocular Geometry Foundation Models](#moge-series-moge-2-and-moge-3-monocular-geometry-foundation-models)
  - [Apple Depth Pro: High-Resolution Metric Depth and Decoupled Focal Length Estimation](#apple-depth-pro-high-resolution-metric-depth-and-decoupled-focal-length-estimation)
  - [AnyCalib: Universal On-Manifold Calibration for Arbitrary Camera Models](#anycalib-universal-on-manifold-calibration-for-arbitrary-camera-models)
  - [UniDepth and UniDAC: Disentangled Pseudo-Spherical Metric Calibration](#unidepth-and-unidac-disentangled-pseudo-spherical-metric-calibration)
  - [Puffin: Spatial Reasoning and Multimodal Thinking with Camera](#puffin-spatial-reasoning-and-multimodal-thinking-with-camera)
- [Quantitative Benchmarks and Cross-Model Architectural Comparison](#quantitative-benchmarks-and-cross-model-architectural-comparison)
- [Fully Automated Blender Integration Pipeline: The Zero-Click fSpy Architecture](#fully-automated-blender-integration-pipeline-the-zero-click-fspy-architecture)
  - [Mathematical Coordinate Transformation](#mathematical-coordinate-transformation)
  - [Automated Ground Plane Estimation for Camera Height](#automated-ground-plane-estimation-for-camera-height)
  - [Automated fSpy JSON Exporter](#automated-fspy-json-exporter)
  - [Headless Blender Automation Script](#headless-blender-automation-script)
- [Systematic Failure Modes, Edge Cases, and Operational Safeguards](#systematic-failure-modes-edge-cases-and-operational-safeguards)
  - [Optical Centre Decentring and Cropped Plates](#optical-centre-decentring-and-cropped-plates)
  - [Scale Ambiguity and Ground Plane Identification](#scale-ambiguity-and-ground-plane-identification)
  - [Specular, Reflective, and Translucent Surfaces](#specular-reflective-and-translucent-surfaces)
  - [Radial and Non-Rectilinear Lens Distortions](#radial-and-non-rectilinear-lens-distortions)
- [Practical Synthesis and Implementation Strategies](#practical-synthesis-and-implementation-strategies)
- [Works Cited](#works-cited)

## Evolution from Classical Vanishing Geometry to Deep Invariant Representations

Single-view camera calibration in visual effects (VFX), computer animation, and architectural visualisation has historically depended on classical projective geometry [1]. Utilities such as fSpy, and its predecessor BLAM, rely on user-defined vanishing lines to compute vanishing points along mutually orthogonal axes [2]. From these intersections, the camera's focal length, field of view (FoV), and spatial orientation (pitch, roll, and yaw) are calculated via closed-form projective equations [2].

While mathematically exact under noise-free conditions, this manual paradigm imposes severe operational limitations [1]:

- It requires structured Manhattan or Atlanta world geometries with visible, unoccluded parallel lines [2].
- It assumes an idealised rectilinear pinhole camera with an unshifted optical centre and zero lens distortion [2].
- It demands human intervention to manually place alignment gizmos, precluding batch processing or real-time pipelines [1].
- It fails when applied to organic landscapes, close-up portraits, non-rectilinear architectural environments, or cropped imagery [8].

Recent deep learning models reformulate monocular camera calibration and inverse rendering by learning continuous geometric representations directly from high-capacity vision backbones [7]. Instead of extracting fragile 2D line segments, modern networks predict dense, per-pixel geometric representations, including gravity up-vectors, latitude fields, optical ray directions, and dense metric point surfaces [7].

These models decouple camera calibration from strict structural scene assumptions [7]. By integrating these neural estimators with automated metric ground-plane solvers and headless 3D engine bridges, technical artists can now take a single arbitrary 2D image, estimate its complete intrinsic and extrinsic parameters, and instantiate a fully matched virtual camera in Blender without manual intervention [7].

## Deep Architectural Approaches to Monocular Calibration and Geometry

### GeoCalib: Second-Order Geometric Optimisation on Neural Perspective Fields

GeoCalib, introduced by Veicht et al. at ECCV 2024, bridges deep semantic feature extraction with classical non-linear optimisation to estimate camera intrinsics and absolute gravity orientation from a single uncalibrated plate [7]. The primary codebase and model checkpoints are available in the [official repository](https://github.com/cvg/GeoCalib) [7].

The architecture addresses the poor generalisation typical of direct parameter regression networks, which routinely overfit to specific camera distributions [1]. GeoCalib employs a SegNeXt-based convolutional vision backbone that outputs two dense perspective vector fields alongside per-pixel uncertainty estimations [16]: an up-vector field and a latitude field.

The up-vector field defines the projected 2D direction of world gravity at each pixel, while the latitude field denotes the elevation angle of the camera ray corresponding to each pixel relative to the horizontal plane [16]. Rather than treating these maps as heuristic visualisations, GeoCalib embeds a differentiable Levenberg-Marquardt (LM) optimiser directly into the training loop [16].

The optimiser minimises a confidence-weighted non-linear least-squares residual between the observed fields and the analytic projection induced by the camera parameters, where focal length is optimised in log-space, gravity is parameterised on the unit sphere, and distortion is parameterised via polynomial coefficients [16].

Because gradients propagate backwards through the unrolled optimiser iterations during training, the SegNeXt encoder learns to prioritise informative geometric visual anchors (such as vertical architecture, human figures, trees, and horizon gradients) while assigning high variance (low confidence weight) to ambiguous or non-upright surfaces [16].

GeoCalib accepts an unconstrained single RGB image of arbitrary resolution [7]. The forward pass outputs horizontal and vertical focal lengths, the vertical and horizontal field of view (vFoV, hFoV), lens distortion parameters, the world-space gravity vector (from which camera roll and pitch are directly solved), and per-pixel uncertainty heatmaps [7]. The model operates completely autonomously, executing feedforward inference without initial manual bounds or seed coordinates [7].

Evaluated on the MegaDepth, LaMAR, Stanford2D3D, and TartanAir benchmarks, GeoCalib achieves substantial performance improvements over prior single-image calibration models [7]. On MegaDepth, GeoCalib achieves an Area Under the Curve (AUC) for roll at _[missing]_ error thresholds of _[missing]_ (median error _[missing]_), pitch AUC of _[missing]_ (median error _[missing]_), and FoV AUC of _[missing]_ (median error _[missing]_) [7]. On LaMAR indoor AR sequences, its roll AUC reaches _[missing]_ with a pitch AUC of _[missing]_ [7].

The model supports standard rectilinear pinhole images, images exhibiting moderate radial distortion, and division-model optics [7]. It handles natural outdoor environments, domestic interiors, and distorted wide-angle shots [7].

Blender integration is well-supported within the community ecosystem [13]. The open-source Atlas Camera extension incorporates GeoCalib to automate camera setups in headless Blender (`bpy`) sessions [13], while custom ComfyUI nodes serialise GeoCalib's predicted pitch, roll, and FoV directly into fSpy-formatted JSON strings [20].

The underlying code is licensed under Apache-2.0, while model weights are distributed under the Creative Commons Attribution 4.0 International License (CC-BY 4.0), permitting unencumbered commercial and studio deployment [7]. Hardware demands are modest: GeoCalib requires approximately _[missing]_ of VRAM and achieves an average inference latency of _[missing]_ on an NVIDIA RTX 3090/4090 GPU [17].

The primary operational limitation of GeoCalib is its structural assumption that the principal point coincides with the geometric centre of the sensor, meaning heavy asymmetric crops induce angular drift in estimated pitch [7]. Additionally, while it fully resolves camera rotation and focal length, it does not infer metric translation (such as camera height above the ground plane) [7].

Consequently, while it functions as a drop-in replacement for fSpy's vanishing-point orientation and FoV extraction, full spatial positioning requires pairing with an automated ground-plane depth detector [7].

### MoGe Series (MoGe-2 and MoGe-3): Monocular Geometry Foundation Models

Developed by Microsoft Research, the MoGe family reformulates monocular 3D perception by predicting continuous metric point maps, surface normal fields, and camera FoV within a single forward pass [11]. The project is documented across foundational publications: _MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision_ (Wang et al., CVPR 2025 Oral) and _MoGe-3: Fine-Detail Monocular Geometry Estimation with Self-Guided Sparse Volumetric Refinement_ (Kong et al., 2026) [11]. The implementation is maintained at <https://github.com/microsoft/MoGe> [14].

MoGe-1 and MoGe-2 pair a high-capacity DINOv2 Vision Transformer encoder (ViT-L/ViT-G) with a convolutional upsampling decoder [11]. To overcome scale-shift ambiguities during monocular training without discarding geometric detail, the architecture implements a closed-form Robust, Optimal, and Efficient (ROE) alignment solver [11]. This enforces affine-invariant 3D point-map supervision across multi-scale spherical neighbourhoods, penalising surface normal discrepancies and geometric warping [11].

MoGe-3 introduces Self-Guided Sparse 3D Refinement (SSR) [24]. Standard 2D convolutional or attention decoders bleed features across sharp depth boundaries, rounding thin geometric structures (such as poles, foliage, and structural edges) [24]. MoGe-3 resolves this by unprojecting the base point map into a sparse 3D voxel shell and iteratively applying sparse 3D convolutions [24]. Because foreground edges and distant background regions occupy physically distinct voxels in 3D Euclidean space, feature bleeding across occluding contours is eliminated [24].

Simultaneously, an auxiliary projection head recovers the horizontal field of view (hFoV) by enforcing projective consistency across the reconstructed 3D points [11].

MoGe ingests an uncalibrated RGB image of arbitrary resolution and aspect ratio [14]. It outputs a dense metric 3D point map (one 3D point per pixel), a metric depth map, an aligned surface normal map, a sky/infinity semantic mask, and camera horizontal/vertical FoV values [11]. The pipeline is fully automated and supports execution via a CLI that converts predicted point geometry into textured `.glb` 3D assets [14].

MoGe achieves high geometric accuracy across unseen benchmarks, setting the current state of the art in open-domain point map and normal estimation [14]. On zero-shot evaluations across nine public datasets, MoGe-3 matches or exceeds the boundary recall of specialised depth models while generating lower point-cloud distortion and preserving fine architectural details [24].

The model supports unconstrained perspective photography, macro shots, wide environments, and full 360° equirectangular panoramas via a dedicated spherical parameterisation pipeline (`moge infer_panorama`) [14].

Blender integration is direct: because the model can export textured `.glb` meshes embedding metric coordinates alongside computed FoV parameters, importing the resulting asset into Blender instantiates both the scene geometry and an aligned camera frustum [14].

MoGe is licensed under the permissive MIT License, enabling integration into commercial and proprietary pipelines [25]. On an NVIDIA RTX 4090, MoGe-2 (ViT-L) runs in _[missing]_ per image, while MoGe-3 (with three SSR refinement iterations) executes in _[missing]_, requiring _[missing]_ of VRAM under half precision (FP16) [14].

The model's primary constraint regarding automated camera matching is that it does not output explicit camera pitch and roll angles as direct scalar variables [11]. These extrinsic orientations must be derived downstream by extracting the normal vector of the ground plane from the predicted point map or normal map [11].

When combined with an automated plane-fitting script, MoGe replaces fSpy's functionality while delivering both the camera solve and an aligned 3D proxy mesh [11].

### Apple Depth Pro: High-Resolution Metric Depth and Decoupled Focal Length Estimation

Depth Pro, developed by Apple Research (Bochkovskii et al., ICLR 2025), is a metric monocular depth foundation model capable of inferring absolute metric scale alongside focal length from an uncalibrated single image [9]. The official implementation is accessible at <https://github.com/apple-aiml-research/ml-depth-pro> [28].

The architecture utilises a multi-scale Vision Transformer design optimised to process global perspective context alongside high-frequency boundary details [28]. The input image is downsampled into hierarchical spatial representations, split into patches, processed via weight-shared ViT encoders, and synthesised via a Dense Prediction Transformer (DPT) decoder [30].

To overcome the scale ambiguity inherent in monocular vision, Depth Pro decouples focal length estimation from dense depth estimation [30]. Rather than training the two heads jointly, which introduces gradient interference and degrades metric stability, Depth Pro freezes the core depth encoder and trains a separate focal length estimation head using large image datasets with EXIF metadata [30].

The predicted focal length in pixels scales the canonical inverse depth map into an absolute metric depth map, using the image width as a normalising factor [9].

Depth Pro takes a single RGB image and outputs a _[missing]_ metric depth map (_[missing]_ native resolution) with absolute physical scale in metres, alongside the estimated horizontal focal length focal length [28]. Operation is entirely automatic, requiring no prior EXIF tags or user hints [28].

Depth Pro exhibits high zero-shot boundary tracing precision, avoiding flying-pixel edge artefacts on intricate silhouettes such as hair, power lines, and foliage [9]. On zero-shot focal length benchmarks, it outperforms competing methods [9]. On the PPR10K evaluation benchmark, _[missing]_ of Depth Pro's focal length predictions achieve an error within _[missing]_ of ground truth (_[missing]_), compared to _[missing]_ for the previous state of the art (SPEC) [33].

It reliably processes portraits, architectural interiors, landscapes, and street photography [9].

Blender integration is implemented through custom nodes (such as ComfyUI-Depth-Pro) and Python unprojection scripts [37]. The predicted focal length in pixels maps directly to Blender's millimetre-based focal length by scaling it with the ratio of sensor width (mm) to image width (pixels).

The metric depth map unprojects to an absolute point cloud where 1 Blender unit equals 1 real-world metre [28].

The codebase is released under the Apple Sample Code License, while the model weights are covered by an accompanying research agreement that restricts unapproved commercial redistribution [28].

Inference executes in _[missing]_ on an NVIDIA V100/A100 or RTX 4090 GPU, requiring _[missing]_ of VRAM, with additional compatibility across Apple Silicon via Metal Performance Shaders (MPS) and ONNX/WebGPU runtimes [28].

Depth Pro's main limitation is that it regresses focal length and metric depth along camera viewing rays without predicting world-space gravity orientation (camera pitch and roll) [28]. Consequently, while it resolves fSpy's focal length estimation and establishes scene scale, it cannot orient the camera relative to a world horizon on its own [28].

### AnyCalib: Universal On-Manifold Calibration for Arbitrary Camera Models

AnyCalib (Tirado-Garin and Civera, ICCV 2025) provides model-agnostic camera calibration across arbitrary optical projections, removing the assumption that input imagery must conform to an ideal rectilinear pinhole model [8]. The codebase is available at <https://github.com/javrtg/AnyCalib> [8].

Built on on-manifold learning principles, AnyCalib formulates single-view calibration as regressing a continuous 2D field-of-view tangent field and a dense 3D viewing ray map (a unit direction per pixel) across all sensor pixels [8]. Rather than locking the output space to a single set of pinhole parameters, the predicted ray field is analytically fitted to user-specified or automatically selected camera projection models [8]:

- Pinhole cameras (`pinhole`) [8].
- Radial distortion models (`radial:k`, `simple_radial:k`) [8].
- Fisheye projections, including Kannala-Brandt (`kb:k`, `simple_kb:k`) [8].
- Unified Camera Models (UCM) and division distortion models [8].

AnyCalib accepts any single 2D image and outputs the target projection parameters (focal lengths, principal point coordinates, distortion coefficients), per-pixel FoV tangent representations, and ray direction vectors [8]. The system runs without manual line annotations, requiring only an optional flag designating the desired target camera model [8].

Evaluated on ScanNet++, MegaDepth, and MonoVO benchmarks, AnyCalib outperforms standard 3D foundation models when calibrating images affected by severe radial distortion, sensor cropping, or non-central optical axes [8]. On ScanNet++ fisheye sequences, its ray-prediction accuracy provides a stable calibration foundation, allowing models such as UniDAC to perform zero-shot metric depth estimation across arbitrary lens types [39].

It supports extreme wide-angle action cameras, drone perspectives, cropped internet images, and ultra-wide fisheye optics [8].

For Blender pipelines, pinhole and simple radial parameters export directly to native Blender camera attributes, while Kannala-Brandt and fisheye parameters can either undistort the source image prior to projection or drive Blender's Panoramic Fisheye camera model in Cycles [8].

The project is distributed under the Apache-2.0 License [8]. Inference executes in _[missing]_ on desktop GPUs, with a VRAM footprint below _[missing]_ [8].

AnyCalib focuses strictly on camera intrinsics, ray configurations, and lens distortion profiles, omitting world-space gravity orientation (pitch and roll) [8]. It is best utilised as an intrinsic pre-processor for distorted plates before downstream extrinsic orientation matching [8].

### UniDepth and UniDAC: Disentangled Pseudo-Spherical Metric Calibration

UniDepth (Piccinelli et al., CVPR 2024; UniDepthV2, 2025) and its multi-camera extension UniDAC (CVPR 2026) combine camera calibration with metric depth estimation by predicting camera intrinsics within the core depth pipeline [12]. The primary repository is hosted at <https://github.com/lpiccinelli-eth/UniDepth> [12], with UniDAC maintained at <https://github.com/girish1511/UniDAC> [39].

Rather than outputting Cartesian coordinates directly, UniDepth represents scenes using a pseudo-spherical parameterisation made of the horizontal azimuth and vertical elevation angles of each camera ray, plus log-metric depth [12]. The architecture incorporates a self-promptable camera module that bootstraps class tokens from a Vision Transformer (ViT-S/14 or ViT-L/14) through self-attention layers to predict four intrinsic scalar offsets (two focal lengths and two principal point coordinates) [12].

Backprojecting pixels through the inverse of the predicted intrinsic matrix yields unit-sphere rays that condition the dense depth decoder [12]. Predicting angular rays independently from metric depth values disentangles camera geometry from scene scale, preventing the two objectives from destabilising each other during training [12].

UniDepth accepts an uncalibrated RGB image and outputs a 3×3 intrinsic camera calibration matrix (K), a dense metric depth map, an uncertainty confidence field, and an unprojected 3D point cloud in camera coordinates [12]. The model runs without human intervention, though it can optionally ingest ground-truth intrinsics when available to further refine depth accuracy [12].

Evaluated on NYUv2, ScanNet++, and KITTI-360, UniDepth and UniDAC exhibit consistent zero-shot cross-dataset transfer [39]. UniDAC achieves a δ1 accuracy of _[missing]_ (with an absolute relative error of _[missing]_) on ScanNet++ fisheye indoor scenes [39].

The models handle rectilinear images, fisheye lenses, and spherical 360° inputs [39].

Predicted intrinsics map directly into Blender camera parameters, while the generated metric point clouds export to PLY format for use as spatial reference geometry [12].

The model code and weights are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0), restricting commercial production without custom licensing [41]. Inference takes _[missing]_ (ViT-S) to _[missing]_ (ViT-L) on an NVIDIA RTX 3080/4080 [40].

Like Depth Pro, UniDepth predicts geometry in camera-centric coordinates rather than a gravity-aligned world frame [12]. Establishing ground plane orientation requires post-processing via normal estimation or extrinsic optimisation [12].

### Puffin: Spatial Reasoning and Multimodal Thinking with Camera

Puffin (Liao et al., ICLR 2026) treats single-view camera estimation as a multimodal reasoning problem [19]. Developed by NTU S-Lab, SenseTime, and the Max Planck Institute, the model introduces the concept of "Thinking with Camera", where the network explicitly reasons about scene geometry before outputting numerical parameters [19]. The project repository is located at <https://github.com/KangLiao929/Puffin> [44].

Puffin integrates a vision encoder (SigLIP/DINOv2), an autoregressive Large Language Model (LLM), and a Diffusion Transformer (DiT) conditioned on camera geometry [44]. The system is trained across four stages on Puffin-4M, a curated dataset of four million vision-language-camera triplets containing ground-truth intrinsics, extrinsics, and geometric reasoning chains [44].

When processing an input image, Puffin generates an internal chain of geometric reasoning: it identifies spatial cues (for example, Dutch angles, converging architectural planes, horizon lines, and foreground-background depth ordering), maps them to photographic terminology, and predicts camera parameters as structured tokens [19] representing roll, pitch, yaw, focal length, and radial distortion [19].

Puffin accepts a single 2D image alongside an instruction prompt (such as _"Reason the spatial distribution of this image in a thinking mode, and then estimate its camera parameters"_) [19]. It outputs a textual spatial analysis followed by numerical predictions for roll, pitch, and FoV [19]. The pipeline is fully automated and supports instruction-based steering [45].

On the MegaDepth benchmark, Puffin achieves lower median errors for roll and pitch than specialised regression models, showing particular robustness on complex, asymmetrical, or non-Manhattan imagery where line-detection algorithms fail [19].

Supported imagery covers architectural scenes, natural landscapes, studio photography, and AI-generated content [19].

The structured token outputs parse into Python dictionaries for configuring camera transforms in Blender [19].

The project is open source and hosted on Hugging Face [44]. However, because it incorporates a multimodal LLM, hardware requirements are higher than for specialised models: running Puffin requires _[missing]_ of VRAM, with inference times between _[missing]_ and _[missing]_ per frame.

While Puffin provides detailed geometric reasoning, token generation introduces higher latency than lightweight architectures like GeoCalib. It replaces fSpy's core orientation and field-of-view estimation while adding natural-language explanations for the recovered parameters [19].

## Quantitative Benchmarks and Cross-Model Architectural Comparison

Evaluating neural calibration models requires assessing angular accuracy across extrinsic parameters (roll and pitch), field of view (FoV), and metric depth recovery [2]. In monocular setups, roll is generally the most constrained parameter because vertical visual cues directly indicate the direction of gravity [7]. Pitch and FoV exhibit higher coupling, as changing camera tilt can closely mimic shifts in focal length and vertical optical centring [11].

The following table summarises performance across the leading single-image calibration and monocular geometry systems, compiled from primary benchmark evaluations on MegaDepth, LaMAR, ScanNet++, and PPR10K:

| Model                         | Parameter outputs                                   | MegaDepth roll AUC (1°/5°/10°)              | MegaDepth pitch AUC (1°/5°/10°)                  | MegaDepth FoV AUC (1°/5°/10°)                  | Median error (roll / pitch / FoV)                    | Primary benchmark metric                                           | Latency and VRAM                   | Licence                           | End-to-end fSpy replacement?                                                   |
| :---------------------------- | :-------------------------------------------------- | :------------------------------------------ | :----------------------------------------------- | :--------------------------------------------- | :--------------------------------------------------- | :----------------------------------------------------------------- | :--------------------------------- | :-------------------------------- | :----------------------------------------------------------------------------- |
| **GeoCalib** (ECCV 2024) [7]  | Focal length, distortion, gravity (pitch, roll) [7] | _[missing]_ [7, 19]                         | _[missing]_ [7, 19]                              | _[missing]_ [7, 19]                            | _[missing]_ [19]                                     | Outperforms ParamNet and UVP by _[missing]_ at fine thresholds [7] | _[missing]_; _[missing]_ VRAM [17] | Apache-2.0 / CC-BY 4.0 [7]        | **Yes** (solves orientation and FoV; height requires plane fitting) [7]        |
| **MoGe-3** (MSRA 2026) [24]   | Metric point map, normals, hFoV [11, 14]            | Derived via normal fitting [11]             | Derived via normal fitting [11]                  | End-to-end projective ray solve [11]           | Sub-degree normal angular error [24]                 | Sets SOTA across 9 zero-shot geometry benchmarks [24]              | _[missing]_; _[missing]_ VRAM [24] | MIT [25]                          | **Partial** (direct proxy mesh and FoV; pitch/roll derived downstream) [11]    |
| **Depth Pro** (ICLR 2025) [9] | Metric depth, focal length (px) [28, 35]            | Not regressed [28]                          | Not regressed [28]                               | Derived from focal length and image width [34] | _[missing]_ on PPR10K focal split [33]               | SOTA zero-shot boundary F1 score and sharp edge tracing [28]       | _[missing]_; _[missing]_ VRAM [28] | Apple Sample Code / Research [28] | **Partial** (solves focal length and scale; requires gravity orientation) [28] |
| **AnyCalib** (ICCV 2025) [8]  | Pinhole, radial, fisheye [8]                        | Not regressed [8]                           | Not regressed [8]                                | Model-agnostic ray field fit [8]               | Mean FoV error _[missing]_ (MegaDepth pinhole) [2]   | Superior ray consistency on cropped and wide-angle optics [8]      | _[missing]_; _[missing]_ VRAM [8]  | Apache-2.0 [8]                    | **No** (specialised for complex intrinsics and non-pinhole lenses) [8]         |
| **UniDAC** (CVPR 2026) [39]   | Intrinsic matrix K, metric depth [12]               | View-space only [12]                        | View-space only [12]                             | Derived directly from the K matrix [12]        | δ1 _[missing]_, AbsRel _[missing]_ on ScanNet++ [39] | Outperforms Metric3D v2 on universal camera inputs [39]            | _[missing]_; _[missing]_ VRAM      | CC BY-NC-SA 4.0 [41]              | **Partial** (solves K and metric geometry; requires horizon alignment) [12]    |
| **Puffin** (ICLR 2026) [19]   | Roll, pitch, yaw, FoV, distortion [19]              | Competitive with specialised baselines [19] | Outperforms baselines on unconstrained sets [19] | Consistent with wide-angle splits [19]         | Direct language-token parameter output [19]          | SOTA unified camera-centric multimodal spatial reasoning [19]      | _[missing]_; _[missing]_ VRAM      | Research open source [44]         | **Yes** (generates camera rotation and FoV via semantic reasoning) [19]        |

Across these benchmarks, specialised geometric models demonstrate distinct trade-offs. GeoCalib provides the most accurate and reliable orientation recovery, achieving an _[missing]_ AUC at _[missing]_ on MegaDepth by enforcing physical constraints on the unit-sphere gravity manifold [7].

However, it does not infer metric depth or scene translation [7]. Conversely, Depth Pro and MoGe-3 excel at surface reconstruction and metric focal length estimation, but require downstream gravity alignment to match world coordinate axes [14].

These complementary strengths suggest that combining a specialised calibration model with a dense geometry estimator yields the most robust fully automated camera matching pipeline [7].

## Fully Automated Blender Integration Pipeline: The Zero-Click fSpy Architecture

### Mathematical Coordinate Transformation

Integrating neural camera predictions into Blender requires reconciling different coordinate conventions [6]. Computer vision libraries (OpenCV, COLMAP, GeoCalib) typically adopt a right-handed camera coordinate frame: +X right, +Y down, +Z forward (viewing direction).

Blender's camera object convention is also right-handed, but oriented differently: +X right, +Y up, −Z forward (viewing direction).

Blender's world space uses a Z-up convention: +X east, +Y north, +Z up.

Given GeoCalib's predicted gravity unit vector expressed in the vision camera frame, the camera's roll and pitch are derived analytically from its components [7].

Assuming the camera's heading (yaw) is aligned with world north (yaw = 0), the rotation aligning the camera to the world coordinate frame is a roll about the Z axis combined with a rotation of pitch + 90° about the X axis, as implemented in the exporter script below.

### Automated Ground Plane Estimation for Camera Height

To match camera height above the ground plane without manual picking, the input image is processed through Depth Pro or MoGe-3 to generate a dense metric point map in camera view-space [14].

The points are rotated into world orientation using this rotation.

A RANSAC estimator isolates the dominant planar surface in the lower third of the scene point cloud [16]. The fitted plane's normal represents the upright ground normal, and the camera's metric elevation above the floor is the plane's distance from the camera.

This establishes the complete 4×4 camera transformation matrix, combining the rotation with a translation that places the camera at this height.

### Automated fSpy JSON Exporter

The following script converts predicted camera parameters into fSpy-compatible JSON format, enabling direct import through the existing fSpy-Blender add-on without opening the standalone fSpy GUI [6]:

```python
import json
import numpy as np


def generate_fspy_calibration_file(
    output_json_path: str,
    image_width: int,
    image_height: int,
    vfov_degrees: float,
    roll_radians: float,
    pitch_radians: float,
    camera_height_meters: float = 1.65,
) -> None:
    """
    Synthesizes predicted camera parameters into an fSpy-compliant JSON file.
    Matches standard coordinate conventions for direct ingestion via fSpy-Blender.
    """
    vfov_rad = np.deg2rad(vfov_degrees)
    aspect_ratio = image_width / image_height
    hfov_rad = 2.0 * np.arctan(np.tan(vfov_rad / 2.0) * aspect_ratio)

    # Rotation matching Blender camera space:
    # Camera views down -Z, with +Y up and +X right.
    # World coordinate frame is Z-up.
    cos_p, sin_p = np.cos(pitch_radians + np.pi / 2.0), np.sin(pitch_radians + np.pi / 2.0)
    cos_r, sin_r = np.cos(roll_radians), np.sin(roll_radians)

    # 3x3 camera-to-world rotation matrix
    R = np.array([
        [cos_r, -sin_r * cos_p,  sin_r * sin_p],
        [sin_r,  cos_r * cos_p, -cos_r * sin_p],
        [  0.0,          sin_p,          cos_p],
    ], dtype=np.float64)

    # Translation placing camera at metric elevation above the origin
    t = np.array([0.0, 0.0, float(camera_height_meters)], dtype=np.float64)

    transform_matrix = np.eye(4, dtype=np.float64)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = t

    fspy_payload = {
        "imageWidth": int(image_width),
        "imageHeight": int(image_height),
        "horizontalFieldOfView": float(hfov_rad),
        "verticalFieldOfView": float(vfov_rad),
        "vanishingPoints": [],
        "hasVerticalAxis": True,
        "principalPoint": {
            "x": image_width / 2.0,
            "y": image_height / 2.0,
        },
        "cameraTransform": {
            "rows": transform_matrix.tolist(),
        },
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(fspy_payload, f, indent=4)
```

### Headless Blender Automation Script

For integrated production pipelines, the following script executes in headless Blender (`blender --background --python setup_matched_camera.py`), automatically building the camera, configuring viewport projection plates, and setting up ground plane geometry [13]:

```python
import bpy
import mathutils
import sys


def build_automated_blender_scene(
    image_filepath: str,
    width_px: int,
    height_px: int,
    vfov_rad: float,
    transform_matrix_4x4: list,
    output_blend_path: str,
) -> None:
    # Reset scene to a clean default state
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # Align render aspect ratio and dimensions with the input image
    scene.render.resolution_x = width_px
    scene.render.resolution_y = height_px
    scene.render.resolution_percentage = 100

    # Instantiate and configure the matched camera
    cam_data = bpy.data.cameras.new(name="AI_Matched_Camera")
    cam_data.sensor_fit = 'VERTICAL'
    cam_data.angle_y = vfov_rad
    cam_data.show_background_images = True

    # Mount the input image as a camera background reference plate
    bg_image = cam_data.background_images.new()
    loaded_img = bpy.data.images.load(image_filepath)
    bg_image.image = loaded_img
    bg_image.frame_method = 'CROP'
    bg_image.display_depth = 'BACK'

    cam_object = bpy.data.objects.new(name="AI_Camera", object_data=cam_data)
    scene.collection.objects.link(cam_object)
    scene.camera = cam_object

    # Set world transform from the solved matrix
    cam_object.matrix_world = mathutils.Matrix(transform_matrix_4x4)

    # Generate a reference ground plane aligned with Z=0
    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0.0, 0.0, 0.0))
    floor_grid = bpy.context.active_object
    floor_grid.name = "Reference_Floor_Plane"
    floor_grid.display_type = 'WIRE'

    # Save the configured scene
    bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)


if __name__ == "__main__":
    pass
```

## Systematic Failure Modes, Edge Cases, and Operational Safeguards

While deep neural models remove the need for manual line alignment, they introduce specific edge cases that require operational handling:

```text
[ Input Plate Evaluation ]
│
├─ Is principal point centred?
│  ├── No  ──> Pre-process intrinsics using AnyCalib
│  └── Yes ──> Proceed to GeoCalib / MoGe-3
│
├─ Are reflective/specular surfaces dominant?
│  ├── Yes ──> Rely on GeoCalib perspective fields; disable dense depth
│  └── No  ──> Run joint GeoCalib + Depth Pro / MoGe-3
│
└─ Does scene contain known physical scale references?
   ├── Yes ──> Scale camera tz against reference object
   └── No  ──> Use default ground-plane height prior (1.65 m)
```

### Optical Centre Decentring and Cropped Plates

Most single-view calibration architectures (including GeoCalib and Depth Pro) assume that the optical centre (principal point) is located at the centre of the image canvas [7]. When an image has undergone off-centre cropping, digital panning, or asymmetric lens shifting, this assumption is violated [8].

A decentred principal point creates an asymmetric perspective field, which models often misinterpret as camera pitch or roll [8]. In pipelines dealing with heavily cropped plates, AnyCalib or UniDepth should be used as an initial calibration stage, as their ray-direction formulations can solve for horizontal and vertical optical centre offsets directly [8].

### Scale Ambiguity and Ground Plane Identification

Monocular depth models estimate metric scale by learning typical physical sizes for familiar objects (such as vehicles, furniture, and people) [31]. In environments lacking standard semantic scale cues, such as aerial drone photography, extreme close-up macro shots, or abstract architectural scenes, the predicted metric scale can drift [35].

While the camera's angular parameters (pitch, roll, and FoV) remain stable, the absolute camera height may scale incorrectly [35].

When processing imagery without recognisable scale anchors, pipelines should incorporate an interactive scale-adjustment factor or default to a standard standing eye-level prior of about 1.65 m [35].

### Specular, Reflective, and Translucent Surfaces

Dense depth estimation models (Depth Pro, MoGe-3, and UniDepth) compute depth by analysing surface textures and occluding boundaries [24]. Environments with extensive reflective or translucent surfaces, such as wet streets, large bodies of water, mirrors, or glass facades, violate standard Lambertian assumptions [9].

This can lead to noisy depth values and flying-pixel artefacts near surface transitions [9].

In scenes dominated by reflections or transparency, camera matching should rely primarily on GeoCalib's global perspective field, which infers orientation from overall scene structure rather than per-pixel surface depths [16].

### Radial and Non-Rectilinear Lens Distortions

Standard pinhole projection models struggle when applied to images with noticeable barrel or pincushion distortion [2]. If an image with barrel distortion is fed directly to a rectilinear solver, straight lines appear curved, leading to inaccurate FoV and pitch estimates [7].

To handle distorted imagery, plates should first be processed through GeoCalib (using its `--camera_model simple_radial` mode) or AnyCalib (using `kb:4` or division models) to estimate radial distortion coefficients [7].

The plate can then be digitally undistorted before camera matching, or mapped directly to Blender Cycles' native Panoramic Fisheye camera model [7].

## Practical Synthesis and Implementation Strategies

For visual effects artists, architectural visualisers, and general 3D practitioners seeking an automated alternative to fSpy:

- **Direct drop-in replacement for fSpy:** GeoCalib provides the most direct functional equivalent to fSpy [7]. It infers camera roll, pitch, focal length, and radial distortion across indoor, outdoor, natural, and architectural scenes without manual vanishing line input [7]. Its fast inference speed (_[missing]_), low VRAM footprint (_[missing]_), and Apache-2.0 licence make it well-suited for interactive tools and add-on development [7].
- **Joint scene reconstruction and camera matching:** Microsoft's MoGe-3 is the preferred solution when both camera matching and proxy geometry generation are required [14]. It recovers fine geometric details, surface normal maps, and camera FoV within a single forward pass, exporting textured `.glb` assets that load directly into Blender [14].
- **High-resolution plates and detail preservation:** Apple Depth Pro provides the highest edge definition and boundary sharpness for high-resolution images, generating _[missing]_ metric depth maps and horizontal focal lengths in _[missing]_ [28]. It is particularly effective for portraiture, product rendering, and scenes with fine silhouette detail [28].
- **The composite production pipeline:** The most robust automated solution pairs **GeoCalib** with **MoGe-3** or **Depth Pro** [7]. In this configuration, GeoCalib determines camera orientation (pitch and roll) and field of view, while the dense depth model reconstructs the visible environment and establishes camera height via automated ground-plane fitting [7]. This two-stage pipeline eliminates manual vanishing-point placement while delivering a fully calibrated camera and matching proxy scene in Blender [13].

## Works Cited

1. [GeoCalib: Learning Single-image Calibration with Geometric Optimization](https://hub.baai.ac.cn/paper/4e5c7bc1-d5fe-41db-b2ec-d9cec2e2cd5e)
2. [CalibAnyView: Beyond Single-View Camera Calibration in the Wild](https://arxiv.org/html/2605.14615v1)
3. [blender-addon · GitHub Topics](https://github.com/topics/blender-addon)
4. [Resources - Arts 'n Science](https://artsnscience.eu/resources/)
5. [Deep Learning for Camera Calibration and Beyond: A Survey - arXiv](https://arxiv.org/html/2303.10559v3)
6. [Import fSpy JSON to Nuke: load a json file with camera ... - GitHub Gist](https://gist.github.com/MitchellKehn/e7ccdfad932886c7e9e4c072a08006c8)
7. [GeoCalib: Single-image Calibration with Geometric Optimization](https://github.com/cvg/GeoCalib)
8. [AnyCalib: On-Manifold Learning for Model-Agnostic Single ... - GitHub](https://github.com/javrtg/AnyCalib)
9. [Depth Pro: Sharp Monocular Metric Depth in ... - HyperAI](https://hyper.ai/en/papers/iclr__2025__aueXfY0Clv)
10. [Minimal Solvers for Single-View Lens-Distorted Camera Auto-Calibration](https://www.researchgate.net/publication/352392703_Minimal_Solvers_for_Single-View_Lens-Distorted_Camera_Auto-Calibration)
11. [MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Wang_MoGe_Unlocking_Accurate_Monocular_Geometry_Estimation_for_Open-Domain_Images_with_CVPR_2025_paper.pdf)
12. [bdck/unidepth-inference - Hugging Face](https://huggingface.co/bdck/unidepth-inference)
13. [atlas-camera - ComfyUI Cloud](https://comfy.icu/extension/mikejamesvfx__atlas-camera)
14. [MoGe: Accurate Monocular Geometry Estimation - GitHub](https://github.com/microsoft/moge)
15. [Paper page - GeoCalib: Learning Single-image Calibration with Geometric Optimization](https://huggingface.co/papers/2409.06704)
16. [Learning Single-image Calibration with Geometric Optimization (supplementary)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05636-supp.pdf)
17. [GeoCalib: Learning Single-image Calibration with Geometric ... - arXiv](https://arxiv.org/html/2409.06704v2)
18. [Burgstall-labs/ComfyUI-VR-Outpaint-Tools - GitHub](https://github.com/Burgstall-labs/ComfyUI-VR-Outpaint-Tools)
19. [Thinking with Camera: A Unified Multimodal Model for Camera-Centric Understanding and Generation](https://proceedings.iclr.cc/paper_files/paper/2026/file/e9882f7f7c44a10acc01132302bac9d8-Paper-Conference.pdf)
20. [Min Float - ComfyUI Cloud - Comfy.ICU](https://comfy.icu/node/SL_MinFloat)
21. [GeoCalib → fSpy JSON - ComfyUI Cloud](https://comfy.icu/node/GeoCalibToFSpy)
22. [Atlas Export Relief Mesh (OBJ) - ComfyUI Cloud](https://comfy.icu/node/AtlasExportReliefMesh)
23. [MoGe - Ruicheng Wang](https://wangrc.site/MoGePage/)
24. [MoGe-3 - Ruicheng Li](https://qft-333.github.io/moge3page/)
25. [pyproject.toml - microsoft/MoGe - GitHub](https://github.com/microsoft/MoGe/blob/main/pyproject.toml)
26. [MoGe: Accurate Monocular Geometry Estimation - ModelScope](https://www.modelscope.cn/models/Comfy-Org/MoGe)
27. [MoGe: Revolutionizing 3D Geometry Estimation from Single Images](https://dev.to/githubopensource/moge-revolutionizing-3d-geometry-estimation-from-single-images-38bf)
28. [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second - GitHub](https://github.com/apple-aiml-research/ml-depth-pro)
29. [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second - Hugging Face](https://huggingface.co/papers/2410.02073)
30. [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second - Medium](https://medium.com/@jiangmen28/apple-depth-pro-sharp-monocular-metric-depth-in-less-than-a-second-bd020a4c3ae7)
31. [Estimating Depth and Focal Length with Apple Depth-Pro - Medium](https://medium.com/@Nivitus./estimating-depth-and-focal-length-with-apple-depth-pro-e49a19392b47)
32. [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second - arXiv v1](https://arxiv.org/html/2410.02073v1)
33. [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second - paper review](https://andlukyane.com/blog/paper-review-depthpro)
34. [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second - arXiv v2](https://arxiv.org/html/2410.02073v2)
35. [Apple Depth Pro: sharp metric depth from one image - AI IDE List](https://aiidelist.com/blog/depth-pro)
36. [Best Depth Estimation Models: Depth Anything V2 and More](https://blog.roboflow.com/depth-estimation-models/)
37. [ComfyUI-Depth-Pro - GitHub](https://github.com/spacepxl/ComfyUI-Depth-Pro)
38. [AnyCalib - ICCV 2025 Open Access Repository](https://openaccess.thecvf.com/content/ICCV2025/html/Tirado-Garin_AnyCalib_On-Manifold_Learning_for_Model-Agnostic_Single-View_Camera_Calibration_ICCV_2025_paper.html)
39. [girish1511/UniDAC - GitHub](https://github.com/girish1511/UniDAC)
40. [lpiccinelli-eth/UniDepth: Universal Monocular Metric Depth Estimation](https://github.com/lpiccinelli-eth/unidepth)
41. [UniDepth: Universal Monocular Metric Depth Estimation - arXiv](https://arxiv.org/html/2403.18913v1)
42. [UniDepthV2: Universal Monocular Metric Depth Estimation ... - arXiv](https://arxiv.org/html/2502.20110v1)
43. [Estimate multiple camera poses and reconstruct the dynamic scene](https://github.com/ljjTYJR/multiple-view-dynamic-reconstruction)
44. [Puffin Series: Towards Unified Multimodal 3D World Models - GitHub](https://github.com/KangLiao929/Puffin)
45. [A Unified Multimodal Model for Camera-Centric Understanding and Generation - arXiv v1](https://arxiv.org/html/2510.08673v1)
46. [Puffin - Thinking with Camera: A Unified Multimodal Model for Camera-Centric Understanding and Generation](https://kangliao929.github.io/projects/puffin/)
47. [A Unified Multimodal Model for Camera-Centric Understanding and Generation - arXiv v2](https://arxiv.org/html/2510.08673v2)
48. [Yikai Wang - alphaXiv](https://www.alphaxiv.org/@yikai-wang-3)
49. [Robust Monocular Depth Estimation Under Crop-Resize-Induced Distortion - MDPI](https://www.mdpi.com/2079-9292/15/10/2180)
50. [Extract matrix from JSON file - Stack Overflow](https://stackoverflow.com/questions/58361808/extract-matrix-from-json-file)
51. [ORBIT: Benchmarking SfM in the Wild with 360° Video - OpenReview](https://openreview.net/pdf?id=R48lWX6QMt)
