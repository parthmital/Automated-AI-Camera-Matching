# **Automated Single-Image Camera Matching and Inverse Rendering for 3D Production**

## **Evolution from Classical Vanishing Geometry to Deep Invariant Representations**

Single-view camera calibration in visual effects (VFX), computer animation, and architectural visualization has historically depended on classical projective geometry1. Utilities such as fSpy—and its predecessor BLAM—rely on user-defined vanishing lines to compute vanishing points along mutually orthogonal axes2. From these intersections, the camera’s focal length, field of view (FoV), and spatial orientation (pitch, roll, and yaw) are calculated via closed-form projective equations2.  
While mathematically exact under noise-free conditions, this manual paradigm imposes severe operational limitations1:

- It requires structured Manhattan or Atlanta world geometries with visible, unoccluded parallel lines2.
- It assumes an idealized rectilinear pinhole camera with an unshifted optical center and zero lens distortion2.
- It demands human intervention to manually place alignment gizmos, precluding batch processing or real-time pipelines1.
- It fails when applied to organic landscapes, close-up portraits, non-rectilinear architectural environments, or cropped imagery8.

Recent deep learning models reformulate monocular camera calibration and inverse rendering by learning continuous geometric representations directly from high-capacity vision backbones7. Instead of extracting fragile 2D line segments, modern networks predict dense, per-pixel geometric representations—including gravity up-vectors, latitude fields, optical ray directions, and dense metric point surfaces7.  
These models decouple camera calibration from strict structural scene assumptions7. By integrating these neural estimators with automated metric ground-plane solvers and headless 3D engine bridges, technical artists can now take a single arbitrary 2D image, estimate its complete intrinsic and extrinsic parameters, and instantiate a fully matched virtual camera in Blender without manual intervention7.

## **Deep Architectural Approaches to Monocular Calibration and Geometry**

### **GeoCalib: Second-Order Geometric Optimization on Neural Perspective Fields**

GeoCalib, introduced by Veicht et al. at ECCV 2024, bridges deep semantic feature extraction with classical non-linear optimization to estimate camera intrinsics and absolute gravity orientation from a single uncalibrated plate7. The primary codebase and model checkpoints are available in the official repository (https://github.com/cvg/GeoCalib)7.  
The architecture addresses the poor generalization typical of direct parameter regression networks, which routinely overfit to specific camera distributions1. GeoCalib employs a SegNeXt-based convolutional vision backbone that outputs two dense perspective vector fields alongside per-pixel uncertainty estimations16:  
![][image1]  
The up-vector field ![][image2] defines the projected 2D direction of world gravity at pixel coordinate ![](<>), while the latitude field ![](<>) denotes the elevation angle of the camera ray corresponding to pixel ![](<>) relative to the horizontal plane16. Rather than treating these maps as heuristic visualizations, GeoCalib embeds a differentiable Levenberg-Marquardt (LM) optimizer directly into the training loop16.  
The optimizer minimizes a confidence-weighted non-linear least-squares residual between the observed fields and the analytic projection induced by parameter vector ![](<>), where focal length ![](<>) is optimized in log-space, gravity ![](<>) is parameterized on the unit two-sphere manifold ![](<>), and distortion is parameterized via polynomial coefficients ![](<>)16.

E(θ) \= Σ \[ σ\_u \* ||u(θ) \- û||² \+ σ\_φ \* ||sin φ(θ) \- sin φ̂||² \]

Because gradients propagate backwards through the unrolled optimizer iterations during training, the SegNeXt encoder learns to prioritize informative geometric visual anchors—such as vertical architecture, human figures, trees, and horizon gradients—while assigning high variance (low confidence weight) to ambiguous or non-upright surfaces16.  
GeoCalib accepts an unconstrained single RGB image of arbitrary resolution7. The forward pass outputs horizontal and vertical focal lengths (![](<>)), the vertical and horizontal field of view (![](<>)), lens distortion parameters (![](<>)), the world-space gravity vector ![](<>) (from which camera roll ![](<>) and pitch ![](<>) are directly solved), and per-pixel uncertainty heatmaps7. The model operates completely autonomously, executing feedforward inference without initial manual bounds or seed coordinates7.  
Evaluated on the MegaDepth, LaMAR, Stanford2D3D, and TartanAir benchmarks, GeoCalib achieves substantial performance improvements over prior single-image calibration models7. On MegaDepth, GeoCalib achieves an Area Under the Curve (AUC) for Roll at ![](<>) error thresholds of ![](<>) (median error ![](<>)), Pitch AUC of ![](<>) (median error ![](<>)), and FoV AUC of ![](<>) (median error ![](<>))7. On LaMAR indoor AR sequences, its Roll AUC reaches ![](<>) with a Pitch AUC of ![](<>)7.  
The model supports standard rectilinear pinhole images, images exhibiting moderate radial distortion, and division-model optics7. It handles natural outdoor environments, domestic interiors, and distorted wide-angle shots7.  
Blender integration is well-supported within the community ecosystem13. The open-source Atlas Camera extension incorporates GeoCalib to automate camera setups in headless Blender (![](<>)) sessions13, while custom ComfyUI nodes serialise GeoCalib's predicted pitch, roll, and FoV directly into fSpy-formatted JSON strings20.  
The underlying code is licensed under Apache-2.0, while model weights are distributed under the Creative Commons Attribution 4.0 International License (CC-BY 4.0), permitting unencumbered commercial and studio deployment7. Hardware demands are modest: GeoCalib requires approximately ![](<>) of VRAM and achieves an average inference latency of ![](<>) on an NVIDIA RTX 3090/4090 GPU17.  
The primary operational limitation of GeoCalib is its structural assumption that the principal point coincides with the geometric center of the sensor (![](<>)), meaning heavy asymmetric crops induce angular drift in estimated pitch7. Additionally, while it fully resolves camera rotation and focal length, it does not infer metric translation ![](<>) (such as camera height above the ground plane)7.  
Consequently, while it functions as a drop-in replacement for fSpy's vanishing-point orientation and FoV extraction, full spatial positioning requires pairing with an automated ground-plane depth detector7.

### **MoGe Series (MoGe-2 & MoGe-3): Monocular Geometry Foundation Models**

Developed by Microsoft Research, the MoGe family reformulates monocular 3D perception by predicting continuous metric point maps, surface normal fields, and camera FoV within a single forward pass11. The project is documented across foundational publications: _MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision_ (Wang et al., CVPR 2025 Oral) and _MoGe-3: Fine-Detail Monocular Geometry Estimation with Self-Guided Sparse Volumetric Refinement_ (Kong et al., 2026\)11. The implementation is maintained at https://github.com/microsoft/MoGe14.  
MoGe-1 and MoGe-2 pair a high-capacity DINOv2 Vision Transformer encoder (ViT-L/ViT-G) with a convolutional upsampling decoder11. To overcome scale-shift ambiguities during monocular training without discarding geometric detail, the architecture implements a closed-form Robust, Optimal, and Efficient (ROE) alignment solver11. This enforces affine-invariant 3D point-map supervision across multi-scale spherical neighborhoods, penalizing surface normal discrepancies and geometric warping11.  
MoGe-3 introduces Self-Guided Sparse 3D Refinement (SSR)24. Standard 2D convolutional or attention decoders bleed features across sharp depth boundaries, rounding thin geometric structures (such as poles, foliage, and structural edges)24. MoGe-3 resolves this by unprojecting the base point map into a sparse 3D voxel shell and iteratively applying sparse 3D convolutions24. Because foreground edges and distant background regions occupy physically distinct voxels in 3D Euclidean space, feature bleeding across occluding contours is eliminated24.  
Simultaneously, an auxiliary projection head recovers the horizontal field of view (![](<>)) by enforcing projective consistency across the reconstructed 3D points11.  
MoGe ingests an uncalibrated RGB image of arbitrary resolution and aspect ratio14. It outputs a dense metric 3D point map ![](<>), a metric depth map, an aligned surface normal map, a sky/infinity semantic mask, and camera horizontal/vertical FoV values11. The pipeline is fully automated and supports execution via a CLI that converts predicted point geometry into textured .glb 3D assets14.  
MoGe achieves high geometric accuracy across unseen benchmarks, setting the current state of the art in open-domain point map and normal estimation14. On zero-shot evaluations across nine public datasets, MoGe-3 matches or exceeds the boundary recall of specialized depth models while generating lower point-cloud distortion and preserving fine architectural details24.  
The model supports unconstrained perspective photography, macro shots, wide environments, and full ![](<>) equirectangular panoramas via a dedicated spherical parameterization pipeline (moge infer\_panorama)14.  
Blender integration is direct: because the model can export textured .glb meshes embedding metric coordinates alongside computed FoV parameters, importing the resulting asset into Blender instantiates both the scene geometry and an aligned camera frustum14.  
MoGe is licensed under the permissive MIT License, enabling integration into commercial and proprietary pipelines25. On an NVIDIA RTX 4090, MoGe-2 (ViT-L) runs in ![](<>) per image, while MoGe-3 (with three SSR refinement iterations) executes in ![](<>), requiring ![](<>) of VRAM under half precision (FP16)14.  
The model's primary constraint regarding automated camera matching is that it does not output explicit camera pitch and roll angles as direct scalar variables11. These extrinsic orientations must be derived downstream by extracting the normal vector of the ground plane from the predicted point map or normal map11.  
When combined with an automated plane-fitting script, MoGe replaces fSpy's functionality while delivering both the camera solve and an aligned 3D proxy mesh11.

### **Apple Depth Pro: High-Resolution Metric Depth and Decoupled Focal Length Estimation**

Depth Pro, developed by Apple Research (Bochkovskii et al., ICLR 2025), is a metric monocular depth foundation model capable of inferring absolute metric scale alongside focal length from an uncalibrated single image9. The official implementation is accessible at https://github.com/apple-aiml-research/ml-depth-pro28.  
The architecture utilizes a multi-scale Vision Transformer design optimized to process global perspective context alongside high-frequency boundary details28. The input image is downsampled into hierarchical spatial representations, split into patches, processed via weight-shared ViT encoders, and synthesized via a Dense Prediction Transformer (DPT) decoder30.  
To overcome the scale ambiguity inherent in monocular vision, Depth Pro decouples focal length estimation from dense depth estimation30. Rather than training the two heads jointly—which introduces gradient interference and degrades metric stability—Depth Pro freezes the core depth encoder and trains a separate focal length estimation head using large image datasets with EXIF metadata30.  
The predicted focal length in pixels (![](<>)) scales the canonical inverse depth map ![](<>) into an absolute metric depth map ![](<>)9:  
![](<>)  
Depth Pro takes a single RGB image and outputs a ![](<>) metric depth map (![](<>) native resolution) with absolute physical scale in meters, alongside the estimated horizontal focal length ![](<>)28. Operation is entirely automatic, requiring no prior EXIF tags or user hints28.  
Depth Pro exhibits high zero-shot boundary tracing precision, avoiding flying-pixel edge artifacts on intricate silhouettes such as hair, power lines, and foliage9. On zero-shot focal length benchmarks, it outperforms competing methods9. On the PPR10K evaluation benchmark, ![](<>) of Depth Pro's focal length predictions achieve an error within ![](<>) of ground truth (![](<>)), compared to ![](<>) for the previous state of the art (SPEC)33.  
It reliably processes portraits, architectural interiors, landscapes, and street photography9.  
Blender integration is implemented through custom nodes (such as ComfyUI-Depth-Pro) and Python unprojection scripts37. The predicted focal length ![](<>) maps directly to Blender's millimeter-based focal length:  
![](<>)  
The metric depth map unprojects to an absolute point cloud where 1 Blender unit equals 1 real-world meter28.  
The codebase is released under the Apple Sample Code License, while the model weights are covered by an accompanying research agreement that restricts unapproved commercial redistribution28.  
Inference executes in ![](<>) on an NVIDIA V100/A100 or RTX 4090 GPU, requiring ![](<>) of VRAM, with additional compatibility across Apple Silicon via Metal Performance Shaders (MPS) and ONNX/WebGPU runtimes28.  
Depth Pro's main limitation is that it regresses focal length and metric depth along camera viewing rays without predicting world-space gravity orientation (camera pitch and roll)28. Consequently, while it resolves fSpy’s focal length estimation and establishes scene scale, it cannot orient the camera relative to a world horizon on its own28.

### **AnyCalib: Universal On-Manifold Calibration for Arbitrary Camera Models**

AnyCalib (Tirado-Garin and Civera, ICCV 2025\) provides model-agnostic camera calibration across arbitrary optical projections, removing the assumption that input imagery must conform to an ideal rectilinear pinhole model8. The codebase is available at https://github.com/javrtg/AnyCalib8.  
Built on on-manifold learning principles, AnyCalib formulates single-view calibration as regressing a continuous 2D field-of-view tangent field and a dense 3D viewing ray map (![](<>)) across all sensor pixels8. Rather than locking the output space to a single set of pinhole parameters, the predicted ray field is analytically fitted to user-specified or automatically selected camera projection models8:

- Pinhole cameras (![](<>))8.
- Radial distortion models (radial:k, simple\_radial:k)8.
- Fisheye projections, including Kannala-Brandt (kb:k, simple\_kb:k)8.
- Unified Camera Models (UCM) and division distortion models8.

AnyCalib accepts any single 2D image and outputs the target projection parameters (focal lengths, principal point coordinates, distortion coefficients), per-pixel FoV tangent representations, and ray direction vectors8. The system runs without manual line annotations, requiring only an optional flag designating the desired target camera model8.  
Evaluated on ScanNet++, MegaDepth, and MonoVO benchmarks, AnyCalib outperforms standard 3D foundation models when calibrating images affected by severe radial distortion, sensor cropping, or non-central optical axes8. On ScanNet++ fisheye sequences, its ray-prediction accuracy provides a stable calibration foundation, allowing models such as UniDAC to perform zero-shot metric depth estimation across arbitrary lens types39.  
It supports extreme wide-angle action cameras, drone perspectives, cropped internet images, and ultra-wide fisheye optics8.  
For Blender pipelines, pinhole and simple radial parameters export directly to native Blender camera attributes, while Kannala-Brandt and fisheye parameters can either undistort the source image prior to projection or drive Blender's Panoramic Fisheye camera model in Cycles8.  
The project is distributed under the Apache-2.0 License8. Inference executes in ![](<>) on desktop GPUs, with a VRAM footprint below ![](<>)8.  
AnyCalib focuses strictly on camera intrinsics, ray configurations, and lens distortion profiles, omitting world-space gravity orientation (pitch and roll)8. It is best utilized as an intrinsic pre-processor for distorted plates before downstream extrinsic orientation matching8.

### **UniDepth and UniDAC: Disentangled Pseudo-Spherical Metric Calibration**

UniDepth (Piccinelli et al., CVPR 2024; UniDepthV2, 2025\) and its multi-camera extension UniDAC (CVPR 2026\) combine camera calibration with metric depth estimation by predicting camera intrinsics within the core depth pipeline12. The primary repository is hosted at https://github.com/lpiccinelli-eth/UniDepth12, with UniDAC maintained at https://github.com/girish1511/UniDAC39.  
Rather than outputting Cartesian coordinates directly, UniDepth represents scenes using a pseudo-spherical parameterization ![](<>), where ![](<>) and ![](<>) represent the horizontal azimuth and vertical elevation angles of each camera ray, and ![](<>) represents log-metric depth12. The architecture incorporates a self-promptable camera module that bootstraps class tokens from a Vision Transformer (ViT-S/14 or ViT-L/14) through self-attention layers to predict four intrinsic scalar offsets ![](<>)12.  
Backprojecting pixels through the inverted predicted intrinsic matrix ![](<>) yields unit-sphere rays that condition the dense depth decoder12. Predicting angular rays independently from metric depth values disentangles camera geometry from scene scale, preventing the two objectives from destabilizing each other during training12.  
UniDepth accepts an uncalibrated RGB image and outputs a ![](<>) intrinsic camera calibration matrix ![](<>), a dense metric depth map, an uncertainty confidence field, and an unprojected 3D point cloud in camera coordinates12. The model runs without human intervention, though it can optionally ingest ground-truth intrinsics when available to further refine depth accuracy12.  
Evaluated on NYUv2, ScanNet++, and KITTI-360, UniDepth and UniDAC exhibit consistent zero-shot cross-dataset transfer39. UniDAC achieves a ![](<>) accuracy of ![](<>) (with an absolute relative error of ![](<>)) on ScanNet++ fisheye indoor scenes39.  
The models handle rectilinear images, fisheye lenses, and spherical ![](<>) inputs39.  
Predicted intrinsics map directly into Blender camera parameters, while the generated metric point clouds export to PLY format for use as spatial reference geometry12.  
The model code and weights are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0), restricting commercial production without custom licensing41. Inference takes ![](<>) (ViT-S) to ![](<>) (ViT-L) on an NVIDIA RTX 3080/408040.  
Like Depth Pro, UniDepth predicts geometry in camera-centric coordinates rather than a gravity-aligned world frame12. Establishing ground plane orientation requires post-processing via normal estimation or extrinsic optimization12.

### **Puffin: Spatial Reasoning and Multimodal Thinking with Camera**

Puffin (Liao et al., ICLR 2026\) treats single-view camera estimation as a multimodal reasoning problem19. Developed by NTU S-Lab, SenseTime, and the Max Planck Institute, the model introduces the concept of "Thinking with Camera," where the network explicitly reasons about scene geometry before outputting numerical parameters19. The project repository is located at https://github.com/KangLiao929/Puffin44.  
Puffin integrates a vision encoder (SigLIP/DINOv2), an autoregressive Large Language Model (LLM), and a Diffusion Transformer (DiT) conditioned on camera geometry44. The system is trained across four stages on Puffin-4M, a curated dataset of four million vision-language-camera triplets containing ground-truth intrinsics, extrinsics, and geometric reasoning chains44.  
When processing an input image, Puffin generates an internal chain of geometric reasoning: it identifies spatial cues (e.g., Dutch angles, converging architectural planes, horizon lines, foreground-background depth ordering), maps them to photographic terminology, and predicts camera parameters as structured tokens19:  
![](<>)  
representing roll, pitch, yaw, focal length, and radial distortion19.  
Puffin accepts a single 2D image alongside an instruction prompt (such as "Reason the spatial distribution of this image in a thinking mode, and then estimate its camera parameters")19. It outputs a textual spatial analysis followed by numerical predictions for roll, pitch, and FoV19. The pipeline is fully automated and supports instruction-based steering45.  
On the MegaDepth benchmark, Puffin achieves lower median errors for roll and pitch than specialized regression models, showing particular robustness on complex, asymmetrical, or non-Manhattan imagery where line-detection algorithms fail19.  
Supported imagery covers architectural scenes, natural landscapes, studio photography, and AI-generated content19.  
The structured token outputs parse into Python dictionaries for configuring camera transforms in Blender19.  
The project is open source and hosted on Hugging Face44. However, because it incorporates a multimodal LLM, hardware requirements are higher than for specialized models: running Puffin requires ![](<>) of VRAM, with inference times between ![](<>) and ![](<>) per frame.  
While Puffin provides detailed geometric reasoning, token generation introduces higher latency than lightweight architectures like GeoCalib. It replaces fSpy's core orientation and field-of-view estimation while adding natural-language explanations for the recovered parameters19.

## **Quantitative Benchmarks and Cross-Model Architectural Comparison**

Evaluating neural calibration models requires assessing angular accuracy across extrinsic parameters (roll ![](<>) and pitch ![](<>)), field of view (![](<>)), and metric depth recovery2. In monocular setups, roll is generally the most constrained parameter because vertical visual cues directly indicate the direction of gravity7. Pitch and FoV exhibit higher coupling, as changing camera tilt can closely mimic shifts in focal length and vertical optical centering11.  
The following table summarizes performance across the leading single-image calibration and monocular geometry systems, compiled from primary benchmark evaluations on MegaDepth, LaMAR, ScanNet++, and PPR10K:

| Model Architecture          | Parameter Outputs                                      | MegaDepth Roll AUC (1∘/5∘/10∘)           | MegaDepth Pitch AUC (1∘/5∘/10∘)               | MegaDepth FoV AUC (1∘/5∘/10∘)          | Median Error (Roll / Pitch / FoV)           | Primary Benchmark Metric                                   | Latency & Hardware Footprint | Software License               | End-to-End fSpy Replacement?                                                   |
| :-------------------------- | :----------------------------------------------------- | :--------------------------------------- | :-------------------------------------------- | :------------------------------------- | :------------------------------------------ | :--------------------------------------------------------- | :--------------------------- | :----------------------------- | :----------------------------------------------------------------------------- |
| **GeoCalib** (ECCV 2024\)7  | ![](<>) (Pitch, Roll)7                                 | ![](<>) \[cite: 7, 19\]                  | ![](<>) \[cite: 7, 19\]                       | ![](<>) \[cite: 7, 19\]                | ![](<>) \[cite: 19\]                        | Outperforms ParamNet & UVP by ![](<>) at fine thresholds7  | ![](<>); ![](<>) VRAM17      | Apache-2.0 / CC-BY 4.07        | **Yes** (Solves orientation & FoV; height requires plane fitting)7             |
| **MoGe-3** (MSRA 2026\)24   | Metric Point Map, Normals, ![](<>) \[cite: 11, 14\]    | Derived via normal fitting11             | Derived via normal fitting11                  | End-to-end projective ray solve11      | Sub-degree normal angular error24           | Sets SOTA across 9 zero-shot geometry benchmarks24         | ![](<>); ![](<>) VRAM24      | MIT25                          | **Partial** (Direct proxy mesh & FoV; pitch/roll derived downstream)11         |
| **Depth Pro** (ICLR 2025\)9 | Metric Depth (![](<>)), Focal ![](<>) \[cite: 28, 35\] | Not regressed28                          | Not regressed28                               | Derived from ![](<>) and width34       | ![](<>) on PPR10K focal split33             | SOTA zero-shot boundary F1 score and sharp edge tracing28  | ![](<>); ![](<>) VRAM28      | Apple Sample Code / Research28 | **Partial** (Solves focal length & scale; requires gravity orientation)28      |
| **AnyCalib** (ICCV 2025\)8  | ![](<>), ![](<>), Fisheye8                             | Not regressed8                           | Not regressed8                                | Model-agnostic ray field fit8          | Mean FoV error ![](<>) (MegaDepth pinhole)2 | Superior ray consistency on cropped and wide-angle optics8 | ![](<>); ![](<>) VRAM8       | Apache-2.08                    | **No** (Specialized for complex intrinsics and non-pinhole lenses)8            |
| **UniDAC** (CVPR 2026\)39   | Intrinsic Matrix ![](<>), Metric Depth12               | View-space only12                        | View-space only12                             | Derived directly from ![](<>) matrix12 | ![](<>), ![](<>) on ScanNet++39             | Outperforms Metric3D v2 on universal camera inputs39       | ![](<>); ![](<>) VRAM        | CC BY-NC-SA 4.041              | **Partial** (Solves ![](<>) and metric geometry; requires horizon alignment)12 |
| **Puffin** (ICLR 2026\)19   | Roll, Pitch, Yaw, FoV, Distortion19                    | Competitive with specialized baselines19 | Outperforms baselines on unconstrained sets19 | Consistent with wide-angle splits19    | Direct language-token parameter output19    | SOTA unified camera-centric multimodal spatial reasoning19 | ![](<>); ![](<>) VRAM        | Research Open Source44         | **Yes** (Generates camera rotation & FoV via semantic reasoning)19             |

Across these benchmarks, specialized geometric models demonstrate distinct trade-offs. GeoCalib provides the most accurate and reliable orientation recovery, achieving an ![](<>) AUC at ![](<>) on MegaDepth by enforcing physical constraints on the ![](<>) gravity manifold7.  
However, it does not infer metric depth or scene translation7. Conversely, Depth Pro and MoGe-3 excel at surface reconstruction and metric focal length estimation, but require downstream gravity alignment to match world coordinate axes14.  
These complementary strengths suggest that combining a specialized calibration model with a dense geometry estimator yields the most robust fully automated camera matching pipeline7.

## **Fully Automated Blender Integration Pipeline: The Zero-Click fSpy Architecture**

### **Mathematical Coordinate Transformation**

Integrating neural camera predictions into Blender requires reconciling different coordinate conventions6. Computer vision libraries (OpenCV, COLMAP, GeoCalib) typically adopt a right-handed camera coordinate frame:  
![](<>)  
Blender's camera object convention is also right-handed, but oriented differently:  
![](<>)  
Blender’s world space uses a ![](<>) convention:  
![](<>)  
Given GeoCalib’s predicted gravity unit vector ![](<>) expressed in the vision camera frame, the camera's roll ![](<>) and pitch ![](<>) are derived analytically7:  
![](<>)  
![](<>)  
Assuming the camera's heading (yaw ![](<>)) is aligned with world north (![](<>)), the rotation matrix aligning the camera to the world coordinate frame is:  
![](<>)

### **Automated Ground Plane Estimation for Camera Height**

To match camera height (![](<>)) above the ground plane without manual picking, the input image is processed through Depth Pro or MoGe-3 to generate a dense metric point map ![](<>) in camera view-space14.  
The points are rotated into world orientation using ![](<>):  
![](<>)  
A RANSAC estimator isolates the dominant planar surface in the lower third of the scene point cloud16. The fitted plane equation satisfies:  
![](<>)  
where ![](<>) represents the upright ground normal. The camera's metric elevation above the floor is:  
![](<>)  
This establishes the complete ![](<>) camera transformation matrix ![](<>):  
![](<>)

### **Automated fSpy JSON Exporter**

The following script converts predicted camera parameters into fSpy-compatible JSON format, enabling direct import through the existing fSpy-Blender add-on without opening the standalone fSpy GUI6:

Python  
import json  
import numpy as np

def generate\_fspy\_calibration\_file(  
output\_json\_path: str,  
image\_width: int,  
image\_height: int,  
vfov\_degrees: float,  
roll\_radians: float,  
pitch\_radians: float,  
camera\_height\_meters: float \= 1.65  
) \-\> None:  
"""  
Synthesizes predicted camera parameters into an fSpy-compliant JSON file.  
Matches standard coordinate conventions for direct ingestion via fSpy-Blender.  
"""  
vfov\_rad \= np.deg2rad(vfov\_degrees)  
aspect\_ratio \= image\_width / image\_height  
hfov\_rad \= 2.0 \* np.arctan(np.tan(vfov\_rad / 2.0) \* aspect\_ratio)

    \# Rotation matching Blender camera space:
    \# Camera views down \-Z, with \+Y up and \+X right.
    \# World coordinate frame is Z-up.
    cos\_p, sin\_p \= np.cos(pitch\_radians \+ np.pi / 2.0), np.sin(pitch\_radians \+ np.pi / 2.0)
    cos\_r, sin\_r \= np.cos(roll\_radians), np.sin(roll\_radians)

    \# 3x3 camera-to-world rotation matrix
    R \= np.array(\[
        \[ cos\_r, \-sin\_r \* cos\_p,  sin\_r \* sin\_p\],
        \[ sin\_r,  cos\_r \* cos\_p, \-cos\_r \* sin\_p\],
        \[   0.0,          sin\_p,          cos\_p\]
    \], dtype=np.float64)

    \# Translation placing camera at metric elevation above the origin
    t \= np.array(\[0.0, 0.0, float(camera\_height\_meters)\], dtype=np.float64)

    transform\_matrix \= np.eye(4, dtype=np.float64)
    transform\_matrix\[:3, :3\] \= R
    transform\_matrix\[:3, 3\] \= t

    fspy\_payload \= {
        "imageWidth": int(image\_width),
        "imageHeight": int(image\_height),
        "horizontalFieldOfView": float(hfov\_rad),
        "verticalFieldOfView": float(vfov\_rad),
        "vanishingPoints": \[\],
        "hasVerticalAxis": True,
        "principalPoint": {
            "x": image\_width / 2.0,
            "y": image\_height / 2.0
        },
        "cameraTransform": {
            "rows": transform\_matrix.tolist()
        }
    }

    with open(output\_json\_path, "w", encoding="utf-8") as f:
        json.dump(fspy\_payload, f, indent=4)

### **Headless Blender Automation Script**

For integrated production pipelines, the following script executes in headless Blender (blender \--background \--python setup\_matched\_camera.py), automatically building the camera, configuring viewport projection plates, and setting up ground plane geometry13:

Python  
import bpy  
import mathutils  
import sys

def build\_automated\_blender\_scene(  
image\_filepath: str,  
width\_px: int,  
height\_px: int,  
vfov\_rad: float,  
transform\_matrix\_4x4: list,  
output\_blend\_path: str  
) \-\> None:  
\# Reset scene to a clean default state  
bpy.ops.wm.read\_factory\_settings(use\_empty=True)  
scene \= bpy.context.scene

    \# Align render aspect ratio and dimensions with the input image
    scene.render.resolution\_x \= width\_px
    scene.render.resolution\_y \= height\_px
    scene.render.resolution\_percentage \= 100

    \# Instantiate and configure the matched camera
    cam\_data \= bpy.data.cameras.new(name="AI\_Matched\_Camera")
    cam\_data.sensor\_fit \= 'VERTICAL'
    cam\_data.angle\_y \= vfov\_rad
    cam\_data.show\_background\_images \= True

    \# Mount the input image as a camera background reference plate
    bg\_image \= cam\_data.background\_images.new()
    loaded\_img \= bpy.data.images.load(image\_filepath)
    bg\_image.image \= loaded\_img
    bg\_image.frame\_method \= 'CROP'
    bg\_image.display\_depth \= 'BACK'

    cam\_object \= bpy.data.objects.new(name="AI\_Camera", object\_data=cam\_data)
    scene.collection.objects.link(cam\_object)
    scene.camera \= cam\_object

    \# Set world transform from the solved matrix
    cam\_object.matrix\_world \= mathutils.Matrix(transform\_matrix\_4x4)

    \# Generate a reference ground plane aligned with Z=0
    bpy.ops.mesh.primitive\_plane\_add(size=20.0, location=(0.0, 0.0, 0.0))
    floor\_grid \= bpy.context.active\_object
    floor\_grid.name \= "Reference\_Floor\_Plane"
    floor\_grid.display\_type \= 'WIRE'

    \# Save the configured scene
    bpy.ops.wm.save\_as\_mainfile(filepath=output\_blend\_path)

if \_\_name\_\_ \== "\_\_main\_\_":  
pass

## **Systematic Failure Modes, Edge Cases, and Operational Safeguards**

While deep neural models remove the need for manual line alignment, they introduce specific edge cases that require operational handling:

\[ Input Plate Evaluation \]  
│  
Is principal point centered?  
├── No ───\> Pre-process intrinsics using AnyCalib  
└── Yes ──\> Proceed to GeoCalib / MoGe-3  
│  
Are reflective/specular surfaces dominant?  
├── Yes ──\> Rely on GeoCalib perspective fields; disable dense depth  
└── No ───\> Run joint GeoCalib \+ Depth Pro / MoGe-3  
│  
Does scene contain known physical scale references?  
├── Yes ──\> Scale camera tz against reference object  
└── No ───\> Use default ground-plane height prior (1.65 m)

### **Optical Center Decentering and Cropped Plates**

Most single-view calibration architectures (including GeoCalib and Depth Pro) assume that the optical center (principal point) is located at the center of the image canvas (![](<>))7. When an image has undergone off-center cropping, digital panning, or asymmetric lens shifting, this assumption is violated8.  
A decentered principal point creates an asymmetric perspective field, which models often misinterpret as camera pitch or roll8. In pipelines dealing with heavily cropped plates, AnyCalib or UniDepth should be used as an initial calibration stage, as their ray-direction formulations can solve for optical center offsets (![](<>)) directly8.

### **Scale Ambiguity and Ground Plane Identification**

Monocular depth models estimate metric scale by learning typical physical sizes for familiar objects (such as vehicles, furniture, and people)31. In environments lacking standard semantic scale cues—such as aerial drone photography, extreme close-up macro shots, or abstract architectural scenes—the predicted metric scale can drift35.  
While the camera’s angular parameters (pitch, roll, and FoV) remain stable, the absolute camera height (![](<>)) may scale incorrectly35.  
When processing imagery without recognizable scale anchors, pipelines should incorporate an interactive scale-adjustment factor or default to a standard standing eye-level prior (![](<>))35.

### **Specular, Reflective, and Translucent Surfaces**

Dense depth estimation models (Depth Pro, MoGe-3, and UniDepth) compute depth by analyzing surface textures and occluding boundaries24. Environments with extensive reflective or translucent surfaces—such as wet streets, large bodies of water, mirrors, or glass facades—violate standard Lambertian assumptions9.  
This can lead to noisy depth values and flying-pixel artifacts near surface transitions9.  
In scenes dominated by reflections or transparency, camera matching should rely primarily on GeoCalib's global perspective field, which infers orientation from overall scene structure rather than per-pixel surface depths16.

### **Radial and Non-Rectilinear Lens Distortions**

Standard pinhole projection models struggle when applied to images with noticeable barrel or pincushion distortion2. If an image with barrel distortion is fed directly to a rectilinear solver, straight lines appear curved, leading to inaccurate FoV and pitch estimates7.  
To handle distorted imagery, plates should first be processed through GeoCalib (using its \--camera\_model simple\_radial mode) or AnyCalib (using kb:4 or division models) to estimate distortion coefficients (![](<>))7.  
The plate can then be digitally undistorted before camera matching, or mapped directly to Blender Cycles' native Panoramic Fisheye camera model7.

## **Practical Synthesis and Implementation Strategies**

For visual effects artists, architectural visualizers, and general 3D practitioners seeking an automated alternative to fSpy:

- **Direct Drop-In Replacement for fSpy:** GeoCalib provides the most direct functional equivalent to fSpy7. It infers camera roll, pitch, focal length, and radial distortion across indoor, outdoor, natural, and architectural scenes without manual vanishing line input7. Its fast inference speed (![](<>)), low VRAM footprint (![](<>)), and Apache-2.0 license make it well-suited for interactive tools and add-on development7.
- **Joint Scene Reconstruction and Camera Matching:** Microsoft's MoGe-3 is the preferred solution when both camera matching and proxy geometry generation are required14. It recovers fine geometric details, surface normal maps, and camera FoV within a single forward pass, exporting textured .glb assets that load directly into Blender14.
- **High-Resolution Plates and Detail Preservation:** Apple Depth Pro provides the highest edge definition and boundary sharpness for high-resolution images, generating ![](<>) metric depth maps and horizontal focal lengths in ![](<>)28. It is particularly effective for portraiture, product rendering, and scenes with fine silhouette detail28.
- **The Composite Production Pipeline:** The most robust automated solution pairs **GeoCalib** with **MoGe-3** or **Depth Pro**7. In this configuration, GeoCalib determines camera orientation (pitch and roll) and field of view, while the dense depth model reconstructs the visible environment and establishes camera height (![](<>)) via automated ground-plane fitting7. This two-stage pipeline eliminates manual vanishing-point placement while delivering a fully calibrated camera and matching proxy scene in Blender13.

#### **Works cited**

> 1. GeoCalib: Learning Single-image Calibration with Geometric, [https://hub.baai.ac.cn/paper/4e5c7bc1-d5fe-41db-b2ec-d9cec2e2cd5e](https://hub.baai.ac.cn/paper/4e5c7bc1-d5fe-41db-b2ec-d9cec2e2cd5e)
> 2. CalibAnyView: Beyond Single-View Camera Calibration in the Wild, [https://arxiv.org/html/2605.14615v1](https://arxiv.org/html/2605.14615v1)
> 3. blender-addon · GitHub Topics, [https://github.com/topics/blender-addon](https://github.com/topics/blender-addon)
> 4. Resources \- Arts 'n Science, [https://artsnscience.eu/resources/](https://artsnscience.eu/resources/)
> 5. Deep Learning for Camera Calibration and Beyond: A Survey \- arXiv, [https://arxiv.org/html/2303.10559v3](https://arxiv.org/html/2303.10559v3)
> 6. \[Import fSpy JSON to Nuke\] load a json file with camera ... \- GitHub Gist, [https://gist.github.com/MitchellKehn/e7ccdfad932886c7e9e4c072a08006c8](https://gist.github.com/MitchellKehn/e7ccdfad932886c7e9e4c072a08006c8)
> 7. GeoCalib: Single-image Calibration with Geometric Optimization, [https://github.com/cvg/GeoCalib](https://github.com/cvg/GeoCalib)
> 8. AnyCalib: On-Manifold Learning for Model-Agnostic Single ... \- GitHub, [https://github.com/javrtg/AnyCalib](https://github.com/javrtg/AnyCalib)
> 9. DEPTH PRO: SHARP MONOCULAR METRIC DEPTH IN ... \- HyperAI, [https://hyper.ai/en/papers/iclr\_\_2025\_\_aueXfY0Clv](https://hyper.ai/en/papers/iclr__2025__aueXfY0Clv)
> 10. Minimal Solvers for Single-View Lens-Distorted Camera Auto, [https://www.researchgate.net/publication/352392703\_Minimal\_Solvers\_for\_Single-View\_Lens-Distorted\_Camera\_Auto-Calibration](https://www.researchgate.net/publication/352392703_Minimal_Solvers_for_Single-View_Lens-Distorted_Camera_Auto-Calibration)
> 11. MoGe: Unlocking Accurate Monocular Geometry Estimation for, [https://www.openaccess.thecvf.com/content/CVPR2025/papers/Wang\_MoGe\_Unlocking\_Accurate\_Monocular\_Geometry\_Estimation\_for\_Open-Domain\_Images\_with\_CVPR\_2025\_paper.pdf](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Wang_MoGe_Unlocking_Accurate_Monocular_Geometry_Estimation_for_Open-Domain_Images_with_CVPR_2025_paper.pdf)
> 12. bdck/unidepth-inference \- Hugging Face, [https://huggingface.co/bdck/unidepth-inference](https://huggingface.co/bdck/unidepth-inference)
> 13. atlas-camera \- ComfyUI Cloud, [https://comfy.icu/extension/mikejamesvfx\_\_atlas-camera](https://comfy.icu/extension/mikejamesvfx__atlas-camera)
> 14. MoGe: Accurate Monocular Geometry Estimation \- GitHub, [https://github.com/microsoft/moge](https://github.com/microsoft/moge)
> 15. Paper page \- GeoCalib: Learning Single-image Calibration with, [https://huggingface.co/papers/2409.06704](https://huggingface.co/papers/2409.06704)
> 16. Learning Single-image Calibration with Geometric Optimization, [https://www.ecva.net/papers/eccv\_2024/papers\_ECCV/papers/05636-supp.pdf](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05636-supp.pdf)
> 17. GeoCalib: Learning Single-image Calibration with Geometric ... \- arXiv, [https://arxiv.org/html/2409.06704v2](https://arxiv.org/html/2409.06704v2)
> 18. Burgstall-labs/ComfyUI-VR-Outpaint-Tools \- GitHub, [https://github.com/Burgstall-labs/ComfyUI-VR-Outpaint-Tools](https://github.com/Burgstall-labs/ComfyUI-VR-Outpaint-Tools)
> 19. THINKING WITH CAMERA:AUNIFIED MULTIMODAL MODEL FOR, [https://proceedings.iclr.cc/paper\_files/paper/2026/file/e9882f7f7c44a10acc01132302bac9d8-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2026/file/e9882f7f7c44a10acc01132302bac9d8-Paper-Conference.pdf)
> 20. Min Float \- ComfyUI Cloud \- Comfy.ICU, [https://comfy.icu/node/SL\_MinFloat](https://comfy.icu/node/SL_MinFloat)
> 21. GeoCalib → fSpy JSON \- ComfyUI Cloud, [https://comfy.icu/node/GeoCalibToFSpy](https://comfy.icu/node/GeoCalibToFSpy)
> 22. Atlas Export Relief Mesh (OBJ) \- ComfyUI Cloud, [https://comfy.icu/node/AtlasExportReliefMesh](https://comfy.icu/node/AtlasExportReliefMesh)
> 23. MoGe \- Ruicheng Wang, [https://wangrc.site/MoGePage/](https://wangrc.site/MoGePage/)
> 24. MoGe-3 \- Ruicheng Li, [https://qft-333.github.io/moge3page/](https://qft-333.github.io/moge3page/)
> 25. pyproject.toml \- microsoft/MoGe \- GitHub, [https://github.com/microsoft/MoGe/blob/main/pyproject.toml](https://github.com/microsoft/MoGe/blob/main/pyproject.toml)
> 26. MoGe: Accurate Monocular Geometry Estimation \- ModelScope, [https://www.modelscope.cn/models/Comfy-Org/MoGe](https://www.modelscope.cn/models/Comfy-Org/MoGe)
> 27. MoGe: Revolutionizing 3D Geometry Estimation from Single Images, [https://dev.to/githubopensource/moge-revolutionizing-3d-geometry-estimation-from-single-images-38bf](https://dev.to/githubopensource/moge-revolutionizing-3d-geometry-estimation-from-single-images-38bf)
> 28. Depth Pro: Sharp Monocular Metric Depth in Less Than a ... \- GitHub, [https://github.com/apple-aiml-research/ml-depth-pro](https://github.com/apple-aiml-research/ml-depth-pro)
> 29. Depth Pro: Sharp Monocular Metric Depth in Less Than a Second, [https://huggingface.co/papers/2410.02073](https://huggingface.co/papers/2410.02073)
> 30. DEPTH PRO: ShARP Monocular METric DEPTH IN LESS THAN A, [https://medium.com/@jiangmen28/apple-depth-pro-sharp-monocular-metric-depth-in-less-than-a-second-bd020a4c3ae7](https://medium.com/@jiangmen28/apple-depth-pro-sharp-monocular-metric-depth-in-less-than-a-second-bd020a4c3ae7)
> 31. Estimating Depth and Focal Length with Apple Depth-Pro \- Medium, [https://medium.com/@Nivitus./estimating-depth-and-focal-length-with-apple-depth-pro-e49a19392b47](https://medium.com/@Nivitus./estimating-depth-and-focal-length-with-apple-depth-pro-e49a19392b47)
> 32. Depth Pro: Sharp Monocular Metric Depth in Less Than a Second, [https://arxiv.org/html/2410.02073v1](https://arxiv.org/html/2410.02073v1)
> 33. Depth Pro: Sharp Monocular Metric Depth in Less Than a Second, [https://andlukyane.com/blog/paper-review-depthpro](https://andlukyane.com/blog/paper-review-depthpro)
> 34. Depth Pro: Sharp Monocular Metric Depth in Less Than a Second, [https://arxiv.org/html/2410.02073v2](https://arxiv.org/html/2410.02073v2)
> 35. Apple Depth Pro: sharp metric depth from one image \- AI IDE List, [https://aiidelist.com/blog/depth-pro](https://aiidelist.com/blog/depth-pro)
> 36. Best Depth Estimation Models: Depth Anything V2 & More, [https://blog.roboflow.com/depth-estimation-models/](https://blog.roboflow.com/depth-estimation-models/)
> 37. ComfyUI-Depth-Pro \- GitHub, [https://github.com/spacepxl/ComfyUI-Depth-Pro](https://github.com/spacepxl/ComfyUI-Depth-Pro)
> 38. AnyCalib \- ICCV 2025 Open Access Repository, [https://openaccess.thecvf.com/content/ICCV2025/html/Tirado-Garin\_AnyCalib\_On-Manifold\_Learning\_for\_Model-Agnostic\_Single-View\_Camera\_Calibration\_ICCV\_2025\_paper.html](https://openaccess.thecvf.com/content/ICCV2025/html/Tirado-Garin_AnyCalib_On-Manifold_Learning_for_Model-Agnostic_Single-View_Camera_Calibration_ICCV_2025_paper.html)
> 39. girish1511/UniDAC \- GitHub, [https://github.com/girish1511/UniDAC](https://github.com/girish1511/UniDAC)
> 40. lpiccinelli-eth/UniDepth: Universal Monocular Metric Depth Estimation, [https://github.com/lpiccinelli-eth/unidepth](https://github.com/lpiccinelli-eth/unidepth)
> 41. UniDepth: Universal Monocular Metric Depth Estimation \- arXiv, [https://arxiv.org/html/2403.18913v1](https://arxiv.org/html/2403.18913v1)
> 42. UniDepthV2: Universal Monocular Metric Depth Estimation ... \- arXiv, [https://arxiv.org/html/2502.20110v1](https://arxiv.org/html/2502.20110v1)
> 43. Estimate multiple camera poses and reconstruct the dynamic scene, [https://github.com/ljjTYJR/multiple-view-dynamic-reconstruction](https://github.com/ljjTYJR/multiple-view-dynamic-reconstruction)
> 44. Puffin Series: Towards Unified Multimodal 3D World Models \- GitHub, [https://github.com/KangLiao929/Puffin](https://github.com/KangLiao929/Puffin)
> 45. A Unified Multimodal Model for Camera-Centric Understanding and, [https://arxiv.org/html/2510.08673v1](https://arxiv.org/html/2510.08673v1)
> 46. Puffin Thinking with Camera: A Unified Multimodal Model for, [https://kangliao929.github.io/projects/puffin/](https://kangliao929.github.io/projects/puffin/)
> 47. A Unified Multimodal Model for Camera-Centric Understanding and, [https://arxiv.org/html/2510.08673v2](https://arxiv.org/html/2510.08673v2)
> 48. Yikai Wang | alphaXiv, [https://www.alphaxiv.org/@yikai-wang-3](https://www.alphaxiv.org/@yikai-wang-3)
> 49. Robust Monocular Depth Estimation Under Crop-Resize-Induced, [https://www.mdpi.com/2079-9292/15/10/2180](https://www.mdpi.com/2079-9292/15/10/2180)
> 50. Extract matrix from jSon file \- Stack Overflow, [https://stackoverflow.com/questions/58361808/extract-matrix-from-json-file](https://stackoverflow.com/questions/58361808/extract-matrix-from-json-file)
> 51. ORBIT: Benchmarking SfM in the Wild with 360° Video \- OpenReview, [https://openreview.net/pdf?id=R48lWX6QMt](https://openreview.net/pdf?id=R48lWX6QMt)

[image1]: data:image/png;base64,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
[image2]: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAaCAYAAABctMd+AAABDklEQVR4XmNgGAWjYBRQDFiBWByIZYDYDYiloeKMQCwJxWZAbA4VJwkYA/FXIP4Pxb5QcR4kMRBeCBUnCYBcrsqAaTjI5XVI4mQZDgPohoMAsq9GDYcDFyD+BxWHGQ5LXaBUxAFlCzNA4ggngBkehCSWgyS+lQFiGMgSmNhVID4ExH+B+A4QG0C0YQKYC3sZIK4D4Y9QMRB+BsQZULUwC3qg/HAGiP7rDJD8ghOA0jfIqyDDmYFYjAHiYmQAM7wcygfpOQAVK4KKkQ3QDQc55i4DxPWgeKIIwAyfD+XHQPl7gJgfpohcADMc5FpQXIAidBUQSyArIhegBwvVQB8Q/2KAGP4FTY5iwIIuMCAAAO0xUbrMRUwsAAAAAElFTkSuQmCC
