# Third-party notices and credits

AI Camera Match integrates existing research and open-source tools. It does not copy their source code into this repository; the solver environment downloads them from their official sources at the pinned commits listed below. Their licences apply to those downloads.

## Models and libraries installed into the solver environment

| Project                                                        | Used for                                                                                         | Source (pinned)                                                                                 | Code licence                  | Weights licence                                                        |
| :------------------------------------------------------------- | :----------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------- | :---------------------------- | :--------------------------------------------------------------------- |
| GeoCalib                                                       | Focal length, lens distortion, gravity (roll and pitch), plate undistortion                      | [cvg/GeoCalib](https://github.com/cvg/GeoCalib) @ `97b8968`                                     | Apache-2.0                    | CC-BY 4.0                                                              |
| MoGe (MoGe-2, MoGe-3)                                          | Metric point map, normals and the proxy mesh; the mesh export follows MoGe's `moge infer` script | [microsoft/MoGe](https://github.com/microsoft/MoGe) @ `74fbce0`                                 | MIT                           | MIT ([model card](https://huggingface.co/Ruicheng/moge-2-vitl-normal)) |
| utils3d-moge                                                   | Mesh construction from point maps (MoGe dependency)                                              | [EasternJournalist/utils3d-moge](https://github.com/EasternJournalist/utils3d-moge) @ `62f09d5` | MIT                           | n/a                                                                    |
| PyTorch, torchvision                                           | Model runtime                                                                                    | [pytorch.org](https://pytorch.org)                                                              | BSD-3-Clause                  | n/a                                                                    |
| OpenCV, NumPy, SciPy, Pillow, trimesh, Kornia, huggingface_hub | Image I/O, maths, mesh export, model download                                                    | PyPI                                                                                            | Apache-2.0 / BSD / MIT / HPND | n/a                                                                    |

Please cite the papers when you publish work made with this add-on:

- A. Veicht, P.-E. Sarlin, P. Lindenberger, M. Pollefeys. _GeoCalib: Learning Single-image Calibration with Geometric Optimization._ ECCV 2024. [arXiv:2409.06704](https://arxiv.org/abs/2409.06704)
- R. Wang et al. _MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision._ CVPR 2025. [arXiv:2410.19115](https://arxiv.org/abs/2410.19115)
- R. Wang et al. _MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details._ [arXiv:2507.02546](https://arxiv.org/abs/2507.02546)
- _MoGe-3: Fine-Detail Monocular Geometry Estimation with Self-Guided Sparse Volumetric Refinement._ [arXiv:2607.17967](https://arxiv.org/abs/2607.17967)

## Formats and workflow

- **fSpy** by Per Gantelius ([stuffmatic/fSpy](https://github.com/stuffmatic/fSpy), GPL-3.0). The `.fspy` writer in `worker/aicm_worker/fspy.py` implements fSpy's documented project file format (`project_file_format.md`).
- **fSpy-Blender** by Per Gantelius ([stuffmatic/fSpy-Blender](https://github.com/stuffmatic/fSpy-Blender), GPL-3.0). The camera, plate and render-resolution setup follows its importer, and the `.fspy` files written here are laid out to be read by it.

## Prior art

- **Atlas Camera** ([mikejamesvfx/atlas-camera](https://github.com/mikejamesvfx/atlas-camera), MIT) combines GeoCalib, MoGe-2 and depth models for photo-to-camera solving with an experimental Blender bridge. No code is taken from it.
