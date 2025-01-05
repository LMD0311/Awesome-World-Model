# Awesome World Models for Autonomous Driving [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

Collect some World Models (for Autonomous Driving) papers. 

If you find some ignored papers, **feel free to [*create pull requests*](https://github.com/LMD0311/Awesome-World-Model/blob/main/ContributionGuidelines.md), [*open issues*](https://github.com/LMD0311/Awesome-World-Model/issues/new), or [*email* me](mailto:xzhou03@hust.edu.cn) / [*Qi Wang*](mailto:qiwang067@163.com)**. Contributions in any form to make this list more comprehensive are welcome. 📣📣📣

If you find this repository useful, please consider  **giving us a star** 🌟. 

Feel free to share this list with others! 🥳🥳🥳

## Workshop & Challenge

- [`CVPR 2024 Workshop & Challenge | OpenDriveLab`](https://opendrivelab.com/challenge2024/#predictive_world_model) Track #4: Predictive World Model.
  > Serving as an abstract spatio-temporal representation of reality, the world model can predict future states based on the current state. The learning process of world models has the potential to elevate a pre-trained foundation model to the next level. Given vision-only inputs, the neural network outputs point clouds in the future to testify its predictive capability of the world.
  
- [`CVPR 2023 Workshop on Autonomous Driving`](https://cvpr2023.wad.vision/) CHALLENGE 3: ARGOVERSE CHALLENGES, [3D Occupancy Forecasting](https://eval.ai/web/challenges/challenge-page/1977/overview) using the [Argoverse 2 Sensor Dataset](https://www.argoverse.org/av2.html#sensor-link). Predict the spacetime occupancy of the world for the next 3 seconds.

## Papers

### World model original paper

- Using Occupancy Grids for Mobile Robot Perception and Navigation [[paper](http://www.sci.brooklyn.cuny.edu/~parsons/courses/3415-fall-2011/papers/elfes.pdf)]

### Technical blog or video

- **`Yann LeCun`**: A Path Towards Autonomous Machine Intelligence [[paper](https://openreview.net/pdf?id=BZ5a1r-kVsf)] [[Video](https://www.youtube.com/watch?v=OKkEdTchsiE)]
- **`CVPR'23 WAD`** Keynote - Ashok Elluswamy, Tesla [[Video](https://www.youtube.com/watch?v=6x-Xb_uT7ts)]
- **`Wayve`** Introducing GAIA-1: A Cutting-Edge Generative AI Model for Autonomy [[blog](https://wayve.ai/thinking/introducing-gaia1/)] 
  > World models are the basis for the ability to predict what might happen next, which is fundamentally important for autonomous driving. They can act as a learned simulator, or a mental “what if” thought experiment for model-based reinforcement learning (RL) or planning. By incorporating world models into our driving models, we can enable them to understand human decisions better and ultimately generalise to more real-world situations.
  

### Survey

- A survey on multimodal large language models for autonomous driving. **`WACVW 2024`** [[Paper](https://arxiv.org/abs/2311.12320)] [[Code](https://github.com/IrohXu/Awesome-Multimodal-LLM-Autonomous-Driving)]
- World Models: The Safety Perspective. **`ISSREW`** [[Paper](https://arxiv.org/abs/2411.07690)
- Understanding World or Predicting Future? A Comprehensive Survey of World Models. **`arXiv 2024.11`** [[Paper](https://arxiv.org/abs/2411.14499)]
- Exploring the Interplay Between Video Generation and World Models in Autonomous Driving: A Survey. **`arXiv 2024.11`** [[Paper](https://arxiv.org/abs/2411.02914)]
- Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI. **`arXiv 2024.7`** [[Paper](https://arxiv.org/abs/2407.06886)] [[Code](https://github.com/HCPLab-SYSU/Embodied_AI_Paper_List)]
- Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond. **`arXiv 2024.5`** [[Paper](https://arxiv.org/abs/2405.03520)] [[Code](https://github.com/GigaAI-research/General-World-Models-Survey)]
- World Models for Autonomous Driving: An Initial Survey. **`2024.3, arxiv`** [[Paper](https://arxiv.org/abs/2403.02622)]

### 2024

- [**SEM2**] Enhance Sample Efficiency and Robustness of End-to-end Urban Autonomous Driving via Semantic Masked World Model. **`TITS`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10538211/)]
- **Vista**: A Generalizable Driving World Model with High Fidelity and Versatile Controllability. **`NeurIPS 2024`** [[Paper](https://arxiv.org/abs/2405.17398)] [[Code](https://github.com/OpenDriveLab/Vista)]
- **DrivingDojo Dataset**: Advancing Interactive and Knowledge-Enriched Driving World Model. **`NeurIPS 2024`** [[Paper](https://arxiv.org/abs/2410.10738)] [[Project](https://drivingdojo.github.io/)]
- **Think2Drive**: Efficient Reinforcement Learning by Thinking in Latent World Model for Quasi-Realistic Autonomous Driving. **`ECCV 2024`** [[Paper](https://arxiv.org/abs/2402.16720)]
- [**MARL-CCE**] Modelling Competitive Behaviors in Autonomous Driving Under Generative World Model. **`ECCV 2024`** [[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05085.pdf)] [[Code](https://github.com/qiaoguanren/MARL-CCE)]
- **DriveDreamer**: Towards Real-world-driven World Models for Autonomous Driving. **`ECCV 2024`** [[Paper](https://arxiv.org/abs/2309.09777)] [[Code](https://github.com/JeffWang987/DriveDreamer)]
- **GenAD**: Generative End-to-End Autonomous Driving. **`ECCV 2024`** [[Paper](https://arxiv.org/abs/2402.11502)] [[Code](https://github.com/wzzheng/GenAD)]
- **OccWorld**: Learning a 3D Occupancy World Model for Autonomous Driving. **`ECCV 2024`** [[Paper](https://arxiv.org/abs/2311.16038)] [[Code](https://github.com/wzzheng/OccWorld)]
- [**NeMo**] Neural Volumetric World Models for Autonomous Driving. **`ECCV 2024`** [[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02571.pdf)]
- **CarFormer**: Self-Driving with Learned Object-Centric Representations. **`ECCV 2024`** [[Paper](https://arxiv.org/abs/2407.15843)] [[Code](https://kuis-ai.github.io/CarFormer/)]
- [**MARL-CCE**] Modelling-Competitive-Behaviors-in-Autonomous-Driving-Under-Generative-World-Model. **`ECCV 2024`** [[Code](https://github.com/qiaoguanren/MARL-CCE)]
- [**GUMP**] Solving Motion Planning Tasks with a Scalable Generative Model. **`ECCV 2024`** [[Paper](https://arxiv.org/abs/2407.02797)] [[Code](https://github.com/HorizonRobotics/GUMP/)]
- **DrivingDiffusion**: Layout-Guided multi-view driving scene video generation with latent diffusion model. **`ECCV 2024`** [[Paper](https://arxiv.org/abs/2310.07771)] [[Code](https://github.com/shalfun/DrivingDiffusion)]
- **3D-VLA**: A 3D Vision-Language-Action Generative World Model.  **`ICML 2024`** [[Paper](https://arxiv.org/abs/2403.09631)]
- [**ViDAR**] Visual Point Cloud Forecasting enables Scalable Autonomous Driving. **`CVPR 2024`** [[Paper](https://arxiv.org/abs/2312.17655)] [[Code](https://github.com/OpenDriveLab/ViDAR)]
- [**GenAD**] Generalized Predictive Model for Autonomous Driving. **`CVPR 2024`** [[Paper](https://arxiv.org/abs/2403.09630)] [[Data](https://github.com/OpenDriveLab/DriveAGI?tab=readme-ov-file#genad-dataset-opendv-youtube)]
- **Cam4DOCC**: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications. **`CVPR 2024`** [[Paper](https://arxiv.org/abs/2311.17663)] [[Code](https://github.com/haomo-ai/Cam4DOcc)]
- [**Drive-WM**] Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving. **`CVPR 2024`** [[Paper](https://arxiv.org/abs/2311.17918)] [[Code](https://github.com/BraveGroup/Drive-WM)]
- **DriveWorld**: 4D Pre-trained Scene Understanding via World Models for Autonomous Driving. **`CVPR 2024`** [[Paper](https://arxiv.org/abs/2405.04390)]
- **Panacea**: Panoramic and Controllable Video Generation for Autonomous Driving. **`CVPR 2024`** [[Paper](https://arxiv.org/abs/2311.16813)] [[Code](https://panacea-ad.github.io/)]
- **UnO**: Unsupervised Occupancy Fields for Perception and Forecasting. **`CVPR 2024`** [[Paper](https://arxiv.org/abs/2406.08691)] [[Code](https://waabi.ai/research/uno)]
- **MagicDrive**: Street View Generation with Diverse 3D Geometry Control. **`ICLR 2024`** [[Paper](https://arxiv.org/abs/2310.02601)] [[Code](https://github.com/cure-lab/MagicDrive)]
- **Copilot4D**: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion. **`ICLR 2024`** [[Paper](https://arxiv.org/abs/2311.01017)]
- **SafeDreamer**: Safe Reinforcement Learning with World Models. **`ICLR 2024`** [[Paper](https://openreview.net/forum?id=tsE5HLYtYg)] [[Code](https://github.com/PKU-Alignment/SafeDreamer)]
- **DrivingWorld**: Constructing World Model for Autonomous Driving via Video GPT. **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.19505)] [[Code](https://github.com/YvanYin/DrivingWorld)]
- **DrivingGPT**: Unifying Driving World Modeling and Planning with Multi-modal Autoregressive Transformers. **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.18607)] [[Project](https://rogerchern.github.io/DrivingGPT/)]
- An Efficient Occupancy World Model via Decoupled Dynamic Flow and Image-assisted Training. **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.13772)]
- **GEM**: A Generalizable Ego-Vision Multimodal World Model for Fine-Grained Ego-Motion, Object Dynamics, and Scene Composition Control. **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.11198)] [[Project](https://vita-epfl.github.io/GEM.github.io/)]
- **GaussianWorld**: Gaussian World Model for Streaming 3D Occupancy Prediction. **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.10373)] [[Code](https://github.com/zuosc19/GaussianWorld)]
- **Doe-1**: Closed-Loop Autonomous Driving with Large World Model. **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.09627)] [[Code](https://github.com/wzzheng/Doe)]
- [**DrivePhysica**] Physical Informed Driving World Model. **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.08410)] [[Code](https://metadrivescape.github.io/papers_project/DrivePhysica/page.html)]
- **HoloDrive**: Holistic 2D-3D Multi-Modal Street Scene Generation for Autonomous Driving. **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.01407)]
- **InfinityDrive**: Breaking Time Limits in Driving World Models. **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.01522)] [[Project Page](https://metadrivescape.github.io/papers_project/InfinityDrive/page.html)]
- **ReconDreamer**: Crafting World Models for Driving Scene Reconstruction via Online Restoration. **`arXiv 2024.11`** [[Paper](https://arxiv.org/abs/2411.19548)] [[Code](https://github.com/GigaAI-research/ReconDreamer)]
- Generating Out-Of-Distribution Scenarios Using Language Models. **`arXiv 2024.11`** [[Paper](https://arxiv.org/abs/2411.16554)]
- **Imagine-2-Drive**: High-Fidelity World Modeling in CARLA for Autonomous Vehicles. **`arXiv 2024.11`** [[Paper](https://arxiv.org/abs/2411.10171)] [[Project Page](https://anantagrg.github.io/Imagine-2-Drive.github.io/)]
- **WorldSimBench**: Towards Video Generation Models as World Simulator. **`arXiv 2024.10`** [[Paper](https://arxiv.org/abs/2410.18072)] [[Project Page](https://iranqin.github.io/WorldSimBench.github.io/)]
- **DriveDreamer4D**: World Models Are Effective Data Machines for 4D Driving Scene Representation. **`arXiv 2024.10`** [[Paper](https://arxiv.org/abs/2410.13571)] [[Project Page](https://drivedreamer4d.github.io/)]
- **DOME**: Taming Diffusion Model into High-Fidelity Controllable Occupancy World Model. **`arXiv 2024.10`** [[Paper](https://arxiv.org/abs/2410.10429)] [[Project Page](https://gusongen.github.io/DOME)]
- [**SSR**] Does End-to-End Autonomous Driving Really Need Perception Tasks? **`arXiv 2024.9`** [[Paper](https://arxiv.org/abs/2409.18341)] [[Code](https://github.com/PeidongLi/SSR)]
- Mitigating Covariate Shift in Imitation Learning for Autonomous Vehicles Using Latent Space Generative World Models. **`arXiv 2024.9`** [[Paper](https://arxiv.org/abs/2409.16663)]
- [**LatentDriver**] Learning Multiple Probabilistic Decisions from Latent World Model in Autonomous Driving. **`arXiv 2024.9`** [[Paper](https://arxiv.org/abs/2409.15730)] [[Code](https://github.com/Sephirex-X/LatentDriver)]
- **RenderWorld**: World Model with Self-Supervised 3D Label. **`arXiv 2024.9`** [[Paper](https://arxiv.org/abs/2409.11356)]
- **OccLLaMA**: An Occupancy-Language-Action Generative World Model for Autonomous Driving. **`arXiv 2024.9`** [[Paper](https://arxiv.org/abs/2409.03272)]
- **DriveGenVLM**: Real-world Video Generation for Vision Language Model based Autonomous Driving. **`arXiv 2024.8`** [[Paper](https://arxiv.org/abs/2408.16647)]
- [**Drive-OccWorld**] Driving in the Occupancy World: Vision-Centric 4D Occupancy Forecasting and Planning via World Models for Autonomous Driving. **`arXiv 2024.8`** [[Paper](https://arxiv.org/abs/2408.14197)]
- **BEVWorld**: A Multimodal World Model for Autonomous Driving via Unified BEV Latent Space. **`arXiv 2024.7`** [[Paper](https://arxiv.org/abs/2407.05679)] [[Code](https://github.com/zympsyche/BevWorld)]
- [**TOKEN**] Tokenize the World into Object-level Knowledge to Address Long-tail Events in Autonomous Driving. **`arXiv 2024.7`** [[Paper](https://arxiv.org/abs/2407.00959)]
- **UMAD**: Unsupervised Mask-Level Anomaly Detection for Autonomous Driving. **`arXiv 2024.6`** [[Paper](https://arxiv.org/abs/2406.06370)]
- **SimGen**: Simulator-conditioned Driving Scene Generation. **`arXiv 2024.6`** [[Paper](https://arxiv.org/abs/2406.09386)] [[Code](https://metadriverse.github.io/simgen/)]
- [**AdaptiveDriver**] Planning with Adaptive World Models for Autonomous Driving. **`arXiv 2024.6`** [[Paper](https://arxiv.org/abs/2406.10714)] [[Code](https://arunbalajeev.github.io/world_models_planning/world_model_paper.html)]
- [**LAW**] Enhancing End-to-End Autonomous Driving with Latent World Model. **`arXiv 2024.6`** [[Paper](https://arxiv.org/abs/2406.08481)] [[Code](https://github.com/BraveGroup/LAW)]
- [**Delphi**] Unleashing Generalization of End-to-End Autonomous Driving with Controllable Long Video Generation. **`arXiv 2024.6`** [[Paper](https://arxiv.org/abs/2406.01349)] [[Code](https://github.com/westlake-autolab/Delphi)]
- **OccSora**: 4D Occupancy Generation Models as World Simulators for Autonomous Driving. **`arXiv 2024.5`** [[Paper](https://arxiv.org/abs/2405.20337)] [[Code](https://github.com/wzzheng/OccSora)]
- **MagicDrive3D**: Controllable 3D Generation for Any-View Rendering in Street Scenes. **`arXiv 2024.5`** [[Paper](https://arxiv.org/abs/2405.14475)] [[Code](https://gaoruiyuan.com/magicdrive3d/)]
- **CarDreamer**: Open-Source Learning Platform for World Model based Autonomous Driving. **`arXiv 2024.5`** [[Paper](https://arxiv.org/abs/2405.09111)] [[Code](https://github.com/ucd-dare/CarDreamer)]
- [**DriveSim**] Probing Multimodal LLMs as World Models for Driving. **`arXiv 2024.5`** [[Paper](https://arxiv.org/abs/2405.05956)] [[Code](https://github.com/sreeramsa/DriveSim)]
- **LidarDM**: Generative LiDAR Simulation in a Generated World. **`arXiv 2024.4`** [[Paper](https://arxiv.org/abs/2404.02903)] [[Code](https://github.com/vzyrianov/lidardm)]
- **SubjectDrive**: Scaling Generative Data in Autonomous Driving via Subject Control. **`arXiv 2024.3`** [[Paper](https://arxiv.org/abs/2403.19438)] [[Project](https://subjectdrive.github.io/)]
- **DriveDreamer-2**: LLM-Enhanced World Models for Diverse Driving Video Generation. **`arXiv 2024.3`** [[Paper](https://arxiv.org/abs/2403.06845)] [[Code](https://drivedreamer2.github.io/)]

### 2023

- **TrafficBots**: Towards World Models for Autonomous Driving Simulation and Motion Prediction. **`ICRA 2023`** [[Paper](https://arxiv.org/abs/2303.04116)] [[Code](https://github.com/zhejz/TrafficBots)]
- **WoVoGen**: World Volume-aware Diffusion for Controllable Multi-camera Driving Scene Generation. **`arXiv 2023.12`** [[Paper](https://arxiv.org/abs/2312.02934)] [[Code](https://github.com/fudan-zvg/WoVoGen)]
- [**CTT**] Categorical Traffic Transformer: Interpretable and Diverse Behavior Prediction with Tokenized Latent. **`arXiv 2023.11`** [[Paper](https://arxiv.org/abs/2311.18307)]
- **MUVO**: A Multimodal Generative World Model for Autonomous Driving with Geometric Representations. **`arXiv 2023.11`** [[Paper](https://arxiv.org/abs/2311.11762)]
- **GAIA-1**: A Generative World Model for Autonomous Driving. **`arXiv 2023.9`** [[Paper](https://arxiv.org/abs/2309.17080)]
- **ADriver-I**: A General World Model for Autonomous Driving. **`arXiv 2023.9`** [[Paper](https://arxiv.org/abs/2311.13549)]
- **UniWorld**: Autonomous Driving Pre-training via World Models. **`arXiv 2023.8`** [[Paper](https://arxiv.org/abs/2308.07234)] [[Code](https://github.com/chaytonmin/UniWorld)]

### 2022

- [**MILE**] Model-Based Imitation Learning for Urban Driving. **`NeurIPS 2022`** [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/827cb489449ea216e4a257c47e407d18-Abstract-Conference.html)] [[Code](https://github.com/wayveai/mile)]
- **Iso-Dream**: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models.  **`NeurIPS 2022 Spotlight`** [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9316769afaaeeaad42a9e3633b14e801-Abstract-Conference.html)] [[Code](https://github.com/panmt/Iso-Dream)]
- **Symphony**: Learning Realistic and Diverse Agents for Autonomous Driving Simulation. **`ICRA 2022`** [[Paper](https://arxiv.org/abs/2205.03195)] 
- Hierarchical Model-Based Imitation Learning for Planning in Autonomous Driving. **`IROS 2022`** [[Paper](https://arxiv.org/abs/2210.09539)]
- [**SEM2**] Enhance Sample Efficiency and Robustness of End-to-end Urban Autonomous Driving via Semantic Masked World Model. **`NeurIPS 2022 workshop`** [[Paper](https://arxiv.org/abs/2210.04017)]

## Other World Model Paper

### 2024
- [**SMAC**] Grounded Answers for Multi-agent Decision-making Problem through Generative World Model. **`NeurIPS 2024`** [[Paper](https://arxiv.org/abs/2410.02664)]
- [**CoWorld**] Making Offline RL Online: Collaborative World Models for Offline Visual Reinforcement Learning. **`NeurIPS 2024`** [[Paper](https://arxiv.org/pdf/2305.15260)] [[Website](https://qiwang067.github.io/coworld)] [[Torch Code](https://github.com/qiwang067/CoWorld)]
- [**Diamond**] Diffusion for World Modeling: Visual Details Matter in Atari. **`NeurIPS 2024`**  [[Paper](https://arxiv.org/abs/2405.12399)] [[Code](https://github.com/eloialonso/diamond)]
- **PIVOT-R**: Primitive-Driven Waypoint-Aware World Model for Robotic Manipulation. **`NeurIPS 2024`** [[Paper](https://arxiv.org/pdf/2410.10394)]
- [**MUN**]Learning World Models for Unconstrained Goal Navigation. **`NeurIPS 2024`** [[Paper](https://arxiv.org/abs/2411.02446)] [[Code](https://github.com/RU-Automated-Reasoning-Group/MUN)]
- **VidMan**: Exploiting Implicit Dynamics from Video Diffusion Model for Effective Robot Manipulation. **`NeurIPS 24`** [[Paper](https://arxiv.org/abs/2411.09153)]
- **Adaptive World Models**: Learning Behaviors by Latent Imagination Under Non-Stationarity. **`NeurIPSW 2024`** [[Paper](https://arxiv.org/abs/2411.01342)]
- Emergence of Implicit World Models from Mortal Agents. **`NeurIPSW 2024`** [[Paper](https://arxiv.org/abs/2411.12304)]
- Causal World Representation in the GPT Model. **`NeurIPSW 2024`** [[Paper](https://arxiv.org/abs/2412.07446)]
- **PreLAR**: World Model Pre-training with Learnable Action Representation. **`ECCV 2024`** [[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03363.pdf)] [[Code](https://github.com/zhanglixuan0720/PreLAR)]
- [**CWM**] Understanding Physical Dynamics with Counterfactual World Modeling. **`ECCV 2024`** [[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03523.pdf)] [[Code](https://neuroailab.github.io/cwm-physics/)]
- **ManiGaussian**: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation. **`ECCV 2024`** [[Paper](https://arxiv.org/abs/2403.08321)] [[Code](https://github.com/GuanxingLu/ManiGaussian)]

- [**DWL**] Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning. **`RSS 2024 (Best Paper Award Finalist)`** [[Paper](https://arxiv.org/abs/2408.14472)]
- [**LLM-Sim**] Can Language Models Serve as Text-Based World Simulators? **`ACL`** [[Paper](https://arxiv.org/abs/2406.06485)] [[Code](https://github.com/cognitiveailab/GPT-simulator)]
- **RoboDreamer**: Learning Compositional World Models for Robot Imagination. **`ICML 2024`** [[Paper](https://arxiv.org/abs/2404.12377)] [[Code](https://robovideo.github.io/)]
- [**Δ-IRIS**] Efficient World Models with Context-Aware Tokenization. **`ICML 2024`** [[Paper](https://arxiv.org/abs/2406.19320)] [[Code](https://github.com/vmicheli/delta-iris)]
- **AD3**: Implicit Action is the Key for World Models to Distinguish the Diverse Visual Distractors. **`ICML 2024`** [[Paper](https://arxiv.org/abs/2403.09976)]
- **Hieros**: Hierarchical Imagination on Structured State Space Sequence World Models. **`ICML 2024`** [[Paper](https://arxiv.org/abs/2310.05167)]
- [**HRSSM**] Learning Latent Dynamic Robust Representations for World Models.**`ICML 2024`** [[Paper](https://arxiv.org/abs/2405.06263)] [[Code](https://github.com/bit1029public/HRSSM)]
- **HarmonyDream**: Task Harmonization Inside World Models.**`ICML 2024`** [[Paper](https://openreview.net/forum?id=x0yIaw2fgk)] [[Code](https://github.com/thuml/HarmonyDream)]
- [**REM**] Improving Token-Based World Models with Parallel Observation Prediction.**`ICML 2024`** [[Paper](https://arxiv.org/abs/2402.05643)] [[Code](https://github.com/leor-c/REM)]
- Do Transformer World Models Give Better Policy Gradients? **`ICML 2024`** [[Paper](https://arxiv.org/abs/2402.05290)]
- **TD-MPC2**: Scalable, Robust World Models for Continuous Control. **`ICLR 2024`** [[Paper](https://arxiv.org/pdf/2310.16828)] [[Torch Code](https://github.com/nicklashansen/tdmpc2)]
- **DreamSmooth**: Improving Model-based Reinforcement Learning via Reward Smoothing. **`ICLR 2024`** [[Paper](https://arxiv.org/pdf/2311.01450)]
- [**R2I**] Mastering Memory Tasks with World Models. **`ICLR 2024`** [[Paper](http://arxiv.org/pdf/2403.04253)] [[JAX Code](https://github.com/OpenDriveLab/ViDAR)]
- **MAMBA**: an Effective World Model Approach for Meta-Reinforcement Learning. **`ICLR 2024`**  [[Paper](https://arxiv.org/abs/2403.09859)] [[Code](https://github.com/zoharri/mamba)]
- Multi-Task Interactive Robot Fleet Learning with Visual World Models. **`CoRL 2024`** [[Paper](https://arxiv.org/abs/2410.22689)] [[Code](https://ut-austin-rpl.github.io/sirius-fleet/)]
- **Towards Physically Interpretable World Models**: Meaningful Weakly Supervised Representations for Visual Trajectory Prediction. **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.13772)]
- **Dream to Manipulate**: Compositional World Models Empowering Robot Imitation Learning with Imagination. **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.14957)]  [[Project](https://leobarcellona.github.io/DreamToManipulate/)]
- Transformers Use Causal World Models in Maze-Solving Tasks. **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.11867)]
- **Owl-1**: Omni World Model for Consistent Long Video Generation. **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.09600)] [[Code](https://github.com/huang-yh/Owl)]
- **StoryWeaver**: A Unified World Model for Knowledge-Enhanced Story Character Customization. **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.07375)] [[Code](https://github.com/Aria-Zhangjl/StoryWeaver)]
- **SimuDICE**: Offline Policy Optimization Through World Model Updates and DICE Estimation. **`BNAIC 2024`** [[Paper](https://arxiv.org/abs/2412.06486)]
- Bounded Exploration with World Model Uncertainty in Soft Actor-Critic Reinforcement Learning Algorithm. **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.06139)]
- **Genie 2**: A large-scale foundation world model.  **`2024.12`** **`Google DeepMind`** [[Blog](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)]
- [**NWM**] Navigation World Models.  **`arXiv 2024.12`** **`Yann LeCun`** [[Paper](https://arxiv.org/abs/2412.03572)] [[Project](https://www.amirbar.net/nwm/)]
- **The Matrix**: Infinite-Horizon World Generation with Real-Time Moving Control.  **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.03568)] [[Project](https://thematrix1999.github.io/)]
- **Motion Prompting**: Controlling Video Generation with Motion Trajectories.  **`arXiv 2024.12`** [[Paper](https://arxiv.org/abs/2412.02700)] [[Project](https://motion-prompting.github.io/)]
- Generative World Explorer. **`arXiv 2024.11`** [[Paper](https://arxiv.org/abs/2411.11844)] [[Project](https://generative-world-explorer.github.io/)]
- [**WebDreamer**] Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents. **`arXiv 2024.11`** [[Paper](https://arxiv.org/abs/2411.06559)] [[Code](https://github.com/OSU-NLP-Group/WebDreamer)]
- **WHALE**: Towards Generalizable and Scalable World Models for Embodied Decision-making. **`arXiv 2024.11`** [[Paper](https://arxiv.org/abs/2411.05619)]
- **DINO-WM**: World Models on Pre-trained Visual Features enable Zero-shot Planning. **`arXiv 2024.11`** **`Yann LeCun`** [[Paper](https://arxiv.org/abs/2411.04983)]
- Scaling Laws for Pre-training Agents and World Models. **`arXiv 2024.11`** [[Paper](https://arxiv.org/abs/2411.04434)]
- [**Phyworld**] How Far is Video Generation from World Model: A Physical Law Perspective. **`arXiv 2024.11`** [[Paper](https://arxiv.org/abs/2411.02385)] [[Project](https://phyworld.github.io/)]
- **IGOR**: Image-GOal Representations are the Atomic Control Units for Foundation Models in Embodied AI. **`arXiv 2024.10`** [[Paper](https://arxiv.org/abs/2411.00785)] [[Project](https://www.microsoft.com/en-us/research/project/igor-image-goal-representations/)]
- **EVA**: An Embodied World Model for Future Video Anticipation. **`arXiv 2024.10`** [[Paper](https://arxiv.org/abs/2410.15461)] 
- **VisualPredicator**: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning. **`arXiv 2024.10`** [[Paper](https://arxiv.org/abs/2410.23156)] 
- [**LLMCWM**] Language Agents Meet Causality -- Bridging LLMs and Causal World Models. **`arXiv 2024.10`** [[Paper](https://arxiv.org/abs/2410.19923)] [[Code](https://github.com/j0hngou/LLMCWM/)]
- Reward-free World Models for Online Imitation Learning. **`arXiv 2024.10`** [[Paper](https://arxiv.org/abs/2410.14081)]
- **Web Agents with World Models**: Learning and Leveraging Environment Dynamics in Web Navigation. **`arXiv 2024.10`** [[Paper](https://arxiv.org/abs/2410.13232)]
- [**GLIMO**] Grounding Large Language Models In Embodied Environment With Imperfect World Models. **`arXiv 2024.10`** [[Paper](https://arxiv.org/abs/2410.02664)]
- **AVID**: Adapting Video Diffusion Models to World Models. **`arXiv 2024.10`** [[Paper](https://arxiv.org/abs/2410.12822)] [[Code](https://github.com/microsoft/causica/tree/main/research_experiments/avid)]
- [**WMP**] World Model-based Perception for Visual Legged Locomotion. **`arXiv 2024.9`** [[Paper](https://arxiv.org/abs/2409.16784)] [[Project](https://wmp-loco.github.io/)]
- [**OSWM**] One-shot World Models Using a Transformer Trained on a Synthetic Prior. **`arXiv 2024.9`** [[Paper](https://arxiv.org/abs/2409.14084)]
- **R-AIF**: Solving Sparse-Reward Robotic Tasks from Pixels with Active Inference and World Models. **`arXiv 2024.9`** [[Paper](https://arxiv.org/abs/2409.14216)]
- Representing Positional Information in Generative World Models for Object Manipulation. **`arXiv 2024.9`** [[Paper](https://arxiv.org/abs/2409.12005)]
- Making Large Language Models into World Models with Precondition and Effect Knowledge. **`arXiv 2024.9`** [[Paper](https://arxiv.org/abs/2409.12278)]
- **DexSim2Real$^2$**: Building Explicit World Model for Precise Articulated Object Dexterous Manipulation. **`arXiv 2024.9`** [[Paper](https://arxiv.org/abs/2409.08750)]
- Efficient Exploration and Discriminative World Model Learning with an Object-Centric Abstraction. **`arXiv 2024.8`** [[Paper](https://arxiv.org/abs/2408.11816)]
- [**MoReFree**] World Models Increase Autonomy in Reinforcement Learning. **`arXiv 2024.8`** [[Paper](https://arxiv.org/abs/2408.09807)] [[Project](https://sites.google.com/view/morefree)]
- **UrbanWorld**: An Urban World Model for 3D City Generation. **`arXiv 2024.7`** [[Paper](https://arxiv.org/abs/2407.119656)]
- **PWM**: Policy Learning with Large World Models. **`arXiv 2024.7`** [[Paper](https://arxiv.org/abs/2407.02466)] [[Code](https://www.imgeorgiev.com/pwm/)]
- **Predicting vs. Acting**: A Trade-off Between World Modeling & Agent Modeling. **`arXiv 2024.7`** [[Paper](https://arxiv.org/abs/2407.02446)]
- [**GenRL**] Multimodal foundation world models for generalist embodied agents. **`arXiv 2024.6`** [[Paper](https://arxiv.org/abs/2406.18043)] [[Code](https://github.com/mazpie/genrl)]
- [**DLLM**] World Models with Hints of Large Language Models for Goal Achieving. **`arXiv 2024.6`** [[Paper](http://arxiv.org/pdf/2406.07381)]
- Cognitive Map for Language Models: Optimal Planning via Verbally Representing the World Model. **`arXiv 2024.6`** [[Paper](https://arxiv.org/abs/2406.15275)]
- **CityBench**: Evaluating the Capabilities of Large Language Model as World Model. **`arXiv 2024.6`** [[Paper](https://arxiv.org/abs/2406.13945)] [[Code](https://github.com/tsinghua-fib-lab/CityBench)]
- **CoDreamer**: Communication-Based Decentralised World Models. **`arXiv 2024.6`** [[Paper](https://arxiv.org/abs/2406.13600)]
- [**EBWM**] Cognitively Inspired Energy-Based World Models. **`arXiv 2024.6`** [[Paper](https://arxiv.org/abs/2406.08862)]
- Evaluating the World Model Implicit in a Generative Model. **`arXiv 2024.6`** [[Paper](https://arxiv.org/abs/2406.03689)] [[Code](https://github.com/mazpie/genrl)]
- Transformers and Slot Encoding for Sample Efficient Physical World Modelling. **`arXiv 2024.5`** [[Paper](https://arxiv.org/abs/2405.20180)] [[Code](https://github.com/torchipeppo/transformers-and-slot-encoding-for-wm)]
- [**Puppeteer**] Hierarchical World Models as Visual Whole-Body Humanoid Controllers. **`arXiv 2024.5`** **`Yann LeCun`** [[Paper](https://arxiv.org/abs/2405.18418)] [[Code](https://nicklashansen.com/rlpuppeteer)]
- **BWArea Model**: Learning World Model, Inverse Dynamics, and Policy for Controllable Language Generation. **`arXiv 2024.5`** [[Paper](https://arxiv.org/abs/2405.17039)]
- **Pandora**: Towards General World Model with Natural Language Actions and Video States. [[Paper](https://world-model.maitrix.org/assets/pandora.pdf)] [[Code](https://github.com/maitrix-org/Pandora)]
- [**WKM**] Agent Planning with World Knowledge Model. **`arXiv 2024.5`**  [[Paper](https://arxiv.org/abs/2405.14205)] [[Code](https://github.com/zjunlp/WKM)]
- **Newton**™ – a first-of-its-kind foundation model for understanding the physical world. **`Archetype AI`** [[Blog](https://www.archetypeai.io/blog/introducing-archetype-ai---understand-the-real-world-in-real-time)]
- **Compete and Compose**: Learning Independent Mechanisms for Modular World Models. **`arXiv 2024.4`**  [[Paper](https://arxiv.org/abs/2404.15109)]
- **MagicTime**: Time-lapse Video Generation Models as Metamorphic Simulators. **`arXiv 2024.4`**  [[Paper](https://arxiv.org/abs/2404.05014)] [[Code](https://github.com/PKU-YuanGroup/MagicTime)]
- **Dreaming of Many Worlds**: Learning Contextual World Models Aids Zero-Shot Generalization. **`arXiv 2024.3`**  [[Paper](https://arxiv.org/abs/2403.10967)] [[Code](https://github.com/sai-prasanna/dreaming_of_many_worlds)]
- **ManiGaussian**: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation. **`arXiv 2024.3`**  [[Paper](https://arxiv.org/abs/2403.08321)] [[Code](https://guanxinglu.github.io/ManiGaussian/)]
- **V-JEPA**: Video Joint Embedding Predictive Architecture. **`Meta AI`** **`Yann LeCun`** [[Blog](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)] [[Paper](https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/)] [[Code](https://github.com/facebookresearch/jepa)]
- [**IWM**] Learning and Leveraging World Models in Visual Representation Learning. **`Meta AI`** [[Paper](https://arxiv.org/abs/2403.00504)] 
- **Genie**: Generative Interactive Environments. **`DeepMind`** [[Paper](https://arxiv.org/abs/2402.15391)] [[Blog](https://sites.google.com/view/genie-2024/home)]
- [**Sora**] Video generation models as world simulators. **`OpenAI`** [[Technical report](https://openai.com/research/video-generation-models-as-world-simulators)]
- [**LWM**] World Model on Million-Length Video And Language With RingAttention. **`arXiv 2024.2`**  [[Paper](https://arxiv.org/abs/2402.08268)] [[Code](https://github.com/LargeWorldModel/LWM)]
- Planning with an Ensemble of World Models. **`OpenReview`** [[Paper](https://openreview.net/forum?id=cvGdPXaydP)]
- **WorldDreamer**: Towards General World Models for Video Generation via Predicting Masked Tokens. **`arXiv 2024.1`** [[Paper](https://arxiv.org/abs/2401.09985)] [[Code](https://github.com/JeffWang987/WorldDreamer)]

### 2023
- [**IRIS**] Transformers are Sample Efficient World Models. **`ICLR 2023 Oral`** [[Paper](https://arxiv.org/pdf/2209.00588)] [[Torch Code](https://github.com/eloialonso/iris)]
- **STORM**: Efficient Stochastic Transformer based World Models for Reinforcement Learning. **`NIPS 2023`** [[Paper](https://arxiv.org/pdf/2310.09615)] [[Torch Code](https://github.com/weipu-zhang/STORM)]
- [**TWM**] Transformer-based World Models Are Happy with 100k Interactions. **`ICLR 2023`** [[Paper](https://arxiv.org/pdf/2303.07109)] [[Torch Code](https://github.com/jrobine/twm)]
- [**Dynalang**] Learning to Model the World with Language. **`arXiv 2023.8`** [[Paper](https://arxiv.org/pdf/2308.01399)] [[JAX Code](https://github.com/jlin816/dynalang)]
- [**DreamerV3**] Mastering Atari with Discrete World Models. **`arXiv 2023.1`** [[Paper](https://arxiv.org/pdf/2301.04104)] [[JAX Code](https://github.com/danijar/dreamerv3)] [[Torch Code](https://github.com/NM512/dreamerv3-torch)]
### 2022
- [**TD-MPC**] Temporal Difference Learning for Model Predictive Control. **`ICML 2022`** [[Paper](https://arxiv.org/pdf/2203.04955)][[Torch Code](https://github.com/nicklashansen/tdmpc)]
- **DreamerPro**: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations. **`ICML 2022`** [[Paper](https://proceedings.mlr.press/v162/deng22a/deng22a.pdf)] [[TF Code](https://github.com/fdeng18/dreamer-pro)]
- **DayDreamer**: World Models for Physical Robot Learning. **`CoRL 2022`** [[Paper](https://proceedings.mlr.press/v205/wu23c/wu23c.pdf)] [[TF Code](https://github.com/danijar/daydreamer)]
- Deep Hierarchical Planning from Pixels. **`NIPS 2022`** [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/a766f56d2da42cae20b5652970ec04ef-Paper-Conference.pdf)] [[TF Code](https://github.com/danijar/director)]
- **Iso-Dream**: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models. **`NIPS 2022 Spotlight`** [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/9316769afaaeeaad42a9e3633b14e801-Paper-Conference.pdf)] [[Torch Code](https://github.com/panmt/Iso-Dream)]
- **DreamingV2**: Reinforcement Learning with Discrete World Models without Reconstruction. **`arXiv 2022.3`** [[Paper](https://arxiv.org/pdf/2203.00494)] 
### 2021
- [**DreamerV2**] Mastering Atari with Discrete World Models. **`ICLR 2021`** [[Paper](https://arxiv.org/pdf/2010.02193)] [[TF Code](https://github.com/danijar/dreamerv2)] [[Torch Code](https://github.com/jsikyoon/dreamer-torch)]
- **Dreaming**: Model-based Reinforcement Learning by Latent Imagination without Reconstruction. **`ICRA 2021`** [[Paper](https://arxiv.org/pdf/2007.14535)]
### 2020
- [**DreamerV1**] Dream to Control: Learning Behaviors by Latent Imagination. **`ICLR 2020`** [[Paper](https://arxiv.org/pdf/1912.01603)] [[TF Code](https://github.com/danijar/dreamer)] [[Torch Code](https://github.com/juliusfrost/dreamer-pytorch)]
- [**Plan2Explore**] Planning to Explore via Self-Supervised World Models. **`ICML 2020`** [[Paper](https://arxiv.org/pdf/2005.05960)] [[TF Code](https://github.com/ramanans1/plan2explore)] [[Torch Code](https://github.com/yusukeurakami/plan2explore-pytorch)]

### 2018
* World Models. **`NIPS 2018 Oral`** [[Paper](https://arxiv.org/pdf/1803.10122)]
