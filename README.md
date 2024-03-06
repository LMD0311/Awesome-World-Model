# Awesome World Models for Autonomous Driving [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

Collect some World Models (for Autonomous Driving) papers. 

If you find some ignored papers, **feel free to [*create pull requests*](https://github.com/LMD0311/Awesome-World-Model/blob/main/ContributionGuidelines.md), [*open issues*](https://github.com/LMD0311/Awesome-World-Model/issues/new), or [*email* me](mailto:xzhou03@hust.edu.cn)**. Contributions in any form to make this list more comprehensive are welcome. 📣📣📣

If you find this repository useful, please consider **[citing](#citation)** and **giving us a star** 🌟. 

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
- Introducing GAIA-1: A Cutting-Edge Generative AI Model for Autonomy [[blog](https://wayve.ai/thinking/introducing-gaia1/)] 
  > World models are the basis for the ability to predict what might happen next, which is fundamentally important for autonomous driving. They can act as a learned simulator, or a mental “what if” thought experiment for model-based reinforcement learning (RL) or planning. By incorporating world models into our driving models, we can enable them to understand human decisions better and ultimately generalise to more real-world situations.
  

## 2024

- [**ViDAR**] Visual Point Cloud Forecasting enables Scalable Autonomous Driving. **`CVPR 2024`** [[Paper](https://arxiv.org/abs/2312.17655)] [[Code](https://github.com/OpenDriveLab/ViDAR)]
- [**Cam4DOCC**] Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications. **`CVPR 2024`** [[Paper](https://arxiv.org/abs/2311.17663)] [[Code](https://github.com/haomo-ai/Cam4DOcc)]
- [**Drive-WM**] Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving. **`CVPR 2024`** [[Paper](https://arxiv.org/abs/2311.17918)] [[Code](https://github.com/BraveGroup/Drive-WM)]
- Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion. **`ICLR 2024`** [[Paper](https://arxiv.org/abs/2311.01017)]
- [**SafeDreamer**] SafeDreamer: Safe Reinforcement Learning with World Models. **`ICLR 2024`** [[Paper](https://openreview.net/forum?id=tsE5HLYtYg)] [[Code](https://github.com/PKU-Alignment/SafeDreamer)]
- [**Genie**] Genie: Generative Interactive Environments. **`DeepMind`** [[Paper](https://arxiv.org/abs/2402.15391)] [[Blog](https://sites.google.com/view/genie-2024/home)]
- [**Sora**] Video generation models as world simulators. **`OpenAI`** [[Technical report](https://openai.com/research/video-generation-models-as-world-simulators)]
- [**IWM**] Learning and Leveraging World Models in Visual Representation Learning. **`Meta AI`** [[Paper](https://arxiv.org/abs/2403.00504)] 
- [**V-JEPA**] V-JEPA: Video Joint Embedding Predictive Architecture. **`Meta AI`** [[Blog](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)] [[Paper](https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/)] [[Code](https://github.com/facebookresearch/jepa)]
- [**Think2Drive**] Think2Drive: Efficient Reinforcement Learning by Thinking in Latent World Model for Quasi-Realistic Autonomous Driving (in CARLA-v2) **`2024.2, arxiv`** [[Paper](https://arxiv.org/abs/2402.16720)]
- [**LWM**] World Model on Million-Length Video And Language With RingAttention. **`2024.2, arxiv`**  [[Paper](https://arxiv.org/abs/2402.08268)] [[Code](https://github.com/LargeWorldModel/LWM)]
- Planning with an Ensemble of World Models. **`OpenReview`** [[Paper](https://openreview.net/forum?id=cvGdPXaydP)]
- [**WorldDreamer**] WorldDreamer: Towards General World Models for Video Generation via Predicting Masked Tokens. **`2024.1, arxiv`** [[Paper](https://arxiv.org/abs/2401.09985)] [[Code](https://github.com/JeffWang987/WorldDreamer)]

## 2023

- [**TrafficBots**] TrafficBots: Towards World Models for Autonomous Driving Simulation and Motion Prediction. **`ICRA 2023`** [[Paper](https://arxiv.org/abs/2303.04116)] [[Code](https://github.com/zhejz/TrafficBots)]
- [**WoVoGen**] WoVoGen: World Volume-aware Diffusion for Controllable Multi-camera Driving Scene Generation. **`2023.12, arxiv`** [[Paper](https://arxiv.org/abs/2312.02934)] [[Code](https://github.com/fudan-zvg/WoVoGen)]
- [**CTT**] Categorical Traffic Transformer: Interpretable and Diverse Behavior Prediction with Tokenized Latent. **`2023.11, arxiv`** [[Paper](https://arxiv.org/abs/2311.18307)]
- [**OccWorld**] OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving. **`2023.11, arxiv`** [[Paper](https://arxiv.org/abs/2311.16038)] [[Code](https://github.com/wzzheng/OccWorld)]
- [**MUVO**] MUVO: A Multimodal Generative World Model for Autonomous Driving with Geometric Representations.(2023.11, arxiv) [[Paper](https://arxiv.org/abs/2311.11762)]
- [**GAIA-1**] GAIA-1: A Generative World Model for Autonomous Driving. **`2023.9, arxiv`** [[Paper](https://arxiv.org/abs/2309.17080)]
- [**ADriver-I**] ADriver-I: A General World Model for Autonomous Driving. **`2023.9, arxiv`** [[Paper](https://arxiv.org/abs/2311.13549)]
- [**DriveDreamer**] DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. **`2023.9, arxiv`** [[Paper](https://arxiv.org/abs/2309.09777)] [[Code](https://github.com/JeffWang987/DriveDreamer)]
- [**UniWorld**] UniWorld: Autonomous Driving Pre-training via World Models. **`2023.8, arxiv`** [[Paper](https://arxiv.org/abs/2308.07234)] [[Code](https://github.com/chaytonmin/UniWorld)]

## 2022

- [**MILE**] Model-Based Imitation Learning for Urban Driving. **`NeurIPS 2022`** [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/827cb489449ea216e4a257c47e407d18-Abstract-Conference.html)] [[Code](https://github.com/wayveai/mile)]
- [**Symphony**] Symphony: Learning Realistic and Diverse Agents for Autonomous Driving Simulation. **`ICRA 2022`** [[Paper](https://arxiv.org/abs/2205.03195)] 
- Hierarchical Model-Based Imitation Learning for Planning in Autonomous Driving. **`IROS 2022`** [[Paper](https://arxiv.org/abs/2210.09539)]

