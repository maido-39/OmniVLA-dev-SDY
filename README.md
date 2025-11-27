# OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://omnivla-nav.github.io)


[Noriaki Hirose](https://sites.google.com/view/noriaki-hirose/)<sup>1, 2</sup>, [Catherine Glossop](https://catglossop.github.io/)<sup>1</sup>, [Dhruv Shah](https://robodhruv.github.io/)<sup>3</sup>, [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)<sup>1</sup>

<sup>1</sup> UC Berkeley (_Berkeley AI Research_),  <sup>2</sup> Toyota Motor North America, ,  <sup>3</sup> Princeton University

### Installation
Please set up a conda environment (see instructions in [SETUP.md](SETUP.md)).

### Inference
1. Download our checkpoints and place them in our directory. "omnivla-original" is the trained checkpoints of the OmniVLA for paper submission. "omnivla-original-balance" contains the trained checkpoints of OmniVLA that account for the data balance in the LeLaN dataset. And "omnivla-finetuned-cast" is finetuned checkpoints with the [CAST](https://huggingface.co/datasets/catglossop/CAST-dataset) dataset.
    ```
    git clone https://huggingface.co/NHirose/omnivla-original
    git clone https://huggingface.co/NHirose/omnivla-original-balance    
    git clone https://huggingface.co/NHirose/omnivla-finetuned-cast
    ```
2. Run OmniVLA using a sample current image, goal images, GPS pose, and language prompt. You can view the generated trajectory in the output figure 1_ex.jpg.
    ```
    python inference/run_omnivla.py
    ```
3. Change the goal modality: by default, our code generates actions based on the language prompt. To use a different modality, you can modify the settings around line 560. 
    
4. Run OmniVLA to control the real robot. Modify "run_omnivla.py" to update the robot’s state (camera image, GPS signal) and adjust the goal information accordingly. Then, feed the generated velocity commands to your robot.

5. To try the finetuned checkpoints with the CAST dataset, update the path and step number in "InferenceConfig" within "run_omnivla.py".

### Isaac Sim Simulation

This repository includes integration with NVIDIA Isaac Sim for simulating OmniVLA navigation with a quadruped robot (Spot).

#### Prerequisites

1. **NVIDIA Isaac Sim**: Install Isaac Sim 4.5.0 or later from [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/isaac-sim/).
2. **Submodule Setup**: The Isaac Sim Spot simulation environment is included as a submodule:
   ```bash
   git submodule update --init --recursive
   ```

#### Running the Simulation

1. **Start Isaac Sim** and ensure it's properly configured.

2. **Run the simulation** (choose one):
   
   **Basic simulation** (without prompt editing):
   ```bash
   python simulations/quadruped_simulation.py
   ```
   
   **Simulation with real-time prompt editing**:
   ```bash
   python simulations/quadruped_simulation_with_prompt_editor.py
   ```

3. **Simulation Features**:
   - OmniVLA inference runs in a separate thread for real-time performance
   - Shared memory communication between Isaac Sim and OmniVLA
   - Real-time visualization with Pygame (ego and top-down camera views)
   - Automatic CSV logging of observations, actions, and commands
   - Automatic visualization generation upon simulation completion
   - **Real-time language prompt editing** (in `quadruped_simulation_with_prompt_editor.py`):
     - Edit `sim_data/current_prompt.txt` during simulation to update the language prompt
     - Changes are automatically detected and applied to the OmniVLA model
     - Terminal notifications show when prompts are updated

4. **Data Output**:
   - Simulation data is saved to `sim_data/YYYY-MM-DD_HH-MM-SS-omnivla_sim/`
   - CSV file (`data.csv`) contains: timestamps, robot positions, headings, goal information, velocity commands, and waypoints
   - Visualization plots are automatically generated and saved:
     - `trajectory_and_commands.png`: Trajectory, velocity commands, and heading plots
     - `analysis.png`: Distance to goal, velocity magnitude, waypoint directions, and heading error

5. **Visualizing Results** (standalone):
   ```bash
   python simulations/visualize_simulation.py sim_data/YYYY-MM-DD_HH-MM-SS-omnivla_sim/data.csv --wall-size 10.0 --save-dir <output_directory>
   ```

#### Configuration

**Basic Simulation** (`quadruped_simulation.py`):
- Modify goal image and language prompt in the script (around line 327-328)
- Adjust simulation parameters in `extern/isaacsim-spot-remotecontroldemo/quadruped_example.py` (DEFAULT_CONFIG)

**Simulation with Prompt Editor** (`quadruped_simulation_with_prompt_editor.py`):
- Initial language prompt is set in the script (around line 359)
- **Real-time prompt editing**: Edit `sim_data/current_prompt.txt` during simulation
- The file is automatically created at startup with the initial prompt
- Changes to the file are detected and applied automatically
- Terminal output shows prompt update notifications

**Yaw Response Tuning**:
- Adjust `yaw_gain` parameter in `SimOmniVLAConfig` to control angular velocity response
- Default: `1.5`, higher values (e.g., `2.0`, `2.5`) increase yaw sensitivity
- Example: `yaw_gain=2.5` in `quadruped_simulation_with_prompt_editor.py` (around line 383)

**Other Settings**:
- Camera settings, robot limits, and environment parameters can be customized in `extern/isaacsim-spot-remotecontroldemo/quadruped_example.py` (DEFAULT_CONFIG)
- Number of boxes: Modify `DEFAULT_CONFIG["num_boxes"]` in the simulation script

#### Notes

- The simulation uses `enable_csv_logging=False` and `enable_image_saving=False` by default for performance
- To save camera images, set `enable_image_saving=True` in the simulation initialization
- Press `Ctrl+C` to gracefully terminate the simulation
- Experiment data directories (`expr_data/`, `sim_data/`) are automatically ignored by git

### Training
We provide the training code along with a sample dataloader to help you quickly understand the required data loading structure. Since preparing the full training dataset is resource-intensive, we include this simplified code base for convenience.

1. Downloading MBRA project code base:
    ```
    cd ..
    git clone https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA.git
    ```
2. Downloading MBRA model:
    ```
    cd OmniVLA_internal
    git clone https://huggingface.co/NHirose/MBRA/
    ```
3. You can set the training or debugging mode at line 10 in vla-scripts/train_omnivla.py. Note that even in debugging mode, the code requires at least 20 GB of GPU memory (we use an NVIDIA RTX 4090).

4. You can configure visualization at line 11 in vla-scripts/train_omnivla.py. During training, it should be set to False.
    
5. Training our policy from OpenVLA checkpoints (Please fill X):
    ```
    torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/train_omnivla.py  --vla_path openvla/openvla-7b --dataset_name omnivla --num_images_in_input 2 --batch_size X --wandb_entity "X" --wandb_project "omnivla"
    ```
6. Finetuning our OmniVLA (Please fill X):
    ```
    torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/train_omnivla.py  --vla_path ./omnivla-original --dataset_name omnivla --num_images_in_input 2 --batch_size X --wandb_entity "X" --wandb_project "omnivla"
    ````
7. Memo finetuning our OmniVLA on our large navigation dataset:
    ```
    conda activate omnivla_2
    cd /media/noriaki/Noriaki_Data/OmniVLA
    torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train_omnivla_dataset.py  --vla_path ./omnivla-original --dataset_name omnivla --wandb_entity "noriaki-hirose"   --wandb_project "omnivla"
    ```

### Training with GNM, LeLaN, Frodobots, BDD and CAST datasets
We provide training code that supports multiple public datasets. Before following the full training process, please first ensure that you can run the example training with the sample dataloader.

1. Downloading all datasets from the original website. ([GNM](https://github.com/robodhruv/visualnav-transformer), [LeLaN](https://github.com/NHirose/learning-language-navigation), [Frodobots](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA), [CAST](https://openvla-oft.github.io/)) Please verify that the downloaded datasets work properly in their original codebase, except BDD dataset.

2. Downloading the modified BDD dataset with MBRA annotations from [here](https://huggingface.co/datasets/NHirose/BDD_OmniVLA) and extract it. The image sequences in the modified dataset remain subject to the [original BDD license](http://bdd-data.berkeley.edu/download.html), while the additional MBRA annotations are released under the MIT license.

3. Downloading the lerobot code base for the Frodobots dataset dataloader:
    ```
    git clone https://github.com/huggingface/lerobot.git 
    ```
4. Edit the data path in config_nav/mbra_and_dataset_config.yaml:

5. Training our policy from OpenVLA checkpoints (Please fill X):
    ```
    torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/train_omnivla_dataset.py  --vla_path ./omnivla-original --dataset_name omnivla --wandb_entity "X"   --wandb_project "omnivla"
    ```
       
In our training setup, we use 8 Nvidia H100 GPUs (80 GB each) across 8 nodes. The batch sizes are configured as [LeLaN, GNM, Frodobots, BDD] = [4, 1, 1, 1], with gradient accumulation set to 4 steps. When finetuning with CAST dataset, we set the batch size as [LeLaN, CAST, GNM, Frodobots, BDD] = [2, 2, 1, 1, 1]. To do so, you need to directly edit train_omnivla_dataset.py.
    
### Acknowledgement
We implement our ideas and design choices on top of the pretrained checkpoints. Our work builds upon the [OpenVLA-OFT](https://openvla-oft.github.io/) codebase, with additional code added to create OmniVLA. As such, our implementation leverages many components of the OpenVLA-OFT codebase. We sincerely appreciate the effort and contributions of the OpenVLA-OFT team!

**Isaac Sim Integration**: The simulation environment is based on the Isaac Sim Spot Remote Control Demo, which is included as a submodule (`extern/isaacsim-spot-remotecontroldemo`). The Isaac Sim code is subject to NVIDIA's license terms. The OmniVLA integration code (`simulations/quadruped_simulation.py`, `simulations/quadruped_simulation_with_prompt_editor.py`, `simulations/visualize_simulation.py`, `inference/sim_omnivla.py`) is released under the MIT license as part of this repository.

## Citing
```
@misc{hirose2025omnivla,
      title={OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation}, 
      author={Noriaki Hirose and Catherine Glossop and Dhruv Shah and Sergey Levine},
      year={2025},
      eprint={2509.19480},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.19480}, 
}
```

## Additional Contributions

**Isaac Sim Simulation Integration** (Copyright (c) 2025 maido-39)

The following components were added by maido-39 to enable OmniVLA simulation with Isaac Sim:

- **Simulation Integration** (`simulations/quadruped_simulation.py`): Integration of OmniVLA with NVIDIA Isaac Sim for quadruped robot (Spot) simulation, including shared memory communication, multi-threaded inference, and real-time visualization.

- **Simulation with Prompt Editor** (`simulations/quadruped_simulation_with_prompt_editor.py`): Enhanced simulation script with real-time language prompt editing capability. Allows users to modify the language prompt during simulation by editing a text file, with automatic detection and application of changes.

- **OmniVLA Simulation Wrapper** (`inference/sim_omnivla.py`): Wrapper class for running OmniVLA inference in simulation environments, with shared memory channels for observation and command exchange. Includes configurable `yaw_gain` parameter for tuning angular velocity response.

- **Simulation Visualization** (`simulations/visualize_simulation.py`): Comprehensive visualization tools for analyzing simulation results, including trajectory plots, velocity commands, and performance metrics.

- **CSV Data Logging**: Automatic logging of observations, actions, velocity commands, and waypoints during simulation runs.

- **Shared Memory Communication System**: High-performance inter-thread communication for real-time data exchange between Isaac Sim and OmniVLA inference threads.

- **File-based Prompt Editing**: Real-time language prompt modification system that monitors `sim_data/current_prompt.txt` and automatically applies changes to the OmniVLA model during simulation.

These additions are released under the MIT license as part of this repository.
