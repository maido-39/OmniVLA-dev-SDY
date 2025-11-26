# Setup Instructions

## Set Up Conda Environment

```bash
# Create and activate conda environment
conda create -n omnivla python=3.10 -y
conda activate omnivla

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install numpy==1.26.4 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# Clone openvla-oft repo and pip install to download dependencies
git clone https://github.com/NHirose/OmniVLA.git
cd OmniVLA
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

## IsaacSim Spot Remote Control Demo 서브모듈

OmniVLA는 `extern/isaacsim-spot-remotecontroldemo` 경로에 [maido-39/Isaacsim-spot-remotecontroldemo](https://github.com/maido-39/Isaacsim-spot-remotecontroldemo) 리포지토리를 서브모듈로 포함합니다. Spot 시뮬레이션/원격제어 코드는 여기에서 바로 참조할 수 있으며, 추후 `inference/` 하위 스크립트에서 import 하거나 자산을 불러오는 용도로 사용할 수 있습니다.

### 새로 클론할 때

```bash
git clone --recurse-submodules https://github.com/NHirose/OmniVLA.git
```

### 기존 클론에서 업데이트할 때

```bash
git submodule update --init --recursive
```

### 경로 활용 예시

```bash
export ISAACSIM_SPOT_REPO=${PWD}/extern/isaacsim-spot-remotecontroldemo
python inference/run_omnivla-ARIL.py --isaacsim-spot-root "${ISAACSIM_SPOT_REPO}"
```

추후 inference 코드에서 IsaacSim 유틸을 불러올 때 위 환경 변수를 사용하거나, `extern/isaacsim-spot-remotecontroldemo` 경로를 직접 참조하면 됩니다.
