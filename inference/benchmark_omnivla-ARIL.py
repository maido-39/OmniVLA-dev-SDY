# ===============================================================
# OmniVLA Inference Benchmark
# ===============================================================
# 
# 벤치마크 코드: inference 속도 측정
# ---------------------------
# Paths and System Setup
# ---------------------------
import sys, os
sys.path.insert(0, '..')

import time, math, json
from typing import Optional, Tuple, Type, Dict, List
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import utm

# ---------------------------
# Custom Imports
# ---------------------------
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.projectors import ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead_idcat, L1RegressionDistHead
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE

from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

# ===============================================================
# Utility Functions
# ===============================================================
def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    if not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt")) and module_name == "pose_projector":
        module_name = "proprio_projector"
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)

def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")

def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: "InferenceConfig",
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
) -> DDP:
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return module

# ===============================================================
# Benchmark Timer Context Manager
# ===============================================================
class Timer:
    def __init__(self, name: str, track_vram: bool = False):
        self.name = name
        self.start_time = None
        self.elapsed_time = 0.0
        self.track_vram = track_vram
        self.start_vram = 0.0
        self.end_vram = 0.0
        self.peak_vram = 0.0
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if self.track_vram:
                self.start_vram = torch.cuda.memory_allocated() / (1024**3)  # GB
                torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if self.track_vram:
                self.end_vram = torch.cuda.memory_allocated() / (1024**3)  # GB
                self.peak_vram = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        self.elapsed_time = time.time() - self.start_time
    
    def get_time(self) -> float:
        return self.elapsed_time
    
    def get_vram_allocated(self) -> float:
        """Returns VRAM allocated at the end of the operation in GB"""
        return self.end_vram
    
    def get_vram_peak(self) -> float:
        """Returns peak VRAM usage during the operation in GB"""
        return self.peak_vram
    
    def get_vram_delta(self) -> float:
        """Returns VRAM delta (end - start) in GB"""
        return self.end_vram - self.start_vram

# ===============================================================
# Benchmark Statistics
# ===============================================================
class BenchmarkStats:
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.vram_allocated: Dict[str, List[float]] = defaultdict(list)
        self.vram_peak: Dict[str, List[float]] = defaultdict(list)
        self.vram_delta: Dict[str, List[float]] = defaultdict(list)
    
    def add_timing(self, name: str, elapsed_time: float):
        self.timings[name].append(elapsed_time)
    
    def add_vram(self, name: str, allocated: float, peak: float, delta: float):
        self.vram_allocated[name].append(allocated)
        self.vram_peak[name].append(peak)
        self.vram_delta[name].append(delta)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        if name not in self.timings or len(self.timings[name]) == 0:
            return {}
        
        times = self.timings[name]
        return {
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
            "median": np.median(times),
            "p95": np.percentile(times, 95),
            "p99": np.percentile(times, 99),
        }
    
    def get_vram_stats(self, name: str) -> Dict[str, float]:
        stats = {}
        if name in self.vram_allocated and len(self.vram_allocated[name]) > 0:
            stats["allocated_mean"] = np.mean(self.vram_allocated[name])
            stats["allocated_max"] = np.max(self.vram_allocated[name])
            stats["peak_mean"] = np.mean(self.vram_peak[name])
            stats["peak_max"] = np.max(self.vram_peak[name])
            stats["delta_mean"] = np.mean(self.vram_delta[name])
        return stats
    
    def print_summary(self):
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        for name in sorted(self.timings.keys()):
            stats = self.get_stats(name)
            if stats:
                print(f"\n{name}:")
                print(f"  Time - Mean:   {stats['mean']*1000:.2f} ms")
                print(f"  Time - Std:    {stats['std']*1000:.2f} ms")
                print(f"  Time - Min:    {stats['min']*1000:.2f} ms")
                print(f"  Time - Max:    {stats['max']*1000:.2f} ms")
                print(f"  Time - Median: {stats['median']*1000:.2f} ms")
                print(f"  Time - P95:    {stats['p95']*1000:.2f} ms")
                print(f"  Time - P99:    {stats['p99']*1000:.2f} ms")
                
                vram_stats = self.get_vram_stats(name)
                if vram_stats:
                    print(f"  VRAM - Allocated (mean): {vram_stats['allocated_mean']:.3f} GB")
                    print(f"  VRAM - Allocated (max):  {vram_stats['allocated_max']:.3f} GB")
                    print(f"  VRAM - Peak (mean):     {vram_stats['peak_mean']:.3f} GB")
                    print(f"  VRAM - Peak (max):      {vram_stats['peak_max']:.3f} GB")
                    print(f"  VRAM - Delta (mean):    {vram_stats['delta_mean']:.3f} GB")
        
        print("\n" + "="*80)
    
    def save_to_json(self, filepath: str):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        results = {}
        for name in sorted(self.timings.keys()):
            stats = self.get_stats(name)
            if stats:
                results[name] = {k: float(v) for k, v in stats.items()}
                results[name]["all_timings"] = [float(t) for t in self.timings[name]]
                
                vram_stats = self.get_vram_stats(name)
                if vram_stats:
                    results[name].update({k: float(v) for k, v in vram_stats.items()})
                    results[name]["all_vram_allocated"] = [float(v) for v in self.vram_allocated[name]]
                    results[name]["all_vram_peak"] = [float(v) for v in self.vram_peak[name]]
                    results[name]["all_vram_delta"] = [float(v) for v in self.vram_delta[name]]
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark results saved to: {filepath}")
    
    def visualize(self, save_path: str = None):
        """Create matplotlib visualizations for timing and VRAM usage"""
        if not torch.cuda.is_available():
            print("CUDA not available, skipping VRAM visualization")
            return
        
        # Filter out non-measurement timings
        measurement_names = [n for n in self.timings.keys() 
                            if n not in ["model_loading", "total_inference"]]
        
        if not measurement_names:
            print("No measurement data to visualize")
            return
        
        fig = plt.figure(figsize=(16, 10))
        
        # Plot 1: Timing comparison (bar chart)
        ax1 = plt.subplot(2, 2, 1)
        names = []
        means = []
        stds = []
        for name in sorted(measurement_names):
            stats = self.get_stats(name)
            if stats:
                names.append(name.replace("_", "\n"))
                means.append(stats['mean'] * 1000)  # Convert to ms
                stds.append(stats['std'] * 1000)
        
        x_pos = np.arange(len(names))
        ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('Operation')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Inference Time by Operation')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: VRAM usage (bar chart)
        ax2 = plt.subplot(2, 2, 2)
        vram_names = []
        vram_means = []
        vram_peaks = []
        for name in sorted(measurement_names):
            vram_stats = self.get_vram_stats(name)
            if vram_stats and vram_stats.get('allocated_mean', 0) > 0:
                vram_names.append(name.replace("_", "\n"))
                vram_means.append(vram_stats['allocated_mean'])
                vram_peaks.append(vram_stats['peak_mean'])
        
        if vram_names:
            x_pos_vram = np.arange(len(vram_names))
            width = 0.35
            ax2.bar(x_pos_vram - width/2, vram_means, width, label='Allocated (mean)', alpha=0.7, color='lightcoral')
            ax2.bar(x_pos_vram + width/2, vram_peaks, width, label='Peak (mean)', alpha=0.7, color='coral')
            ax2.set_xlabel('Operation')
            ax2.set_ylabel('VRAM (GB)')
            ax2.set_title('VRAM Usage by Operation')
            ax2.set_xticks(x_pos_vram)
            ax2.set_xticklabels(vram_names, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Timing distribution (box plot)
        ax3 = plt.subplot(2, 2, 3)
        timing_data = []
        timing_labels = []
        for name in sorted(measurement_names):
            if name in self.timings and len(self.timings[name]) > 0:
                timing_data.append([t * 1000 for t in self.timings[name]])  # Convert to ms
                timing_labels.append(name.replace("_", "\n"))
        
        if timing_data:
            bp = ax3.boxplot(timing_data, labels=timing_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightgreen')
                patch.set_alpha(0.7)
            ax3.set_xlabel('Operation')
            ax3.set_ylabel('Time (ms)')
            ax3.set_title('Timing Distribution (Box Plot)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: VRAM over iterations (line plot for forward_pass)
        ax4 = plt.subplot(2, 2, 4)
        if "forward_pass" in self.vram_allocated and len(self.vram_allocated["forward_pass"]) > 0:
            iterations = range(len(self.vram_allocated["forward_pass"]))
            ax4.plot(iterations, self.vram_allocated["forward_pass"], 'o-', label='Allocated', alpha=0.7, color='blue')
            if "forward_pass" in self.vram_peak:
                ax4.plot(iterations, self.vram_peak["forward_pass"], 's-', label='Peak', alpha=0.7, color='red')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('VRAM (GB)')
            ax4.set_title('VRAM Usage Over Iterations (forward_pass)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

# ===============================================================
# Benchmark Inference Class
# ===============================================================
class BenchmarkInference:
    def __init__(self, save_dir, lan_inst_prompt, goal_utm, goal_compass, goal_image_PIL, 
                 action_tokenizer, processor, vla, action_head, pose_projector, device_id, NUM_PATCHES,
                 stats: BenchmarkStats):
        self.tick_rate = 3
        self.lan_inst_prompt = lan_inst_prompt
        self.goal_utm = goal_utm
        self.goal_compass = goal_compass
        self.goal_image_PIL = goal_image_PIL
        self.action_tokenizer = action_tokenizer
        self.processor = processor
        self.vla = vla
        self.action_head = action_head
        self.pose_projector = pose_projector
        self.device_id = device_id
        self.NUM_PATCHES = NUM_PATCHES
        self.count_id = 0
        self.linear, self.angular = 0.0, 0.0
        self.datastore_path_image = save_dir
        self.stats = stats
        
        # Modality flags (from original file)
        self.pose_goal = True
        self.satellite = False
        self.image_goal = False
        self.lan_prompt = False

    # ----------------------------
    # Static Utility Methods
    # ----------------------------
    @staticmethod
    def calculate_relative_position(x_a, y_a, x_b, y_b):
        return x_b - x_a, y_b - y_a

    @staticmethod
    def rotate_to_local_frame(delta_x, delta_y, heading_a_rad):
        rel_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        rel_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return rel_x, rel_y

    # ----------------------------
    # Benchmark Run
    # ----------------------------
    def run_benchmark(self, num_iterations: int = 10, warmup: int = 2):
        print(f"\nStarting benchmark with {warmup} warmup iterations and {num_iterations} measurement iterations...")
        
        # Warmup
        print("Warming up...")
        for i in range(warmup):
            self.run_omnivla_benchmark()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Actual benchmark
        print("Running benchmark measurements...")
        for i in range(num_iterations):
            print(f"  Iteration {i+1}/{num_iterations}...", end="\r")
            self.run_omnivla_benchmark()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\nCompleted {num_iterations} iterations.")

    # ----------------------------
    # OmniVLA Inference with Benchmarking
    # ----------------------------
    def run_omnivla_benchmark(self):
        thres_dist = 30.0
        metric_waypoint_spacing = 0.1

        # Load current GPS & heading
        current_lat = 37.87371258374039
        current_lon = -122.26729417226024
        current_compass = 270.0
        cur_utm = utm.from_latlon(current_lat, current_lon)
        cur_compass = -float(current_compass) / 180.0 * math.pi

        # Override current UTM and compass
        cur_utm = [1.5864308, -1.3926477]
        self.goal_utm = [-12.2266438, 12.6290138]
        cur_compass = 0.825678
        self.goal_compass = 0.91930187

        # Local goal position calculation
        with Timer("goal_position_calculation", track_vram=torch.cuda.is_available()) as timer:
            delta_x, delta_y = self.calculate_relative_position(
                cur_utm[0], cur_utm[1], self.goal_utm[0], self.goal_utm[1]
            )
            relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)
            radius = np.sqrt(relative_x**2 + relative_y**2)
            if radius > thres_dist:
                relative_x *= thres_dist / radius
                relative_y *= thres_dist / radius

            goal_pose_loc_norm = np.array([
                relative_y / metric_waypoint_spacing,
                -relative_x / metric_waypoint_spacing,
                np.cos(self.goal_compass - cur_compass),
                np.sin(self.goal_compass - cur_compass)
            ])
        self.stats.add_timing("goal_position_calculation", timer.get_time())
        if torch.cuda.is_available():
            self.stats.add_vram("goal_position_calculation", 
                               timer.get_vram_allocated(), 
                               timer.get_vram_peak(), 
                               timer.get_vram_delta())

        # Load current image
        with Timer("image_loading", track_vram=torch.cuda.is_available()) as timer:
            current_image_path = "./inference/frame0-0.000-ego.jpg"
            current_image_PIL = Image.open(current_image_path).convert("RGB")
        self.stats.add_timing("image_loading", timer.get_time())
        if torch.cuda.is_available():
            self.stats.add_vram("image_loading", 
                               timer.get_vram_allocated(), 
                               timer.get_vram_peak(), 
                               timer.get_vram_delta())

        # Language instruction
        lan_inst = self.lan_inst_prompt if self.lan_prompt else "xxxx"

        # Prepare batch
        with Timer("data_preprocessing", track_vram=torch.cuda.is_available()) as timer:
            batch = self.data_transformer_omnivla(
                current_image_PIL, lan_inst, self.goal_image_PIL, goal_pose_loc_norm,
                prompt_builder=PurePromptBuilder,
                action_tokenizer=self.action_tokenizer,
                processor=self.processor
            )
        self.stats.add_timing("data_preprocessing", timer.get_time())
        if torch.cuda.is_available():
            self.stats.add_vram("data_preprocessing", 
                               timer.get_vram_allocated(), 
                               timer.get_vram_peak(), 
                               timer.get_vram_delta())

        # Run forward pass
        with Timer("forward_pass", track_vram=torch.cuda.is_available()) as timer:
            actions, modality_id = self.run_forward_pass(
                vla=self.vla.eval(),
                action_head=self.action_head.eval(),
                noisy_action_projector=None,
                pose_projector=self.pose_projector.eval(),
                batch=batch,
                action_tokenizer=self.action_tokenizer,
                device_id=self.device_id,
                use_l1_regression=True,
                use_diffusion=False,
                use_film=False,
                num_patches=self.NUM_PATCHES,
                compute_diffusion_l1=False,
                num_diffusion_steps_train=None,
                mode="train",
                idrun=self.count_id,
            )
        self.stats.add_timing("forward_pass", timer.get_time())
        if torch.cuda.is_available():
            self.stats.add_vram("forward_pass", 
                               timer.get_vram_allocated(), 
                               timer.get_vram_peak(), 
                               timer.get_vram_delta())
        self.count_id += 1

        # Post-processing
        with Timer("post_processing", track_vram=torch.cuda.is_available()) as timer:
            waypoints = actions.float().cpu().numpy()
            waypoint_select = 4
            chosen_waypoint = waypoints[0][waypoint_select].copy()
            chosen_waypoint[:2] *= metric_waypoint_spacing
            dx, dy, hx, hy = chosen_waypoint

            # PD controller
            EPS = 1e-8
            DT = 1 / 3
            if np.abs(dx) < EPS and np.abs(dy) < EPS:
                linear_vel_value = 0
                angular_vel_value = 1.0 * np.arctan2(hy, hx) / DT
            elif np.abs(dx) < EPS:
                linear_vel_value = 0
                angular_vel_value = 1.0 * np.sign(dy) * np.pi / (2 * DT)
            else:
                linear_vel_value = dx / DT
                angular_vel_value = np.arctan(dy / dx) / DT

            linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
            angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)

            # Velocity limitation
            maxv, maxw = 0.3, 0.3
            if np.abs(linear_vel_value) <= maxv:
                if np.abs(angular_vel_value) <= maxw:
                    linear_vel_value_limit = linear_vel_value
                    angular_vel_value_limit = angular_vel_value
                else:
                    rd = linear_vel_value / angular_vel_value
                    linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                    angular_vel_value_limit = maxw * np.sign(angular_vel_value)
            else:
                if np.abs(angular_vel_value) <= 0.001:
                    linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                    angular_vel_value_limit = 0.0
                else:
                    rd = linear_vel_value / angular_vel_value
                    if np.abs(rd) >= maxv / maxw:
                        linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                        angular_vel_value_limit = maxv * np.sign(angular_vel_value) / np.abs(rd)
                    else:
                        linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                        angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        self.stats.add_timing("post_processing", timer.get_time())
        if torch.cuda.is_available():
            self.stats.add_vram("post_processing", 
                               timer.get_vram_allocated(), 
                               timer.get_vram_peak(), 
                               timer.get_vram_delta())

        return linear_vel_value_limit, angular_vel_value_limit

    # ----------------------------
    # Custom Collator
    # ----------------------------
    def collator_custom(self, instances, model_max_length, pad_token_id, padding_side="right", pixel_values_dtype=torch.float32):
        IGNORE_INDEX = -100
        input_ids = pad_sequence([inst["input_ids"] for inst in instances], batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([inst["labels"] for inst in instances], batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, :model_max_length], labels[:, :model_max_length]
        attention_mask = input_ids.ne(pad_token_id)

        pixel_values = [inst["pixel_values_current"] for inst in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [inst["dataset_name"] for inst in instances]
        else:
            dataset_names = None

        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_goal" in instances[0]:
                pixel_values_goal = [inst["pixel_values_goal"] for inst in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_goal)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type: {type(pixel_values)}")

        actions = torch.stack([torch.from_numpy(np.copy(inst["actions"])) for inst in instances])
        goal_pose = torch.stack([torch.from_numpy(np.copy(inst["goal_pose"])) for inst in instances])

        output = dict(
            pixel_values=pixel_values.to(),
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            goal_pose=goal_pose,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

    # ----------------------------
    # Transform Data to Dataset Format
    # ----------------------------
    def transform_datatype(self, inst_obj, actions, goal_pose_cos_sin,
                           current_image_PIL, goal_image_PIL, prompt_builder, action_tokenizer,
                           base_tokenizer, image_transform, predict_stop_token=True):
        IGNORE_INDEX = -100
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(action_tokenizer(future_actions))
        current_action_string = action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        if inst_obj == "xxxx":
            conversation = [
                {"from": "human", "value": "No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {inst_obj}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]

        prompt_builder = prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = torch.tensor(base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX
        if not predict_stop_token:
            labels[-1] = IGNORE_INDEX

        pixel_values_current = image_transform(current_image_PIL)
        pixel_values_goal = image_transform(goal_image_PIL)
        dataset_name = "lelan"

        return dict(
            pixel_values_current=pixel_values_current,
            pixel_values_goal=pixel_values_goal,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=torch.as_tensor(actions),
            goal_pose=goal_pose_cos_sin,
            img_PIL=current_image_PIL,
            inst=inst_obj,
        )

    # ----------------------------
    # Data Transformer for OmniVLA
    # ----------------------------
    def data_transformer_omnivla(self, current_image_PIL, lan_inst, goal_image_PIL, goal_pose_loc_norm,
                                 prompt_builder, action_tokenizer, processor):
        actions = np.random.rand(8, 4)  # dummy actions
        goal_pose_cos_sin = goal_pose_loc_norm

        batch_data = self.transform_datatype(
            lan_inst, actions, goal_pose_cos_sin,
            current_image_PIL, goal_image_PIL,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
        )

        batch = self.collator_custom(
            instances=[batch_data],
            model_max_length=processor.tokenizer.model_max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            padding_side="right"
        )
        return batch

    # ----------------------------
    # Run Forward Pass
    # ----------------------------
    def run_forward_pass(self, vla, action_head, noisy_action_projector, pose_projector,
                         batch, action_tokenizer, device_id, use_l1_regression, use_diffusion,
                         use_film, num_patches, compute_diffusion_l1=False,
                         num_diffusion_steps_train=None, mode="vali", idrun=0) -> Tuple[torch.Tensor, Dict[str, float]]:

        metrics = {}
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

        # Determine modality
        if self.satellite and not self.lan_prompt and not self.pose_goal and not self.image_goal:
            modality_id = torch.as_tensor([0], dtype=torch.float32)
        elif self.satellite and not self.lan_prompt and self.pose_goal and not self.image_goal:
            modality_id = torch.as_tensor([1], dtype=torch.float32)
        elif self.satellite and not self.lan_prompt and not self.pose_goal and self.image_goal:
            modality_id = torch.as_tensor([2], dtype=torch.float32)
        elif self.satellite and not self.lan_prompt and self.pose_goal and self.image_goal:
            modality_id = torch.as_tensor([3], dtype=torch.float32)
        elif not self.satellite and not self.lan_prompt and self.pose_goal and not self.image_goal:
            modality_id = torch.as_tensor([4], dtype=torch.float32)
        elif not self.satellite and not self.lan_prompt and self.pose_goal and self.image_goal:
            modality_id = torch.as_tensor([5], dtype=torch.float32)
        elif not self.satellite and not self.lan_prompt and not self.pose_goal and self.image_goal:
            modality_id = torch.as_tensor([6], dtype=torch.float32)
        elif not self.satellite and self.lan_prompt and not self.pose_goal and not self.image_goal:
            modality_id = torch.as_tensor([7], dtype=torch.float32)
        elif not self.satellite and self.lan_prompt and self.pose_goal and not self.image_goal:
            modality_id = torch.as_tensor([8], dtype=torch.float32)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                modality_id=modality_id.to(torch.bfloat16).to(device_id),
                labels=batch["labels"].to(device_id),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id),
                proprio_projector=pose_projector,
                noisy_actions=noisy_actions if use_diffusion else None,
                noisy_action_projector=noisy_action_projector if use_diffusion else None,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
                use_film=use_film,
            )

        # Prepare data for metrics
        ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
         
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        # Get hidden states for text portion of prompt+response (after the vision patches)
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )  # (B, act_chunk_len, D)

        with torch.no_grad():
            predicted_actions = action_head.predict_action(actions_hidden_states, modality_id.to(torch.bfloat16).to(device_id))                                 

        # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
        return predicted_actions, modality_id

                
# ===============================================================
# Inference Configuration
# ===============================================================
class InferenceConfig:
    resume: bool = True
    
    ## Original Weights
    # vla_path: str = "./omnivla-original" # omnivla-finetuned-cast
    # resume_step: Optional[int] = 120000  
    
    ## Fine-Tuned Weights
    vla_path: str = "./omnivla-finetuned-cast"    
    resume_step: Optional[int] = 210000
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

def define_model(cfg: InferenceConfig, stats: BenchmarkStats):
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Loading OpenVLA Model `{cfg.vla_path}`")

    # GPU setup
    device_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPOSE_DIM: {POSE_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Register OpenVLA model to HF Auto Classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
    
    # Load processor and VLA
    with Timer("model_loading", track_vram=torch.cuda.is_available()) as timer:
        processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device_id)
        
        vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
        vla.to(dtype=torch.bfloat16, device=device_id)
        
        pose_projector = init_module(
            ProprioProjector,
            "pose_projector",
            cfg,
            device_id,
            {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM},            
        )
        
        if cfg.use_l1_regression:
            action_head = init_module(
                L1RegressionActionHead_idcat,
                "action_head",
                cfg,
                device_id,
                {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM},            
                to_bf16=True,
            )
    stats.add_timing("model_loading", timer.get_time())
    if torch.cuda.is_available():
        stats.add_vram("model_loading", 
                       timer.get_vram_allocated(), 
                       timer.get_vram_peak(), 
                       timer.get_vram_delta())
 
    # Get number of vision patches
    NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()    
    NUM_PATCHES += 1 #for goal pose

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    return vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor

# ===============================================================
# Main Entry
# ===============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark OmniVLA inference speed")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations")
    parser.add_argument("--output", type=str, default="./inference/benchmark_results.json", help="Output JSON file path")
    parser.add_argument("--visualize", type=str, default="./inference/benchmark_visualization.png", help="Output visualization image path")
    args = parser.parse_args()
    
    # Initialize benchmark statistics
    stats = BenchmarkStats()
    
    # select modality
    pose_goal = True
    satellite = False
    image_goal = False
    lan_prompt = False

    # Goal definitions
    lan_inst_prompt = "move toward between brown box and blue ball"
    # Use hardcoded UTM values from run_omnivla-ARIL.py
    goal_utm = [-12.2266438, 12.6290138]
    goal_compass = 0.91930187
    goal_image_PIL = Image.open("./inference/frame211-56.597-ego.jpg").convert("RGB")

    # Define models (VLA, action_head, pose_projector, processor, etc.)
    print("="*80)
    print("LOADING MODELS")
    print("="*80)
    cfg = InferenceConfig()
    vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor = define_model(cfg, stats)

    # Run benchmark
    print("="*80)
    print("RUNNING BENCHMARK")
    print("="*80)
    inference = BenchmarkInference(
        save_dir="./inference",
        lan_inst_prompt=lan_inst_prompt,
        goal_utm=goal_utm,
        goal_compass=goal_compass,
        goal_image_PIL=goal_image_PIL,
        action_tokenizer=action_tokenizer,
        processor=processor,
        vla=vla,
        action_head=action_head,
        pose_projector=pose_projector,
        device_id=device_id,
        NUM_PATCHES=NUM_PATCHES,
        stats=stats,
    )
    
    # Measure total inference time
    with Timer("total_inference") as timer:
        inference.run_benchmark(num_iterations=args.num_iterations, warmup=args.warmup)
    stats.add_timing("total_inference", timer.get_time())
    
    # Print summary
    stats.print_summary()
    
    # Save results
    stats.save_to_json(args.output)
    
    # Create visualization
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION")
    print("="*80)
    stats.visualize(save_path=args.visualize)
    
    # Calculate and print throughput
    forward_pass_stats = stats.get_stats("forward_pass")
    if forward_pass_stats:
        fps = 1.0 / forward_pass_stats["mean"]
        print(f"\nThroughput: {fps:.2f} FPS (based on forward_pass mean time)")
    
    # Print VRAM summary
    if torch.cuda.is_available():
        print("\n" + "="*80)
        print("VRAM SUMMARY")
        print("="*80)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved_vram = torch.cuda.memory_reserved(0) / (1024**3)
        allocated_vram = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"Total VRAM:     {total_vram:.2f} GB")
        print(f"Reserved VRAM:   {reserved_vram:.2f} GB")
        print(f"Allocated VRAM:  {allocated_vram:.2f} GB")
        print("="*80)

