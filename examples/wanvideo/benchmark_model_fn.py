import torch
import numpy as np
import argparse
import os
from contextlib import nullcontext
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, model_fn_wan_video
from torch.profiler import profile, ProfilerActivity


def print_rank_0(rank, message):
    """Print message only if rank is 0"""
    if rank == 0:
        print(message)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Benchmark WAN Video model_fn latency')
    parser.add_argument('--warmup_iters', type=int, default=3,
                        help='Number of warmup iterations (default: 3)')
    parser.add_argument('--benchmark_iters', type=int, default=10,
                        help='Number of benchmark iterations (default: 10)')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling with stack traces')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile (cannot be used with CUDA graphs)')
    args = parser.parse_args()

    WARMUP_ITERS = args.warmup_iters
    BENCHMARK_ITERS = args.benchmark_iters
    ENABLE_PROFILE = args.profile
    USE_COMPILE = args.compile


    # Get process rank early
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', -1)))
    if rank == -1:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = torch.cuda.current_device() if torch.cuda.is_available() else 0
        except:
            rank = torch.cuda.current_device() if torch.cuda.is_available() else 0

    # Initialize pipeline
    print_rank_0(rank, "Initializing pipeline...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        use_usp=True,  # Using AITER for ROCm/AMD GPU support
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    pipe.enable_vram_management()

    # Prepare inputs - we'll do preprocessing once and reuse for benchmarking
    prompt = "a boat floating in the middle of ocean, a whale passes by slowly."
    cfg_merge = False # false to benchmark only batch size = 1
    seed = 0
    tiled = True
    height = 480
    width = 832
    num_frames = 81
    cfg_scale = 5.0
    num_inference_steps = 50
    sigma_shift = 5.0

    # Prepare inputs through the pipeline units
    inputs_posi = {
        "prompt": prompt,
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": num_inference_steps,
    }
    inputs_nega = {
        "negative_prompt": "",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": num_inference_steps,
    }
    inputs_shared = {
        "input_image": None,
        "end_image": None,
        "input_video": None,
        "denoising_strength": 1.0,
        "control_video": None,
        "reference_image": None,
        "camera_control_direction": None,
        "camera_control_speed": 1/54,
        "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        "vace_video": None,
        "vace_video_mask": None,
        "vace_reference_image": None,
        "vace_scale": 1.0,
        "seed": seed,
        "rand_device": "cpu",
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "cfg_scale": cfg_scale,
        "cfg_merge": cfg_merge,
        "sigma_shift": sigma_shift,
        "motion_bucket_id": None,
        "tiled": tiled,
        "tile_size": (30, 52),
        "tile_stride": (15, 26),
        "sliding_window_size": None,
        "sliding_window_stride": None,
        "input_audio": None,
        "audio_sample_rate": 16000,
        "s2v_pose_video": None,
        "audio_embeds": None,
        "s2v_pose_latents": None,
        "motion_video": None,
        "animate_pose_video": None,
        "animate_face_video": None,
        "animate_inpaint_video": None,
        "animate_mask_video": None,
    }

    print_rank_0(rank, "Preprocessing inputs through pipeline units...")
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    # Load models
    print_rank_0(rank, "Loading models to device...")
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    # Create a timestep tensor directly (no need for scheduler.set_timesteps)
    # Use a value around 0.5 which represents middle of the diffusion process
    timestep = torch.tensor([0.5], dtype=pipe.torch_dtype, device=pipe.device)

    # Prepare model_fn arguments
    model_fn_kwargs = {
        **models,
        **inputs_shared,
        **inputs_posi,
        "timestep": timestep,
    }

    # Calculate expected sequence length for validation
    latents_shape = inputs_shared['latents'].shape

    print_rank_0(rank, f"\nStarting benchmark with {WARMUP_ITERS} warmup iterations and {BENCHMARK_ITERS} benchmark iterations...")
    print_rank_0(rank, f"Input latents shape: {latents_shape}")
    print_rank_0(rank, f"Timestep: {timestep.item():.4f}")
    if ENABLE_PROFILE:
        print_rank_0(rank, "Profiling enabled: will capture trace with stack")

    # Warmup iterations
    print_rank_0(rank, "\nWarmup phase...")
    for i in range(WARMUP_ITERS):
        with torch.no_grad():
            _ = model_fn_wan_video(**model_fn_kwargs)
        torch.cuda.synchronize()
        print_rank_0(rank, f"  Warmup iteration {i+1}/{WARMUP_ITERS} completed")

    # Apply torch.compile if requested
    if USE_COMPILE:
        print_rank_0(rank, "\nCompiling model_fn with torch.compile...")
        compiled_fn = torch.compile(model_fn_wan_video, mode="max-autotune")
        print_rank_0(rank, "Compilation complete. Running additional warmup for compiled function...")
        # Additional warmup for compiled function
        for i in range(3):
            with torch.no_grad():
                _ = compiled_fn(**model_fn_kwargs)
            torch.cuda.synchronize()
        model_fn = compiled_fn
    else:
        model_fn = model_fn_wan_video

    # Benchmark iterations
    print_rank_0(rank, "\nBenchmark phase...")
    latencies = []

    # Create CUDA events once for reuse
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Setup profiler context if enabled
    if ENABLE_PROFILE:
        trace_dir = "trace_benchmark"
        os.makedirs(trace_dir, exist_ok=True)

        trace_path = os.path.join(trace_dir, f"trace_benchmark_rank_{rank}.json")
        profiler_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False,
            with_stack=True,
        )
    else:
        profiler_context = nullcontext()

    # Run benchmark with or without profiling
    with profiler_context as prof:
        for i in range(BENCHMARK_ITERS):
            start_event.record()

            # Run normally (eager or compiled)
            with torch.no_grad():
                _ = model_fn(**model_fn_kwargs)

            end_event.record()

            torch.cuda.synchronize()
            iteration_latency = start_event.elapsed_time(end_event)  # Already in milliseconds
            latencies.append(iteration_latency)
            print_rank_0(rank, f"  Iteration {i+1}/{BENCHMARK_ITERS}: {iteration_latency:.2f} ms")

    # Save profiling trace if enabled
    if ENABLE_PROFILE and rank == 0:
        try:
            prof.export_chrome_trace(trace_path)
            print_rank_0(rank, f"\nProfile trace saved to {trace_path}")
        except Exception as e:
            print_rank_0(rank, f"\nWarning: Failed to save trace to {trace_path}: {e}")

    # Calculate statistics
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    median_latency = np.median(latencies)

    # Print results (only rank 0)
    print_rank_0(rank, "\n" + "="*60)
    print_rank_0(rank, "BENCHMARK RESULTS")
    print_rank_0(rank, "="*60)
    print_rank_0(rank, f"Number of iterations: {BENCHMARK_ITERS}")
    print_rank_0(rank, f"Mean latency:         {mean_latency:.2f} ms")
    print_rank_0(rank, f"Std deviation:        {std_latency:.2f} ms")
    print_rank_0(rank, f"Median latency:       {median_latency:.2f} ms")
    print_rank_0(rank, f"Min latency:          {min_latency:.2f} ms")
    print_rank_0(rank, f"Max latency:          {max_latency:.2f} ms")
    print_rank_0(rank, "="*60)

    # Save results to file (only rank 0)
    if rank == 0:
        with open("benchmark_results.txt", "w") as f:
            f.write("WAN Video Model Function Benchmark Results\n")
            f.write("="*60 + "\n")
            f.write(f"Configuration:\n")
            f.write(f"  - Height: {height}\n")
            f.write(f"  - Width: {width}\n")
            f.write(f"  - Num frames: {num_frames}\n")
            f.write(f"  - Warmup iterations: {WARMUP_ITERS}\n")
            f.write(f"  - Benchmark iterations: {BENCHMARK_ITERS}\n")
            f.write(f"  - Profiling enabled: {ENABLE_PROFILE}\n")
            f.write(f"  - Input shape: {inputs_shared['latents'].shape}\n")
            f.write("\n")
            f.write(f"Results:\n")
            f.write(f"  - Mean latency:    {mean_latency:.2f} ms\n")
            f.write(f"  - Std deviation:   {std_latency:.2f} ms\n")
            f.write(f"  - Median latency:  {median_latency:.2f} ms\n")
            f.write(f"  - Min latency:     {min_latency:.2f} ms\n")
            f.write(f"  - Max latency:     {max_latency:.2f} ms\n")
            f.write("\n")
            f.write(f"Individual iteration latencies (ms):\n")
            for i, lat in enumerate(latencies):
                f.write(f"  Iteration {i+1}: {lat:.2f}\n")

        print_rank_0(rank, f"\nResults saved to benchmark_results.txt")


if __name__ == "__main__":
    main()
