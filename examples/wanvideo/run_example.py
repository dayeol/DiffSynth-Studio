import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from torch.profiler import profile, ProfilerActivity

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    use_usp=True,
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()



video = pipe(
    cfg_merge=True,
    prompt="a boat floating in the middle of ocean, a whale passes by slowly.",
    seed=0,
    tiled=True,
)

print(f"Profiling trace saved to trace.json")

save_video(video, "video1.mp4", fps=15, quality=5)
