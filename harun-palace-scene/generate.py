#!/usr/bin/env python3
"""Generate the Harun al-Rashid balcony scene with Wan 2.1 (text-to-video),
then add the spoken Arabic line and wind ambience.

Needs an NVIDIA GPU. Run:  python generate.py --model 1.3b   (about 10 GB VRAM)
                           python generate.py --model 14b    (24 GB+ VRAM, better quality)
"""
import argparse
import asyncio
import subprocess
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
from diffusers.utils import export_to_video

HERE = Path(__file__).parent
LINE = "أَمْطِرِي حَيْثُ شِئْتِ، فَسَيَأْتِينِي خَرَاجُكِ"
NEGATIVE = (
    "cinematic color grading, film look, steady tripod shot, gimbal, smooth dolly, "
    "cartoon, CGI, painting, illustration, modern clothing, modern buildings, "
    "text, subtitles, watermark, logo, deformed hands, extra fingers, blurry face, "
    "static image, low quality"
)
MODELS = {
    # name: (repo, height, width, flow_shift)  -- vertical 9:16 framing
    "1.3b": ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", 832, 480, 3.0),
    "14b": ("Wan-AI/Wan2.1-T2V-14B-Diffusers", 1280, 720, 5.0),
}


def make_video(args, out: Path):
    repo, height, width, shift = MODELS[args.model]
    vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(repo, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=shift)
    if args.offload:
        pipe.enable_model_cpu_offload()  # lower VRAM, slower
    else:
        pipe.to("cuda")

    frames = pipe(
        prompt=(HERE / "prompt.txt").read_text().strip(),
        negative_prompt=NEGATIVE,
        height=height,
        width=width,
        num_frames=81,  # ~5 s at 16 fps; must be 4k+1
        guidance_scale=5.0,
        num_inference_steps=args.steps,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).frames[0]
    export_to_video(frames, str(out), fps=16)


async def make_speech(out: Path):
    import edge_tts

    # Deep, calm male voice; slightly slower for gravitas.
    await edge_tts.Communicate(LINE, "ar-SA-HamedNeural", rate="-10%").save(str(out))


def mux(video: Path, speech: Path, out: Path, delay_ms: int):
    """Overlay the line (delayed to land on the raised hand) on brown-noise wind."""
    filt = (
        f"[1:a]adelay={delay_ms}|{delay_ms},volume=1.6[v];"
        "anoisesrc=color=brown:amplitude=0.12:sample_rate=24000,lowpass=f=600,"
        "tremolo=f=0.3:d=0.6[wind];"
        "[v][wind]amix=inputs=2:duration=first:normalize=0,apad[a]"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-i", str(speech), "-filter_complex", filt,
         "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-crf", "28",  # crf 28 = phone-ish compression
         "-c:a", "aac", "-shortest", str(out)],
        check=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default="1.3b")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=800)
    p.add_argument("--offload", action="store_true", help="CPU offload for small GPUs")
    p.add_argument("--delay-ms", type=int, default=1500, help="when the line starts")
    p.add_argument("--outdir", default="output")
    args = p.parse_args()

    outdir = HERE / args.outdir
    outdir.mkdir(exist_ok=True)
    raw, speech, final = outdir / "raw.mp4", outdir / "line.mp3", outdir / "harun_balcony.mp4"

    make_video(args, raw)
    asyncio.run(make_speech(speech))
    mux(raw, speech, final, args.delay_ms)
    print(f"Done: {final}")


if __name__ == "__main__":
    main()
