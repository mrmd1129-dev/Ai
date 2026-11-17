#!/usr/bin/env python3
"""CPU-only fallback (no GPU): small resolution, few frames, few steps.
Encodes the prompt first and frees the 11 GB text encoder before denoising so it fits in ~16 GB RAM.
Expect a rough, low-res draft that takes 1-3 hours on 4 cores."""
import argparse, asyncio, gc, subprocess, time
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
from diffusers.utils import export_to_video
from transformers import AutoTokenizer, UMT5EncoderModel

import generate as g

REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


@torch.no_grad()
def encode(texts):
    tok = AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(REPO, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    out = []
    for t in texts:
        ids = tok([t], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        n = int(ids.attention_mask.sum())
        e = enc(ids.input_ids, ids.attention_mask).last_hidden_state
        e[:, n:] = 0  # match WanPipeline: zero out padding positions
        out.append(e)
    del enc
    gc.collect()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--frames", type=int, default=41)  # 4k+1
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=800)
    p.add_argument("--delay-ms", type=int, default=400)
    p.add_argument("--outdir", default="output_cpu")
    p.add_argument("--encode-only", action="store_true",
                   help="encode the prompt, save embeds.pt and exit (lets you delete the text encoder before downloading the rest)")
    a = p.parse_args()
    torch.set_num_threads(torch.get_num_threads())
    out = g.HERE / a.outdir
    out.mkdir(exist_ok=True)

    t0 = time.time()
    cache = out / "embeds.pt"
    if cache.exists():
        pos, neg = torch.load(cache)
    else:
        pos, neg = encode([(g.HERE / "prompt.txt").read_text().strip(), g.NEGATIVE])
        torch.save((pos, neg), cache)
        print(f"encoded prompt in {time.time()-t0:.0f}s", flush=True)
    if a.encode_only:
        return

    vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(REPO, vae=vae, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)

    def tick(pipe, i, t, kw):
        print(f"step {i+1}/{a.steps}  {time.time()-t0:.0f}s", flush=True)
        return kw

    frames = pipe(
        prompt_embeds=pos, negative_prompt_embeds=neg,
        height=a.height, width=a.width, num_frames=a.frames,
        guidance_scale=5.0, num_inference_steps=a.steps,
        generator=torch.Generator("cpu").manual_seed(a.seed),
        callback_on_step_end=tick,
    ).frames[0]
    raw = out / "raw.mp4"
    export_to_video(frames, str(raw), fps=a.fps)
    print(f"video done in {time.time()-t0:.0f}s", flush=True)

    # Smooth 8 fps -> 24 fps and upscale for phones, then add voice + wind.
    smooth = out / "smooth.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-vf",
                    "minterpolate=fps=24:mi_mode=mci,scale=720:1296:flags=lanczos",
                    "-c:v", "libx264", "-crf", "18", str(smooth)], check=True)
    speech = out / "line.mp3"
    asyncio.run(g.make_speech(speech))
    g.mux(smooth, speech, out / "harun_balcony.mp4", a.delay_ms)
    print(f"Done: {out/'harun_balcony.mp4'}  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
