# Harun al-Rashid — palace balcony scene

Text-to-video of the caliph telling the passing cloud
**«أَمْطِرِي حَيْثُ شِئْتِ، فَسَيَأْتِينِي خَرَاجُكِ»** ("Rain wherever you wish — your land-tax will come to me"),
filmed as shaky vertical phone footage.

## Pipeline
1. **Video**: [Wan 2.1](https://github.com/Wan-Video/Wan2.1) via 🤗 Diffusers (installed from GitHub), 9:16, 81 frames @ 16 fps (~5 s).
2. **Voice**: `edge-tts`, voice `ar-SA-HamedNeural`.
3. **Mux**: ffmpeg overlays the line at 1.5 s on synthetic wind noise, re-encodes at CRF 28 for a phone-compressed look.

## Run
**Google Colab (easiest):** open `colab.ipynb`, set runtime to GPU, run all cells.

**Your own GPU machine:**
```bash
pip install -r requirements.txt     # also needs ffmpeg on PATH
python generate.py --model 1.3b     # ~10 GB VRAM; add --offload for 8 GB cards
python generate.py --model 14b      # 24 GB+ VRAM, 720p, much better faces
```
Output: `output/harun_balcony.mp4` (plus `raw.mp4` and `line.mp3`).

Options: `--seed` (try several), `--steps`, `--delay-ms` (when the line starts).

## Limits
- Wan does not generate speech or lip movement; the voice is overlaid. For lip-sync, run the
  result through [LatentSync](https://github.com/bytedance/LatentSync) or
  [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) with `output/line.mp3`.
- Clips are ~5 s; the 1.3B model often struggles with faces and hands. Generate several seeds.
