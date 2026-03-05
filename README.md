# SurgeryAudit Training Data Pipeline

Builds a labeled dataset of real vs AI-manipulated surgery before/after images.

## Quick Start

```bash
cd surgery-audit/data-pipeline
pip install -r requirements.txt

# 1. Set up Reddit API credentials
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"

# 2. Scrape real surgery photos from Reddit
python scrape_reddit.py --all --limit 500 --output-dir ./data

# 3. Generate synthetic manipulated versions
python generate_synthetic.py --input-dir ./data/real --output-dir ./data/synthetic

# 4. Build balanced train/val/test dataset
python build_dataset.py --data-dir ./data --output-dir ./dataset
```

## Data Sources

### Real Images (authentic class)
- **Reddit**: r/PlasticSurgery, r/cosmeticsurgery, r/Rhinoplasty, r/jawsurgery, r/PlasticSurgeryBeforeAfter
- **Academic**: FFHQ (70K real faces) — download manually from https://github.com/NVlabs/ffhq-dataset
  - Place in `./data/academic/real/`

### Manipulated Images (fake class)
- **Synthetic**: Programmatic manipulations applied to real images (beauty filter, skin smoothing, face reshape, eye enhancement, teeth whitening, AI upscale artifacts, color grading, composites)
- **Reddit**: r/InstagramReality, r/Instagramvsreality (edited/filtered photos)
- **Academic**: SFHQ (425K StyleGAN2 faces) — download from https://github.com/SelfishGene/SFHQ-dataset
  - Place in `./data/academic/fake/`

### NOT used (legal/ethical reasons)
- **RealSelf.com** — ToS prohibits scraping; HIPAA concerns
- Patient photos from surgical practices — requires IRB approval

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `scrape_reddit.py` | Download images from Reddit via PRAW API |
| `generate_synthetic.py` | Apply manipulations to create fake images |
| `build_dataset.py` | Combine sources into balanced train/val/test split |

## Manipulation Types

| Type | Description | Primary Detector |
|------|-------------|-----------------|
| beauty_filter | Bilateral smoothing + slight brightness boost | texture |
| skin_smoothing | Aggressive YCrCb-masked Gaussian + bilateral | texture, noise |
| face_reshape | Grid displacement field (nose slim, jaw narrow) | claude_vision |
| eye_enhancement | Eye region brightening + sharpening | claude_vision |
| teeth_whitening | HSV desaturation + brightening of mouth region | texture |
| ai_upscale | Downsample + cubic upscale + oversharpen | frequency |
| color_grading | Instagram-style warm tint + lifted shadows | texture |
| composite | Random 2-4 of the above combined | multiple |
| double_jpeg | Compress at Q40 then Q92 | ela |

## Output Structure

```
dataset/
  train/
    real/         # 70% of real images
    manipulated/  # 70% of manipulated images
  val/
    real/         # 15% of real images
    manipulated/  # 15% of manipulated images
  test/
    real/         # 15% of real images
    manipulated/  # 15% of manipulated images
  manifest.csv    # Full provenance for every image
  stats.json      # Dataset statistics
```

## Reddit API Setup

1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app"
3. Select "script" type
4. Use `http://localhost:8080` as redirect URI
5. Note your client_id (under app name) and client_secret
6. Set as environment variables

## Scaling Up

For a production classifier, target ~50K images per class:
- Reddit scraping: ~5-20K real surgery images across all subs
- Synthetic generation: apply all 9 manipulation types = 9x multiplier on real images
- Academic datasets fill the gap to 50K
