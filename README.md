# Flappy Bird Lateral Raises

Control Flappy Bird by doing lateral raises in front of your webcam!

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Pygame](https://img.shields.io/badge/Pygame-CE-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)

## How It Works

Uses MediaPipe pose detection to track your arms. When you raise either arm (lateral raise), the bird flaps

## Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/big-flappy-bird.git
cd big-flappy-bird

# Install dependencies with uv
uv sync

# Run the game
uv run python flappy.py
```

## Controls

- **Lateral Raise** - Raise either arm to flap
- **ESC** - Quit

## Requirements

- Python 3.12+
- Webcam
- [uv](https://github.com/astral-sh/uv) or pip

## Credits

- Sprites from [samuelcust/flappy-bird-assets](https://github.com/samuelcust/flappy-bird-assets)
- Pose detection by [MediaPipe](https://mediapipe.dev/)
