# Flappy Bird Lateral Raises

Turn your workout into a high score! This project reimagines the classic Flappy Bird experience by replacing screen taps with physical **lateral raises**. Using your webcam and real-time motion tracking, your body becomes the controller.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Pygame](https://img.shields.io/badge/Pygame-CE-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)

## How It Works

The game utilizes **MediaPipe Pose Landmarking** to monitor your body via webcam. It specifically tracks the relationship between your torso and your upper arms. 

* **The Trigger:** When the system detects your arms moving from your sides toward a horizontal position (a lateral raise), the bird flaps.
* **The Benefit:** Train your lateral deltoids while navigating through endless green pipes.

### Prerequisites
* **Python 3.12+**
* A functional **Webcam**
* **[uv](https://github.com/astral-sh/uv)** (recommended) or `pip`


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
