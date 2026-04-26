# Genie Dynamics Model Implementation

This project implements a Dynamics Model based on Genie Envisioner (GE-base) principles, with Task-centric Batching strategy to improve action control.

## Overview

- **Dynamics Model**: A neural network that predicts next state from current state and action.
- **Task-centric Batching**: Batches data by task to ensure the model learns action dependencies better, improving controllability.

## Files

- `requirements.txt`: Dependencies
- `dynamics_model.py`: Neural network model
- `dataset.py`: Custom dataset and sampler for task-centric batching
- `main.py`: Training script

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the training:

```bash
python main.py
```

## Notes

This is a simplified implementation. For full Genie Envisioner integration, refer to the official repo: https://github.com/AgibotTech/Genie-Envisioner