# LangSplat OS

This repository contains the LangSplat project, an enhanced framework for 3D Gaussian Splatting with semantic scene understanding capabilities.

## Features

- **Training Monitor Platform**: Built-in `TrainingMonitor` for real-time tracking of GPU usage, Gaussian point counts, iteration speeds, and refined loss metrics.
- **Data Strategy**: Advanced preprocessing tools including cropping, smooth transitions, and custom image integration.
- **Frontend Interface**: An intuitive interface for running models, visualizing training state, and interacting with semantic queries.

## Structure

- `frontend/` - React-based user interface for monitoring and interacting with the system.
- `backend/` - Server-side scripts supporting the platform.
- `LangSplat/` - Core LangSplat framework codebase.
- `lerf_ovs/` - Integration of Language Embedded Radiance Fields.
