# CGN: Connectome Graph Networks

## Project Goals

This project uses Graph Neural Networks (GNNs) and Transformers to analyze the ABIDE neuroimaging dataset for Autism Spectrum Disorder (ASD) research. It leverages **PyTorch Geometric** for graph processing, **PyTorch Lightning** for training loops, and **Hydra** for configuration management.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/musickevan1/CGN.git
    cd CGN
    ```

2.  **Install the package:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -e .
    ```

## Usage

### Training

The main entry point for training is `src/train.py`. The project uses Hydra for configuration.

**Run with default configuration:**
```bash
python src/train.py
```

**Override parameters:**
```bash
# Example: Change max epochs and batch size
python src/train.py trainer.max_epochs=50 data.batch_size=64
```

### Testing

Run the test suite using `pytest`:
```bash
pytest
```

## Directory Structure

*   `src/cgn/`: Main package source code.
    *   `data/`: DataModules (e.g., ABIDE, Synthetic).
    *   `models/`: Model architectures (GNN, Transformers).
    *   `system.py`: PyTorch Lightning System (training logic).
*   `configs/`: Hydra configuration files (`train.yaml`, `model/`, `data/`, `trainer/`).
*   `tests/`: Unit and integration tests.
*   `src/train.py`: Training script entry point.

## Configuration

Configurations are stored in `configs/`. You can modify `configs/train.yaml` or creating new config files in the respective subdirectories.

## License

MIT License
