# 5G-NTN Co-Channel Interference Simulator

Python simulation and Streamlit dashboard for studying aggregate co-channel
interference in 5G non-terrestrial network direct-to-device links.

## Live App

The interactive dashboard is deployed on Streamlit Community Cloud:

https://5g-ntn-cochannel-interference-8uvnzwsjmmeprnj3zj6hxj.streamlit.app/

## Contents

- `src/ntn_interference/`: simulation source code.
- `configs/`: scenario configuration files.
- `dashboard/`: Streamlit dashboard.
- `scripts/`: helper scripts for validation and report figures.
- `results/`: generated CSV results and figures.
- `tests/`: basic unit tests.

## How to run

From inside the repository folder:

```bash
pip install -r requirements.txt
python -m ntn_interference.simulation --config configs/baseline.yaml
python -m ntn_interference.simulation --config configs/baseline.yaml --mitigation configs/mitigation.yaml
python -m ntn_interference.plots --results results/data/
```

To launch the interactive dashboard:

```bash
python -m streamlit run dashboard/streamlit_app.py
```

To run checks:

```bash
pytest
python scripts/validate_scenarios.py
```

Requires Python 3.10 or newer.
