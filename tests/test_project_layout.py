from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_runtime_files_exist():
    required = [
        ROOT / "app" / "streamlit_app.py",
        ROOT / "models" / "model.npz",
        ROOT / "models" / "model_meta.json",
        ROOT / "requirements.txt",
        ROOT / ".streamlit" / "config.toml",
    ]
    assert all(path.exists() for path in required)
