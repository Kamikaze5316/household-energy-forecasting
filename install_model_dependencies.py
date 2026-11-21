from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METADATA_PATH = ROOT / "artifacts" / "metadata.json"

PACKAGE_MAP = {
    "numpy": "numpy",
    "scikit_learn": "scikit-learn",
    "xgboost": "xgboost",
    "scipy": "scipy",
    "joblib": "joblib",
}


def main() -> int:
    if not METADATA_PATH.exists():
        print("[INFO] artifacts/metadata.json не найден.")
        print("       Точные версии из Google Colab пока недоступны; оставляем версии из requirements.txt.")
        return 0

    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Не удалось прочитать {METADATA_PATH}: {exc}")
        return 1

    environment = metadata.get("environment", {})
    if not environment:
        print("[WARNING] В metadata.json нет блока environment.")
        print("          Повторно выполните обновлённый Colab Notebook и замените папку artifacts.")
        return 0

    python_trained = str(environment.get("python", ""))
    python_local = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if python_trained and tuple(python_trained.split(".")[:2]) != tuple(python_local.split(".")[:2]):
        print("[WARNING] Версия Python отличается от среды обучения:")
        print(f"          Colab: {python_trained}; локально: {python_local}")
        print("          Для наилучшей совместимости используйте ту же основную/дополнительную версию Python.")

    requirements = []
    for metadata_key, package_name in PACKAGE_MAP.items():
        version = environment.get(metadata_key)
        if version:
            requirements.append(f"{package_name}=={version}")

    if not requirements:
        print("[WARNING] В метаданных нет подходящих версий пакетов.")
        return 0

    print("Устанавливаю версии библиотек, использованные при обучении модели:")
    for requirement in requirements:
        print(" -", requirement)

    command = [sys.executable, "-m", "pip", "install", "--upgrade", *requirements]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
