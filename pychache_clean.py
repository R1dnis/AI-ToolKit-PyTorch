import os
import shutil
import time


def clean_pycache(start_path="."):
    """
    Рекурсивно ищет и удаляет все папки __pycache__
    в директории, где запущен скрипт.
    """
    print(f"Поиск и удаление папок __pycache__ в: {start_path}")
    deleted_count = 0
    # --- (Ruff B007) files -> _files ---  <-- УДАЛЕНО
    for root, dirs, _files in os.walk(start_path):
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                print(f"Удалено: {pycache_path}")
                deleted_count += 1
            except OSError as e:
                print(f"Ошибка удаления {pycache_path}: {e}")

    print(f"\nЗавершено. Удалено {deleted_count} папок __pycache__.")


if __name__ == "__main__":
    start_time = time.time()
    # Запускаем из папки, где лежит скрипт (ai_toolkit/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_pycache(script_dir)
    end_time = time.time()
    print(f"Время выполнения: {end_time - start_time:.2f} сек.")
