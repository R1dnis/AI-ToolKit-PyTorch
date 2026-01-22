import json
import os
import queue
import sys
import threading
import tkinter
from queue import Empty as QueueEmpty
from tkinter import messagebox

import ttkbootstrap as ttk

# --- Установка переменных окружения (должна быть ДО импорта библиотек) ---
os.environ["ALBUMENTATIONS_SUPPRESS_UPDATE_CHECK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from class_database import ClassManager
from constants import CLASS_SETS_FILE, SETTINGS_FILE

# --- Конфигурация системного пути ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Глобальные заглушки для type hinting
AnalysisTab = None
LabelingTab = None
ModelManagementTab = None


class MainWindow(ttk.Window):
    """
    Главное окно приложения. Инициализирует все вкладки и управляет основными настройками.
    """

    def __init__(self):
        super().__init__(themename="superhero")

        # Прячем окно на время инициализации
        self.withdraw()
        self.title("Green Synapse - AI Toolkit")

        try:
            self.state("zoomed")
        except tkinter.TclError:
            try:
                self.attributes("-zoomed", True)
            except tkinter.TclError:
                self.geometry("1920x1080")

        self.settings = {}
        self.class_manager = None
        self.main_layout = None
        self.splash_frame = None

        # Очередь для получения результата из фонового потока
        self.load_queue = queue.Queue()

        self.load_settings()
        self._create_splash_screen()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Показываем окно (со сплэш-скрином) немедленно
        self.deiconify()

        # Запускаем тяжелую загрузку в фоновом потоке
        self.after(100, self.start_heavy_load_thread)

    def _create_splash_screen(self):
        """Создает простой экран-заглушку на время загрузки."""
        self.splash_frame = ttk.Frame(self, padding=20)
        self.splash_frame.pack(fill="both", expand=True)

        splash_label = ttk.Label(
            self.splash_frame,
            text="Загрузка компонентов AI...",
            font=("Segoe UI", 16, "bold"),
        )
        splash_label.pack(pady=20, anchor="center")

        self.progress_bar = ttk.Progressbar(
            self.splash_frame, mode="indeterminate", length=300
        )
        self.progress_bar.pack(pady=10, anchor="center")

        # Анимация будет идти (возможно, "рвано" из-за GIL, но не остановится)
        self.progress_bar.start(15)

        info_label = ttk.Label(
            self.splash_frame,
            text="(Это может занять до 60 секунд на слабых системах)",
            font=("Segoe UI", 10),
        )
        info_label.pack(pady=5, anchor="center")

        self.update_idletasks()

    def start_heavy_load_thread(self):
        """Запускает фоновый поток для выполнения "тяжелых" импортов."""
        print("INFO: Запуск фонового потока (threading) для загрузки...")

        loader_thread = threading.Thread(target=self._heavy_import_task, daemon=True)
        loader_thread.start()

        # Начинаем проверять очередь на результат
        self.after(100, self.check_load_thread)

    def _heavy_import_task(self):
        """
        Выполняется в фоновом потоке (threading).
        Импорты здесь наполняют sys.modules *всего процесса*.
        """
        # Глобальные переменные нужны, чтобы модули вкладок
        # были импортированы и готовы для главного потока
        global AnalysisTab, LabelingTab, ModelManagementTab

        try:
            print("INFO: (Thread) Загрузка torch, matplotlib, cv2...")
            import cv2
            import matplotlib
            import torch

            print("INFO: (Thread) Тяжелые зависимости успешно загружены.")

            from tabs.analysis_tab import AnalysisTab
            from tabs.labeling_tab import LabelingTab
            from tabs.model_management_tab import ModelManagementTab

            print("INFO: (Thread) Загружены модули вкладок.")

            print("INFO: (Thread) Загрузка ClassManager...")
            class_manager = ClassManager()
            class_manager.load_from_json()
            print("INFO: (Thread) ClassManager загружен.")

            self.load_queue.put(class_manager)

        except (ImportError, OSError, ValueError, Exception) as e:
            print(f"CRITICAL: (Thread) Ошибка при загрузке: {e}")
            self.load_queue.put(e)

    def check_load_thread(self):
        """
        Выполняется в главном потоке (GUI).
        Проверяет очередь.
        """
        try:
            result = self.load_queue.get_nowait()

            # ВАЖНО: НЕ останавливаем анимацию здесь

            # Обрабатываем результат
            if isinstance(result, Exception):
                self.progress_bar.stop()  # Останавливаем только при ошибке
                self._show_load_error(result)
            else:
                # Передаем результат в _finish_ui_load
                self._finish_ui_load(result)

        except QueueEmpty:
            # Очередь пуста, компоненты еще грузятся.
            # Проверяем снова через 100 мс.
            self.after(100, self.check_load_thread)

    def _show_load_error(self, e):
        """Вызывается в главном потоке, если загрузка не удалась."""
        if isinstance(e, ImportError):
            messagebox.showerror(
                "Критическая ошибка: Отсутствуют зависимости",
                f"Не удалось загрузить одну из основных библиотек:\n\n{e}\n\n"
                "Пожалуйста, установите все зависимости из 'requirements.txt' и 'requirements/README.md' перед запуском.",
            )
        else:
            messagebox.showerror(
                "Критическая ошибка загрузки",
                f"Не удалось загрузить или прочитать файл {CLASS_SETS_FILE}:\n\n{e}\n\n"
                "Приложение будет использовать пустую базу классов. "
                "Если у вас был файл, проверьте его или удалите, чтобы создать новый.",
            )
        self.destroy()
        sys.exit(1)

    def _finish_ui_load(self, class_manager_result):
        """
        Вызывается в главном потоке.
        Выполняет *только* создание виджетов (быстро).
        """
        print("INFO: (Main) Загрузка завершена. Инициализация UI...")

        self.class_manager = class_manager_result

        # Это единственная оставшаяся блокирующая операция (1-2 сек)
        # Анимация в этот момент может "замереть", но это неизбежно
        # при создании виджетов в Tkinter.
        self._init_ui()

        # ВАЖНО: Останавливаем анимацию ПОСЛЕ инициализации UI
        self.progress_bar.stop()

        self.splash_frame.pack_forget()
        self.main_layout.pack(fill="both", expand=True)
        print("INFO: (Main) UI готов.")

    def _init_ui(self):
        """
        Инициализирует пользовательский интерфейс.
        Все модули (AnalysisTab и т.д.) уже загружены в память.
        """
        self.main_layout = ttk.Frame(self, padding=10)

        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=(10, 5), font=("Segoe UI", 10))

        self.tabs = ttk.Notebook(self.main_layout, style="TNotebook")

        # --- Инициализация и связывание вкладок (Внедрение зависимостей) ---
        # Модули AnalysisTab, ModelManagementTab, LabelingTab УЖЕ в sys.modules

        self.analysis_tab = AnalysisTab(self.tabs, class_manager=self.class_manager)
        self.model_tab = ModelManagementTab(
            self.tabs,
            on_model_trained_callback=self.analysis_tab.set_model_from_checkpoint,
        )
        self.labeling_tab = LabelingTab(
            self.tabs,
            class_manager=self.class_manager,
            on_dataset_created_callback=self.model_tab.set_dataset_path,
        )

        self.tabs.add(self.labeling_tab, text="1. Создание датасета")
        self.tabs.add(self.model_tab, text="2. Обучение модели")
        self.tabs.add(self.analysis_tab, text="3. Анализ и дообучение")
        self.tabs.pack(fill="both", expand=True, padx=5, pady=5)

        # --- Статус-бар ---
        status_frame = ttk.Frame(self.main_layout, padding=(5, 2))
        status_frame.pack(side="bottom", fill="x")
        self.status_label = ttk.Label(status_frame, text="Готов к работе.")
        self.status_label.pack(side="left", fill="x")

        annotations_path = self.settings.get("annotations_save_path", "")
        if annotations_path and os.path.isdir(annotations_path):
            self.labeling_tab.set_annotations_save_path(
                annotations_path, emit_signal=True
            )

    def load_settings(self):
        """
        Загружает настройки из файла settings.json.
        """
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, encoding="utf-8") as f:
                    self.settings = json.load(f)
            except json.JSONDecodeError as e:
                messagebox.showwarning(
                    "Поврежден файл настроек",
                    f"Файл '{SETTINGS_FILE}' поврежден:\n\n{e}\n\n"
                    "Приложение будет использовать настройки по умолчанию.",
                )
                self.settings = {}
            except OSError as e:
                messagebox.showerror(
                    "Ошибка чтения настроек",
                    f"Не удалось прочитать '{SETTINGS_FILE}':\n\n{e}\n\n"
                    "Приложение будет использовать настройки по умолчанию.",
                )
                self.settings = {}
        self.settings.setdefault("annotations_save_path", None)

    def save_settings(self):
        """
        Сохраняет текущие настройки в файл settings.json.
        """
        if hasattr(self, "labeling_tab") and self.labeling_tab:
            self.settings[
                "annotations_save_path"
            ] = self.labeling_tab.annotations_save_path

        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
        except OSError as e:
            messagebox.showerror(
                "Ошибка сохранения настроек",
                f"Не удалось сохранить настройки в '{SETTINGS_FILE}':\n\n{e}",
            )

    def on_closing(self):
        """
        Выполняет действия перед закрытием приложения (сохранение данных).
        """
        try:
            self.save_settings()
        except Exception as e:
            print(f"Ошибка при вызове save_settings(): {e}")

        try:
            if self.class_manager and self.class_manager.is_dirty:
                print("INFO: Обнаружены изменения в ClassManager, сохранение...")
                self.class_manager.save_to_json()
            elif self.class_manager:
                print("INFO: Изменений в ClassManager не обнаружено, пропуск сохранения.")
            else:
                print("WARNING: ClassManager не инициализирован, сохранение пропущено.")
        except (OSError, Exception) as e:
            messagebox.showerror(
                "Ошибка сохранения классов",
                f"Не удалось сохранить '{CLASS_SETS_FILE}':\n\n{e}",
            )

        self.destroy()


if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()