import contextlib
import datetime
import json
import os
import queue
import shutil
import threading
import tkinter as tk
import traceback
import uuid
from tkinter import END, Listbox, Toplevel, filedialog, messagebox
from typing import TYPE_CHECKING, Any

import ttkbootstrap as ttk
import yaml  # Оставляем, т.к. используется в save_feedback_data
from ttkbootstrap.constants import DANGER, LEFT, PRIMARY, SECONDARY

from class_database import ClassManager
from constants import (
    BEST_MODEL_FILE,
    CONFIG_FILE,
    DATA_YAML_FILE,
    FEEDBACK_DIR,
    IMAGES_DIR,
    LABELS_DIR,
    SUPPORTED_IMAGE_FORMATS,
)
from core.config import load_config
from tooltip import ToolTip
from ui_components import PhotoViewer

# --- Блок для подсказок типов (не выполняется во время работы) ---
if TYPE_CHECKING:
    import torch
    from core.engine.predictor import Predictor


# --- Вкладка "Анализ" ---
class AnalysisTab(ttk.Frame):
    def __init__(self, parent=None, class_manager: ClassManager = None):
        super().__init__(parent)
        self.predictor: Predictor | None = None
        self.analysis_image_path = None
        self.analysis_results = None
        # Храним детекции для перерисовки
        self.current_detections = []
        self._is_correction_mode = False

        self._queue_check_job = None

        # Зависимость (DI) от менеджера классов
        if class_manager is None:
            print("КРИТИЧЕСКАЯ ОШИБКА: ClassManager не был передан в AnalysisTab.")
            self.class_manager = ClassManager()
            self.class_manager.class_sets = {}
        else:
            self.class_manager = class_manager

        self.confidence_min = tk.IntVar(value=30)
        self.confidence_max = tk.IntVar(value=60)

        self.draw_mode_var = tk.StringVar(value="poly")
        self.analysis_queue = queue.Queue()  # Очередь для поточного анализа

        self._init_ui()
        self._update_button_states()
        self._create_tooltips()

    # --- Инициализация UI ---

    def _init_ui(self):
        """Создает главный разделитель (splitter) UI."""
        main_splitter = ttk.PanedWindow(self, orient="horizontal")
        main_splitter.pack(fill="both", expand=True)
        left_panel = self._create_left_panel()
        right_panel = self._create_right_panel()
        main_splitter.add(left_panel, weight=3)
        main_splitter.add(right_panel, weight=1)

    def _create_left_panel(self):
        """Создает левую панель (просмотрщик изображений и кнопки)."""
        panel = ttk.Frame(self, padding=10)
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        self.analysis_title_label = ttk.Label(panel, text="Изображение для анализа", font="-size 12 -weight bold")
        self.analysis_title_label.grid(row=0, column=0, sticky="w")

        self.analysis_viewer = PhotoViewer(panel, is_drawing_enabled=False)
        self.analysis_viewer.set_draw_mode('poly')
        self.analysis_viewer.grid(row=1, column=0, sticky="nsew", pady=5)

        self.analysis_viewer.bind_event('annotation_added', self.on_annotation_added)

        button_frame = ttk.Frame(panel)
        button_frame.grid(row=2, column=0, sticky="ew", pady=5)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        self.btn_load_for_analysis = ttk.Button(button_frame, text="1. Загрузить фото для анализа",
                                                command=self.load_image_for_analysis)
        self.btn_load_for_analysis.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.btn_analyze = ttk.Button(button_frame, text="2. Анализировать", command=self.analyze_image,
                                      bootstyle=PRIMARY)
        self.btn_analyze.grid(row=0, column=1, sticky="ew")
        return panel

    def _create_right_panel(self):
        """Создает правую панель (настройки, лог, инструменты коррекции)."""
        panel = ttk.Frame(self, padding=10)
        # Блок "Результаты" (row=2) будет расширяться
        panel.grid_rowconfigure(2, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        # --- Блок выбора модели ---
        model_selection_frame = ttk.LabelFrame(panel, text="Шаг 1: Выбор обученной модели", padding=10)
        model_selection_frame.grid(row=0, column=0, sticky="ew", pady=5)
        model_selection_frame.grid_columnconfigure(0, weight=1)

        self.model_path_label = ttk.Label(model_selection_frame, text="Модель не выбрана.", wraplength=400)
        self.model_path_label.grid(row=0, column=0, sticky="ew", pady=5)
        self.browse_model_button = ttk.Button(model_selection_frame, text="Выбрать папку с моделью",
                                              command=self.browse_model_dir)
        self.browse_model_button.grid(row=1, column=0, sticky="ew")

        analysis_settings_frame = ttk.LabelFrame(panel, text="Настройки анализа", padding=10)
        analysis_settings_frame.grid(row=1, column=0, sticky="ew", pady=5)
        analysis_settings_frame.grid_columnconfigure(0, weight=1)

        # --- Мин. уверенность ---
        self.confidence_label_min = ttk.Label(analysis_settings_frame,
                                              text=f"Мин. уверенность: {self.confidence_min.get()}%")
        self.confidence_label_min.grid(row=0, column=0, sticky="ew")
        self.confidence_slider_min = ttk.Scale(analysis_settings_frame, from_=0, to=100,
                                               variable=self.confidence_min,
                                               command=self._update_confidence_labels)
        self.confidence_slider_min.grid(row=1, column=0, sticky="ew", pady=(0, 5))

        # --- Макс. уверенность ---
        self.confidence_label_max = ttk.Label(analysis_settings_frame,
                                              text=f"Макс. уверенность: {self.confidence_max.get()}%")
        self.confidence_label_max.grid(row=2, column=0, sticky="ew", pady=5)
        self.confidence_slider_max = ttk.Scale(analysis_settings_frame, from_=0, to=100,
                                               variable=self.confidence_max,
                                               command=self._update_confidence_labels)
        self.confidence_slider_max.grid(row=3, column=0, sticky="ew", pady=(0, 5))

        # --- Блок результатов (Лог) ---
        results_group = ttk.LabelFrame(panel, text="Результаты анализа", padding=10)
        results_group.grid(row=2, column=0, sticky="nsew", pady=(10, 0))  # Используем "nsew"
        results_group.rowconfigure(1, weight=1)
        results_group.columnconfigure(0, weight=1)

        results_actions_frame = ttk.Frame(results_group)
        results_actions_frame.grid(row=0, column=0, sticky="ew", pady=(5, 0))
        self.btn_copy_log = ttk.Button(results_actions_frame, text="Копировать лог", command=self.copy_analysis_log)
        self.btn_copy_log.grid(row=0, column=0, sticky="w")

        results_list_frame = ttk.Frame(results_group)
        results_list_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        results_list_frame.rowconfigure(0, weight=1)
        results_list_frame.columnconfigure(0, weight=1)

        results_scrollbar = ttk.Scrollbar(results_list_frame, orient="vertical")
        self.analysis_results_list = Listbox(results_list_frame, background="#2A2A2A", foreground="white",
                                             relief="flat",
                                             borderwidth=0, highlightthickness=0, height=5,
                                             yscrollcommand=results_scrollbar.set)
        results_scrollbar.config(command=self.analysis_results_list.yview)

        self.analysis_results_list.grid(row=0, column=0, sticky="nsew")
        results_scrollbar.grid(row=0, column=1, sticky="ns")

        # --- Блок "Цикл обратной связи" (Коррекция) ---
        correction_title_frame = ttk.Frame(panel)
        correction_title_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        correction_title_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(correction_title_frame, text="Цикл обратной связи", font="-weight bold").grid(row=0, column=0,
                                                                                                sticky="w")
        self.correction_help_tooltip = ToolTip(correction_title_frame, "")
        self.correction_help_tooltip.grid(row=0, column=1, sticky="w", padx=5)

        feedback_group = ttk.LabelFrame(panel, text="Инструменты коррекции", padding=10)
        feedback_group.grid(row=4, column=0, sticky="ew", pady=5)
        feedback_group.grid_columnconfigure(0, weight=1)

        self.btn_toggle_correction = ttk.Button(feedback_group, text="Скорректировать ответ",
                                                command=self.toggle_correction_mode, bootstyle="outline")
        self.btn_toggle_correction.grid(row=0, column=0, sticky="ew", pady=2)

        self.btn_assign_class = ttk.Button(feedback_group, text="Назначить класс",
                                           command=self.assign_class_to_selected, bootstyle=SECONDARY)
        self.btn_assign_class.grid(row=1, column=0, sticky="ew", pady=2)
        self.btn_delete_selected = ttk.Button(feedback_group, text="Удалить выделенное", command=self.delete_selected,
                                              bootstyle=DANGER)
        self.btn_delete_selected.grid(row=2, column=0, sticky="ew", pady=2)
        self.btn_save_feedback = ttk.Button(feedback_group, text="Сохранить исправления для дообучения",
                                            command=self.save_feedback_data, bootstyle="success")
        self.btn_save_feedback.grid(row=3, column=0, sticky="ew", pady=(10, 2))
        return panel


    def get_predictor(self):
        return self.predictor

    def _update_confidence_labels(self, _=None):
        """Обновляет метки уверенности и синхронизирует ползунки."""
        min_val = self.confidence_min.get()
        max_val = self.confidence_max.get()

        # Синхронизация: min не может быть > max
        if min_val > max_val:
            self.confidence_max.set(min_val)
            max_val = min_val

        # Синхронизация: max не может быть < min
        if max_val < min_val:
            self.confidence_min.set(max_val)
            min_val = max_val

        self.confidence_label_min.config(text=f"Мин. уверенность: {int(min_val)}%")
        self.confidence_label_max.config(text=f"Макс. уверенность: {int(max_val)}%")


    def _create_tooltips(self):
        self.correction_help_tooltip.text = (
            "**Цикл обратной связи: Исправление ошибок модели**\n\n"
            "Если модель ошиблась, вы можете это исправить. Ваши исправления можно сохранить как новый датасет для дообучения.\n\n"
            "1. **Скорректировать ответ**: Активирует режим рисования (только полигоны). Вы можете добавлять новые зоны, удалять, перемещать и изменять размер существующих.\n\n"
            "2. **Назначить класс**: Позволяет изменить класс у выбранного объекта (выбрав из списка).\n\n"
            "3. **Сохранить исправления**: Создает папку `feedback_data` с исправленной разметкой. Эту папку можно выбрать на вкладке 'Обучение' и нажать 'Дообучить модель', чтобы улучшить ее на основе ваших правок.\n\n"
            "**Настройки уверенности:**\n"
            "- **Мин. уверенность:** Объекты с уверенностью НИЖЕ этого порога НЕ будут показаны.\n"
            "- **Макс. уверенность:** Объекты с уверенностью ВЫШЕ этого порога НЕ будут показаны.\n\n"
            "Используйте диапазон для фильтрации 'сомнительных' предсказаний (например, 30%-70%) для ручной проверки."
        )

    # --- Утилиты логгирования и UI ---

    def _log_analysis(self, message: str):
        """Добавляет сообщение в Listbox с временной меткой."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.analysis_results_list.insert(END, f"[{timestamp}] {message}")
        self.analysis_results_list.see(END)

    def copy_analysis_log(self):
        try:
            items = self.analysis_results_list.get(0, tk.END)
            if items:
                log_text = "\n".join(items)
                self.clipboard_clear()
                self.clipboard_append(log_text)
                messagebox.showinfo("Успех", "Логи анализа скопированы в буфер обмена.")
            else:
                messagebox.showwarning("Внимание", "Нет данных для копирования.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось скопировать логи: {e}")

    def _update_button_states(self, is_busy: bool = False):
        """Обновляет состояние кнопок, опционально блокируя их во время 'is_busy'."""
        if is_busy:
            # Блокируем все интерактивные элементы во время анализа
            model_ready = False
            image_loaded = False
            can_correct = False
            correction_buttons_state = "disabled"
            self.browse_model_button.config(state="disabled")
            self.btn_load_for_analysis.config(state="disabled")
            self.confidence_slider_min.config(state="disabled")
            self.confidence_slider_max.config(state="disabled")
        else:
            # Восстанавливаем нормальную логику
            model_ready = self.predictor is not None
            image_loaded = self.analysis_image_path is not None
            can_correct = image_loaded and self.analysis_results is not None
            correction_buttons_state = "normal" if self._is_correction_mode else "disabled"

            self.browse_model_button.config(state="normal")
            self.btn_load_for_analysis.config(state="normal")
            self.confidence_slider_min.config(state="normal")
            self.confidence_slider_max.config(state="normal")

        # Общие кнопки
        self.btn_analyze.config(state="normal" if model_ready and image_loaded else "disabled")
        self.btn_toggle_correction.config(state="normal" if can_correct else "disabled")
        self.btn_assign_class.config(state=correction_buttons_state)
        self.btn_delete_selected.config(state=correction_buttons_state)
        self.btn_save_feedback.config(state=correction_buttons_state)

    def _set_ui_busy(self, is_busy: bool):
        """Блокирует или разблокирует UI во время анализа."""
        if is_busy:
            self.btn_analyze.config(text="Анализ...", state="disabled")
            self._update_button_states(is_busy=True)  # Блокируем все остальные
            self.analysis_viewer.set_drawing_enabled(False)  # Блокируем рисование
        else:
            self.btn_analyze.config(text="2. Анализировать")
            # Восстанавливаем режим рисования, если он был
            self.analysis_viewer.set_drawing_enabled(self._is_correction_mode)
            self._update_button_states(is_busy=False)  # Разблокируем все остальные

    # --- Загрузка модели и изображения ---

    def set_model_from_checkpoint(self, checkpoint_path: str):
        """Публичный метод для установки модели (например, из вкладки Обучения)."""
        config_path = None
        model_dir = os.path.dirname(checkpoint_path)
        # Ищем конфиг рекурсивно вверх
        while model_dir != os.path.dirname(model_dir):
            potential_config_path = os.path.join(model_dir, CONFIG_FILE)
            if os.path.exists(potential_config_path):
                config_path = potential_config_path
                break
            model_dir = os.path.dirname(model_dir)

        self.initialize_predictor(checkpoint_path, config_path)

    def browse_model_dir(self):
        """Открывает диалог выбора папки с моделью."""
        model_dir = filedialog.askdirectory(title="Выберите папку с обученной моделью")
        if not model_dir:
            return

        weights_path = None
        config_path = None
        search_dir = None

        # Ищем файл весов рекурсивно
        for root, _dirs, files in os.walk(model_dir):
            if BEST_MODEL_FILE in files:
                weights_path = os.path.join(root, BEST_MODEL_FILE)
                search_dir = root  # Папка, где найдены веса
                break

        if not weights_path:
            messagebox.showwarning("Ошибка",
                                   f"Не удалось найти файл '{BEST_MODEL_FILE}' в указанной папке или ее подпапках.")
            return

        # Ищем конфиг рядом с весами, потом вверх до выбранной папки
        potential_config_path = os.path.join(search_dir, CONFIG_FILE)
        if os.path.exists(potential_config_path):
            config_path = potential_config_path
        else:
            temp_dir = search_dir
            while temp_dir.startswith(model_dir) and temp_dir != model_dir:
                potential_config_path = os.path.join(temp_dir, CONFIG_FILE)
                if os.path.exists(potential_config_path):
                    config_path = potential_config_path
                    break
                temp_dir = os.path.dirname(temp_dir)
            if not config_path:  # Последняя попытка в самой выбранной папке
                potential_config_path = os.path.join(model_dir, CONFIG_FILE)
                if os.path.exists(potential_config_path):
                    config_path = potential_config_path

        self.initialize_predictor(weights_path, config_path)

    def initialize_predictor(self, checkpoint_path: str, config_path: str = None):
        """
        Главный метод инициализации. Загружает "тяжелые" библиотеки
        и создает экземпляр Predictor.
        """
        try:
            import torch
            from core.engine.predictor import Predictor
            from core.models.custom_plant_model import CustomPlantModel
        except ImportError as e:
            messagebox.showerror("Ошибка импорта",
                                 f"Не удалось загрузить необходимые библиотеки (torch, cv2 и т.д.): {e}")
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

            config_data = None
            model_state_dict = None

            # Проверяем, встроен ли конфиг в checkpoint
            if isinstance(checkpoint, dict) and 'config' in checkpoint and 'model_state_dict' in checkpoint:
                config_data = checkpoint['config']
                model_state_dict = checkpoint['model_state_dict']
            elif config_path is not None:  # Если нет, ищем внешний config.yaml
                config_data = load_config(config_path)

                # Загружаем веса в зависимости от формата checkpoint
                if isinstance(checkpoint, torch.nn.Module):
                    model_state_dict = checkpoint.state_dict()
                elif isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model_state_dict = checkpoint['model_state_dict']
                    else:  # Предполагаем, что сам словарь - это state_dict
                        model_state_dict = checkpoint
                else:
                    raise ValueError(f"Файл весов .pt имеет неизвестный формат, но {CONFIG_FILE} найден.")

            else:  # Если ни конфиг не встроен, ни внешний не найден
                raise ValueError(
                    f"Не удалось найти конфигурацию модели. Убедитесь, что в папке с моделью есть файл '{CONFIG_FILE}', "
                    "либо что конфигурация встроена в файл 'best_model.pt'.")

            if config_data is None or model_state_dict is None:
                raise ValueError("Не удалось загрузить конфигурацию или веса модели.")

            # Создаем экземпляр модели
            model_name = config_data.get('model', {}).get('name')
            if model_name == 'CustomPlantModel':
                model_instance = CustomPlantModel(config_data)
                model_instance.load_state_dict(model_state_dict)
                self.predictor = Predictor(model_instance, config_data)

                # Пытаемся получить имя, заданное пользователем
                display_name = config_data.get("display_name")
                if not display_name:
                    # Если его нет (старая модель), используем имя класса
                    display_name = model_name

                self.model_path_label.config(text=f"Загружена модель: {display_name}")

                messagebox.showinfo("Успех", "Модель для анализа успешно загружена.")
            else:
                raise ValueError(f"Неизвестный тип модели: {model_name}")
        except Exception as e:
            messagebox.showerror("Ошибка инициализации", f"Не удалось создать модель: {e}\n{traceback.format_exc()}")
            self.predictor = None
        finally:
            self._update_button_states()

    def load_image_for_analysis(self):
        """Открывает диалог выбора изображения для анализа."""
        path = filedialog.askopenfilename(title="Выберите изображение",
                                          filetypes=[("Image Files", SUPPORTED_IMAGE_FORMATS)])
        if path:
            try:
                # PhotoViewer.set_photo() может вызвать ошибку (напр. PIL.UnidentifiedImageError)
                self.analysis_viewer.set_photo(path)
            except Exception as e:
                messagebox.showerror("Ошибка открытия изображения",
                                     f"Не удалось загрузить файл:\n{path}\n\n"
                                     f"Возможно, файл поврежден или имеет неподдерживаемый формат.\n\nОшибка: {e}")
                return

            self.analysis_image_path = path
            self.analysis_title_label.config(text=f"Анализ изображения: {os.path.basename(path)}")
            self.analysis_results_list.delete(0, END)
            self.analysis_results = None
            self.current_detections = []  # Сбрасываем детекции
            if self._is_correction_mode:  # Если был режим коррекции, выключаем его
                self.toggle_correction_mode()
            self._update_button_states()

    # --- Логика асинхронного анализа (Поток и Очередь) ---

    def _analysis_thread_worker(self):
        """Выполняется в отдельном потоке для выполнения self.predictor.predict."""
        try:
            task_data = self.analysis_queue.get()  # Получаем задачу

            results_json, detections = self.predictor.predict(
                task_data["image_path"],
                threshold_min=task_data["threshold_min"],
                threshold_max=task_data["threshold_max"]
            )

            self.analysis_queue.put({"status": "success", "results_json": results_json, "detections": detections})
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"Ошибка в потоке анализа: {e}\n{traceback_str}")
            self.analysis_queue.put({"status": "error", "error": e, "traceback": traceback_str})

    def _check_analysis_queue(self):
        """Выполняется в UI-потоке для проверки результатов из очереди."""
        try:
            result = self.analysis_queue.get_nowait()
            self.analysis_results_list.delete(0, END)

            if result["status"] == "success":
                self.analysis_results = result["results_json"]
                # Сохраняем детекции для перерисовки
                self.current_detections = result["detections"]

                class_counts = self.analysis_results.get("class_counts", {})
                total_objects = sum(class_counts.values())

                self._log_analysis(f"Найдено объектов: {total_objects}")

                if total_objects > 0:
                    # Сортируем классы по имени для стабильного вывода
                    for class_name, count in sorted(class_counts.items()):
                        report_str = f"-> {class_name}: {count}"
                        self.analysis_results_list.insert(END, report_str)

                # Получаем цвета из менеджера классов и отрисовываем
                if self.class_manager:
                    all_class_colors = self.class_manager.get_class_colors()
                    for det in self.current_detections:
                        clean_class_name = det['class_name'].split(' (')[0]
                        det['color'] = all_class_colors.get(clean_class_name, '#FF0000')
                self.analysis_viewer.redraw_annotations_from_model(self.current_detections)

            elif result["status"] == "error":
                traceback_str = result["traceback"]
                self._log_analysis("Ошибка анализа. Детали ниже:")
                for line in traceback_str.splitlines():
                    self.analysis_results_list.insert(END, line)
                messagebox.showerror("Ошибка", "Произошла ошибка во время анализа. Подробности в окне логов.")
            self._set_ui_busy(False)
        except queue.Empty:
            self._queue_check_job = self.after(100, self._check_analysis_queue)

    def destroy(self):
        if self._queue_check_job:
            self.after_cancel(self._queue_check_job)
        super().destroy()

    # -----------------------------------------------

    def analyze_image(self):
        """Запускает анализ изображения в отдельном потоке."""
        if not self.analysis_image_path or not self.predictor:
            if not self.predictor:
                messagebox.showwarning("Внимание", "Сначала выберите и загрузите модель на панели справа.")
            return

        self.analysis_results_list.delete(0, END)
        self._log_analysis("Запуск анализа в потоке...")
        self.update_idletasks()
        self._set_ui_busy(True)

        threshold_min = self.confidence_min.get() / 100.0
        threshold_max = self.confidence_max.get() / 100.0

        task_data = {
            "image_path": self.analysis_image_path,
            "threshold_min": threshold_min,
            "threshold_max": threshold_max
        }

        # Очищаем очередь, если там что-то осталось
        while not self.analysis_queue.empty():
            with contextlib.suppress(queue.Empty):
                self.analysis_queue.get_nowait()
        self.analysis_queue.put(task_data)

        threading.Thread(target=self._analysis_thread_worker, daemon=True).start()
        self._queue_check_job = self.after(100, self._check_analysis_queue)

    # --- Логика цикла обратной связи (Коррекция) ---

    def toggle_correction_mode(self):
        """Включает/выключает режим коррекции (рисования) на холсте."""
        self._is_correction_mode = not self._is_correction_mode
        self.analysis_viewer.set_drawing_enabled(self._is_correction_mode)

        if not self._is_correction_mode:
            current_annotations = list(self.analysis_viewer.item_map.values())
            try:
                if self.analysis_image_path:
                    self.analysis_viewer.set_photo(self.analysis_image_path)
            except Exception as e:
                print(f"Ошибка при перезагрузке фото в toggle_correction_mode: {e}")
            self.analysis_viewer.redraw_annotations_from_model(current_annotations)

        style = "success" if self._is_correction_mode else "outline"
        self.btn_toggle_correction.config(bootstyle=style)
        self.btn_toggle_correction.config(
            text="Завершить коррекцию" if self._is_correction_mode else "Скорректировать ответ")
        self._update_button_states()

    def _ask_class_dialog(self, title: str, prompt: str, class_list: list, initial_value: str = None) -> str | None:
        """Отображает кастомный модальный диалог с Combobox для выбора класса."""
        dialog = Toplevel(self)
        dialog.title(title)
        dialog.transient(self)  # Делаем диалог модальным для родителя

        dialog.resizable(False, False)

        result = tk.StringVar()

        ttk.Label(dialog, text=prompt, wraplength=300).pack(padx=10, pady=10)

        combo = ttk.Combobox(dialog, values=class_list, state="readonly", width=40)
        if initial_value and initial_value in class_list:
            combo.set(initial_value)
        elif class_list:
            combo.set(class_list[0])
        combo.pack(padx=10, pady=5, fill="x")

        # Обработчики кнопок
        def on_ok():
            result.set(combo.get())
            dialog.destroy()

        def on_cancel():
            result.set("")
            dialog.destroy()

        # Кнопки OK и Отмена
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(padx=10, pady=10, fill="x")
        ok_btn = ttk.Button(btn_frame, text="OK", command=on_ok, bootstyle=PRIMARY)
        ok_btn.pack(side="right", padx=5)
        cancel_btn = ttk.Button(btn_frame, text="Отмена", command=on_cancel, bootstyle=SECONDARY)
        cancel_btn.pack(side="right")

        dialog.bind("<Return>", lambda e: on_ok())
        dialog.bind("<Escape>", lambda e: on_cancel())

        # Центрирование диалога
        dialog.update_idletasks()
        try:
            parent_x = self.winfo_rootx()
            parent_y = self.winfo_rooty()
            parent_w = self.winfo_width()
            parent_h = self.winfo_height()
            dialog_w = dialog.winfo_width()
            dialog_h = dialog.winfo_height()
            x = parent_x + (parent_w // 2) - (dialog_w // 2)
            y = parent_y + (parent_h // 2) - (dialog_h // 2)
            dialog.geometry(f"+{x}+{y}")
        except Exception:
            pass

        dialog.wait_visibility()
        dialog.grab_set()
        dialog.wait_window()

        final_result = result.get()
        return final_result if final_result else None

    def _get_model_class_names(self) -> list[str]:
        """
        Централизованный метод для получения списка классов из конфига.
        """
        if not self.predictor:
            return []

        try:
            task_outputs = self.predictor.config.get('task', {}).get('outputs', {})
            all_classes = set()
            for task in task_outputs.values():
                if task.get('type') == 'detection':
                    all_classes.update(task.get('classes', []))
            return sorted(all_classes - {"background"})
        except Exception as e:
            print(f"Ошибка парсинга классов модели: {e}")
            return []

    def on_annotation_added(self, annotation_type, coords):
        """Событие, которое вызывается PhotoViewer, когда пользователь закончил рисовать."""
        if not self._is_correction_mode:
            return
        annotation_type = 'poly'

        if not self.predictor:
            messagebox.showwarning("Внимание", "Модель не загружена. Невозможно определить список классов.")
            return

        class_names = self._get_model_class_names()

        if not class_names:
            messagebox.showwarning("Внимание", "Список классов модели не найден или пуст.")
            return

        new_class = self._ask_class_dialog(
            "Выбор класса", "Выберите класс для новой аннотации:", class_names
        )

        if new_class and new_class in class_names:
            new_color = '#FF0000'
            if self.class_manager:
                new_color = self.class_manager.get_class_colors().get(new_class, '#FF0000')

            ann_data = {'id': str(uuid.uuid4()), 'class_name': new_class, 'type': annotation_type, 'coords': coords,
                        'color': new_color}
            self.analysis_viewer.add_annotation(ann_data)
        elif new_class:
            messagebox.showwarning("Ошибка", f"Класс '{new_class}' не найден в списке классов модели.")

    def assign_class_to_selected(self):
        """Назначает новый класс для выделенной на холсте аннотации."""
        selected_ids = self.analysis_viewer.get_selected_ids()
        if not selected_ids:
            messagebox.showwarning("Внимание", "Сначала выберите аннотацию на изображении.")
            return
        bbox_id = selected_ids[0]

        if not self.predictor:
            messagebox.showwarning("Внимание", "Модель не загружена. Невозможно определить список классов.")
            return

        current_class_obj = self.analysis_viewer.item_map.get(bbox_id)
        if not current_class_obj:
            return

        current_class = current_class_obj['class_name'].split(' (')[0]
        class_names = self._get_model_class_names()

        if not class_names:
            messagebox.showwarning("Внимание", "Список классов модели не найден или пуст.")
            return

        new_class = self._ask_class_dialog(
            "Назначить класс",
            "Выберите новое имя класса:",
            class_names,
            initial_value=current_class,
        )

        if new_class and new_class in class_names and new_class != new_class:
            new_color = '#FF0000'
            if self.class_manager:
                new_color = self.class_manager.get_class_colors().get(new_class, '#FF0000')

            self.analysis_viewer.item_map[bbox_id]['class_name'] = new_class
            self.analysis_viewer.item_map[bbox_id]['color'] = new_color
            coords = self.analysis_viewer.item_map[bbox_id]['coords']
            self.analysis_viewer.update_annotation_coords(bbox_id, coords)

    def delete_selected(self):
        """Удаляет выделенные на холсте аннотации."""
        selected_ids = self.analysis_viewer.get_selected_ids()
        if not selected_ids:
            messagebox.showwarning("Внимание", "Сначала выберите аннотацию для удаления.")
            return
        for item_id in selected_ids:
            self.analysis_viewer.remove_annotation(item_id)

    def save_feedback_data(self):
        """Сохраняет текущие (исправленные) аннотации для дообучения."""
        if not self.analysis_image_path:
            return

        if not self.predictor:
            messagebox.showwarning("Внимание",
                                   "Модель не загружена. Невозможно определить список классов для data.yaml.")
            return

        try:
            feedback_images_dir = os.path.join(FEEDBACK_DIR, IMAGES_DIR)
            feedback_labels_dir = os.path.join(FEEDBACK_DIR, LABELS_DIR)

            if not os.path.exists(feedback_images_dir):
                if messagebox.askyesno("Создать папку?",
                                       f"Папка '{FEEDBACK_DIR}' не найдена. Создать ее для сохранения исправлений?"):
                    os.makedirs(feedback_images_dir)
                    os.makedirs(feedback_labels_dir)
                else:
                    return

            base_filename = os.path.basename(self.analysis_image_path)
            shutil.copy(self.analysis_image_path, os.path.join(feedback_images_dir, base_filename))

            label_path = os.path.join(feedback_labels_dir, os.path.splitext(base_filename)[0] + '.json')
            annotations_list = list(self.analysis_viewer.item_map.values())

            clean_annotations = []
            for ann in annotations_list:
                clean_ann = ann.copy()
                clean_ann.pop('id', None)
                clean_ann.pop('color', None)
                clean_ann['class_name'] = clean_ann['class_name'].split(' (')[0]

                if 'coords' in clean_ann and clean_ann['coords'] is not None:
                    if clean_ann['type'] == 'rect':
                        clean_ann['coords'] = tuple(float(c) for c in clean_ann['coords'])
                    elif clean_ann['type'] == 'poly':
                        clean_ann['coords'] = [[float(x), float(y)] for x, y in clean_ann['coords']]
                clean_annotations.append(clean_ann)

            full_data = {"image_path": base_filename,
                         "annotations": clean_annotations}

            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(full_data, f, indent=2, ensure_ascii=False)

            data_yaml_path = os.path.join(FEEDBACK_DIR, DATA_YAML_FILE)
            task_outputs = self.predictor.config.get('task', {}).get('outputs', {})
            all_classes = set()
            for task in task_outputs.values():
                if task.get('type') == 'detection':
                    all_classes.update(task.get('classes', []))

            class_names = sorted(all_classes)
            if "background" in class_names:
                class_names.remove("background")
            class_names.insert(0, "background")

            data_yaml = {'path': os.path.abspath(FEEDBACK_DIR), 'train': IMAGES_DIR, 'val': IMAGES_DIR,
                         'nc': len(class_names),
                         'names': class_names}
            with open(data_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

            messagebox.showinfo("Успех",
                                f"Исправления сохранены в папку '{FEEDBACK_DIR}'.\nТеперь вы можете выбрать эту папку на вкладке 'Обучение' для дообучения модели.")
        except OSError as e:
            messagebox.showerror("Ошибка сохранения",
                                 f"Не удалось сохранить данные для дообучения (ошибка I/O):\n{e}\n\n"
                                 "Возможно, диск полон или нет прав на запись.")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения",
                                 f"Не удалось сохранить данные для дообучения: {e}\n{traceback.format_exc()}")