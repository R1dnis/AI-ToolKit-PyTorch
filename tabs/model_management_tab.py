import contextlib
import datetime
import os
import queue
import threading
import time
import tkinter as tk
import traceback
from tkinter import Toplevel, filedialog, messagebox, simpledialog
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import torch
import ttkbootstrap as ttk
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from ttkbootstrap.constants import DANGER, END, PRIMARY, SECONDARY

from constants import (
    BEST_MODEL_FILE,
    CONFIG_FILE,
    DATA_YAML_FILE,
    DEFAULT_DATASET_NAME,
    DEFAULT_MODEL_NAME,
    IMAGES_DIR,
    MODELS_DIR,
    RUNS_DIR,
)
from core.config import get_default_config, load_config, save_config
from core.data.transforms import get_augmentation_levels
from core.engine.trainer import Trainer
from core.models.custom_plant_model import CustomPlantModel
from tooltip import ToolTip

if TYPE_CHECKING:
    pass




class CustomTrainingThread(threading.Thread):
    def __init__(
            self,
            config: dict[str, Any],
            weights_path: str | None,
            queue: queue.Queue,
            interrupt_event: threading.Event,
    ):
        super().__init__()
        self.config = config
        self.weights_path = weights_path
        self.queue = queue
        self.interrupt_event = interrupt_event
        self.daemon = True

    def _log(self, message: str):
        """Отправка стандартного лога в очередь"""
        self.queue.put(("log", message))

    def _vlog(self, message: str):
        """Отправка verbose-лога в очередь"""
        self.queue.put(("verbose_log", message))

    def run(self):
        try:
            self._log("--- Начало обучения кастомной модели ---")

            aug_level = self.config.get("training", {}).get(
                "augmentation_level", "Легкая"
            )
            # config["augmentation_params"] больше не нужен,
            # CustomDataset будет читать augmentation_level напрямую.
            self._log(f"Применение аугментации уровня: {aug_level}")

            model_name = self.config.get("model", {}).get("name")
            if model_name == "CustomPlantModel":
                model = CustomPlantModel(self.config)
                if self.weights_path and os.path.exists(self.weights_path):
                    self._log(
                        f"Загрузка весов из {self.weights_path} для дообучения..."
                    )
                    loaded_data = torch.load(
                        self.weights_path, map_location=torch.device("cpu")
                    )

                    state_to_load = None
                    if isinstance(loaded_data, torch.nn.Module):
                        state_to_load = loaded_data.state_dict()
                        self._log("Обнаружен полный объект модели. Извлечение весов...")
                    elif isinstance(loaded_data, dict):
                        if "model_state_dict" in loaded_data:
                            state_to_load = loaded_data["model_state_dict"]
                            self._log(
                                "Обнаружен checkpoint. Загрузка состояния модели..."
                            )
                        else:
                            state_to_load = loaded_data
                            self._log("Обнаружен словарь состояния. Загрузка весов...")

                    if state_to_load:
                        model.load_state_dict(state_to_load)
                        self._log("Веса успешно загружены в модель.")
                    else:
                        raise TypeError(
                            f"Неподдерживаемый тип данных в файле весов: {type(loaded_data)}"
                        )
            else:
                raise ValueError(f"Неизвестное имя модели: {model_name}")

            trainer = Trainer(
                model,
                self.config,
                self.queue,
                vlog_callback=self._vlog,
                interrupt_event=self.interrupt_event,
            )
            trainer.train()

        except Exception as e:
            self._log(f"Критическая ошибка: {e}")
            self._log(traceback.format_exc())
            self.queue.put(("finished", (False, str(e))))


class ModelManagementTab(ttk.Frame):
    # model_loaded_callback = None

    def __init__(self, parent=None, on_model_trained_callback: callable = None):
        super().__init__(parent)
        self.model_dir = None
        self.config_data = None
        self.training_thread = None
        self.dataset_path_from_labeling = None
        self.val_dataset_path = None
        self.queue = queue.Queue()
        self.preview_window = None
        self.preview_images = []
        self.interactive_widgets = []

        self.training_start_time = None
        self.epochs_total = 0
        self.current_epoch = 0
        self.interrupt_event = threading.Event()
        self.timer_job = None

        self.on_model_trained_callback = on_model_trained_callback

        self.prev_train_loss = float("inf")
        self.prev_val_loss = float("inf")

        self._init_ui()
        self._update_button_states()
        self._create_tooltips()
        self.master.after(100, self.process_queue)

    def _log_to_widget(self, widget: tk.Text, message: str):
        """Добавляет сообщение в Text виджет с временной меткой."""
        try:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            widget.config(state="normal")
            widget.insert(END, f"[{timestamp}] {message}\n")
            widget.see(END)
            widget.config(state="disabled")
        except tk.TclError:
            pass

    def _get_arrow(self, new_val: float, old_val: float) -> str:
        """Возвращает строку со стрелкой и цветом (bootstyle)."""
        if new_val < old_val:
            return " (▼)", "success"  # Упало (хорошо)
        elif new_val > old_val:
            return " (▲)", "danger"  # Возросло (плохо)
        else:
            return " (━)", "default"  # Не изменилось

    def _humanize_time(self, seconds: int) -> str:
        """Преобразует секунды в человекочитаемый формат ETA."""
        if seconds < 60:
            return "меньше минуты"
        if seconds < 120:
            return "около минуты"
        if seconds < 3600:
            minutes = round(seconds / 60)
            if minutes < 10:
                return f"около {minutes} минут"
            else:
                return f"около {minutes} минут"
        hours = round(seconds / 3600)
        if hours == 1:
            return "около часа"
        return f"около {hours} часов"

    def _update_model_state(self):
        """Анализирует Train/Val Loss и обновляет статус модели."""
        t_loss = self.prev_train_loss
        v_loss = self.prev_val_loss

        t_text = self.train_loss_label.cget("text")
        v_text = self.val_loss_label.cget("text")

        if t_loss == float("inf") or v_loss == float("inf"):
            self.model_state_label.config(text="Оценка...", bootstyle="info")
            return

        if "▲" in v_text and "▼" in t_text and v_loss > t_loss * 1.5:
            self.model_state_label.config(text="Переобучение", bootstyle="warning")
            return

        if "▼" in v_text and "▼" in t_text and v_loss > t_loss * 3.0:
            self.model_state_label.config(
                text="Возможно переобучение", bootstyle="warning"
            )
            return

        if "▲" in v_text and "▲" in t_text:
            self.model_state_label.config(
                text="Проблема (Loss растет)", bootstyle="danger"
            )
            return

        if "▼" in v_text and "▼" in t_text and v_loss <= t_loss * 1.5:
            self.model_state_label.config(
                text="Нормальное (Обучается)", bootstyle="success"
            )
            return

        if t_loss > 0.5:
            self.model_state_label.config(text="Недообучение", bootstyle="info")
            return

        if "━" in v_text and "▼" in t_text:
            self.model_state_label.config(text="Схождение (Плато)", bootstyle="success")
            return

        self.model_state_label.config(text="Нормальное (Обучается)", bootstyle="info")

    def process_queue(self):
        try:
            while True:
                msg_type, msg_data = self.queue.get_nowait()

                if msg_type == "log":
                    self._log_to_widget(self.log_area, msg_data)

                elif msg_type == "verbose_log":
                    self._log_to_widget(self.verbose_log_area, msg_data)

                elif msg_type == "status":
                    self.training_status_label.config(text=msg_data)

                elif msg_type == "epoch":
                    if isinstance(msg_data, tuple):
                        self.current_epoch, self.epochs_total = msg_data
                        self.training_status_label.config(
                            text=f"Эпоха: {self.current_epoch}/{self.epochs_total}"
                        )
                        self._update_eta()
                    else:
                        pass

                elif msg_type == "metrics":
                    train_loss = msg_data.get("train_loss")
                    val_loss = msg_data.get("val_loss")

                    if train_loss is not None:
                        arrow, style = self._get_arrow(train_loss, self.prev_train_loss)
                        self.train_loss_label.config(
                            text=f"Train Loss: {train_loss:.4f}{arrow}", bootstyle=style
                        )
                        self.prev_train_loss = train_loss

                    if val_loss is not None:
                        arrow, style = self._get_arrow(val_loss, self.prev_val_loss)
                        self.val_loss_label.config(
                            text=f"Val Loss: {val_loss:.4f}{arrow}", bootstyle=style
                        )
                        self.prev_val_loss = val_loss

                    self._update_model_state()

                elif msg_type == "progress":
                    self.progress_bar["value"] = msg_data

                elif msg_type == "finished":
                    self._stop_timer()
                    success, data = msg_data
                    if success:
                        model_path, history = data
                        self.on_training_finished(success, model_path, history)
                    else:
                        error_message = data
                        self.on_training_finished(success, error_message, None)

        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.process_queue)

    def _init_ui(self):
        splitter = ttk.PanedWindow(self, orient="horizontal")
        splitter.pack(fill="both", expand=True)
        left_panel = self._create_left_panel()
        right_panel = self._create_right_panel()
        splitter.add(left_panel, weight=1)
        splitter.add(right_panel, weight=1)

        self.interactive_widgets.extend(
            [
                self.btn_create_model,
                self.btn_select_model_dir,
                self.btn_select_dataset_dir,
                self.btn_preview_dataset,
                self.btn_start_training,
                self.btn_continue_training,
                self.epochs_input,
                self.lr_input,
                self.batch_size_input,
                self.augmentation_level_combo,
                self.btn_select_val_dataset_dir,
            ]
        )

    def _create_left_panel(self):
        panel = ttk.Frame(self, padding=10)
        panel.grid_rowconfigure(3, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        model_group = ttk.LabelFrame(
            panel, text="Шаг 1: Выбор или создание модели", padding=10
        )
        model_group.grid(row=0, column=0, sticky="ew", pady=5)
        self.model_path_label = ttk.Label(
            model_group, text=DEFAULT_MODEL_NAME, wraplength=400
        )
        self.model_path_label.pack(fill="x", pady=5)
        self.btn_create_model = ttk.Button(
            model_group, text="Создать новую модель", command=self.create_new_model
        )
        self.btn_create_model.pack(fill="x", pady=2)

        self.btn_select_model_dir = ttk.Button(
            model_group, text="Выбрать папку с моделью", command=self.select_model_dir
        )
        self.btn_select_model_dir.pack(fill="x", pady=2)

        dataset_group = ttk.LabelFrame(
            panel, text="Шаг 2: Выбор датасета (Train)", padding=10
        )
        dataset_group.grid(row=1, column=0, sticky="ew", pady=5)
        self.dataset_path_label = ttk.Label(
            dataset_group, text=DEFAULT_DATASET_NAME, wraplength=400
        )
        self.dataset_path_label.pack(fill="x", pady=5)
        dataset_buttons_frame = ttk.Frame(dataset_group)
        dataset_buttons_frame.pack(fill="x")
        self.btn_select_dataset_dir = ttk.Button(
            dataset_buttons_frame, text="Выбрать папку", command=self.select_dataset_dir
        )
        self.btn_select_dataset_dir.pack(
            fill="x", pady=2, side="left", expand=True, padx=(0, 5)
        )
        self.btn_preview_dataset = ttk.Button(
            dataset_buttons_frame, text="Предпросмотр", command=self.preview_dataset
        )
        self.btn_preview_dataset.pack(fill="x", pady=2, side="left", expand=True)

        val_dataset_group = ttk.LabelFrame(
            panel, text="Шаг 2.5: (Опционально) Валидационный датасет", padding=10
        )
        val_dataset_group.grid(row=2, column=0, sticky="ew", pady=5)
        self.val_dataset_path_label = ttk.Label(
            val_dataset_group, text="Валидационный датасет не выбран", wraplength=400
        )
        self.val_dataset_path_label.pack(fill="x", pady=5)
        self.btn_select_val_dataset_dir = ttk.Button(
            val_dataset_group,
            text="Выбрать папку (Валидация)",
            command=self.select_val_dataset_dir,
        )
        self.btn_select_val_dataset_dir.pack(fill="x", pady=2)

        logs_group = ttk.LabelFrame(panel, text="Логи", padding=10)
        logs_group.grid(row=3, column=0, sticky="nsew", pady=5)
        logs_group.grid_rowconfigure(1, weight=1)
        logs_group.grid_columnconfigure(0, weight=1)

        self.btn_copy_logs = ttk.Button(
            logs_group, text="Скопировать лог", command=self.copy_logs_to_clipboard
        )
        self.btn_copy_logs.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))

        log_scrollbar = ttk.Scrollbar(logs_group, orient="vertical")
        self.log_area = tk.Text(
            logs_group,
            height=10,
            state="disabled",
            background="#2A2A2A",
            foreground="white",
            relief="flat",
            yscrollcommand=log_scrollbar.set,
        )
        log_scrollbar.config(command=self.log_area.yview)

        self.log_area.grid(row=1, column=0, sticky="nsew")
        log_scrollbar.grid(row=1, column=1, sticky="ns")

        self.log_context_menu = tk.Menu(self.log_area, tearoff=0)
        self.log_context_menu.add_command(
            label="Копировать", command=self.copy_logs_to_clipboard
        )
        self.log_area.bind("<Button-3>", self.show_log_context_menu)

        return panel

    def show_log_context_menu(self, event):
        self.log_context_menu.tk_popup(event.x_root, event.y_root)

    def copy_logs_to_clipboard(self):
        try:
            self.clipboard_clear()
            self.clipboard_append(self.log_area.get("1.0", tk.END))
            messagebox.showinfo(
                "Скопировано",
                "Содержимое лога скопировано в буфер обмена.",
                parent=self,
            )
        except Exception as e:
            messagebox.showerror(
                "Ошибка", f"Не удалось скопировать лог: {e}", parent=self
            )

    def _create_right_panel(self):
        panel = ttk.Frame(self, padding=10)
        panel.grid_rowconfigure(3, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        params_title_frame = ttk.Frame(panel)
        params_title_frame.grid(row=0, column=0, sticky="ew")

        params_title_frame.columnconfigure(0, weight=0)
        params_title_frame.columnconfigure(1, weight=1)

        ttk.Label(
            params_title_frame, text="Параметры и запуск", font="-weight bold"
        ).grid(row=0, column=0, sticky="w")

        self.params_help_tooltip = ToolTip(params_title_frame, "")
        self.params_help_tooltip.grid(row=0, column=1, sticky="w", padx=5)

        params_group = ttk.LabelFrame(
            panel, text="Шаг 3: Настройка параметров", padding=10
        )
        params_group.grid(row=1, column=0, sticky="ew", pady=5)
        form_layout = ttk.Frame(params_group)
        form_layout.pack(fill="x")
        ttk.Label(form_layout, text="Количество эпох:").grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.epochs_input = ttk.Spinbox(form_layout, from_=1, to=10000)
        self.epochs_input.set(100)
        self.epochs_input.grid(row=0, column=1, sticky="ew", pady=5, padx=5)
        ttk.Label(form_layout, text="Скорость обучения (LR):").grid(
            row=1, column=0, sticky="w", pady=5
        )
        self.lr_input = ttk.Entry(form_layout)
        self.lr_input.insert(0, "0.001")
        self.lr_input.grid(row=1, column=1, sticky="ew", pady=5, padx=5)
        ttk.Label(form_layout, text="Размер батча:").grid(
            row=2, column=0, sticky="w", pady=5
        )
        self.batch_size_input = ttk.Spinbox(form_layout, from_=1, to=512)
        self.batch_size_input.set(8)
        self.batch_size_input.grid(row=2, column=1, sticky="ew", pady=5, padx=5)

        ttk.Label(form_layout, text="Уровень аугментации:").grid(
            row=3, column=0, sticky="w", pady=5
        )
        self.augmentation_level_combo = ttk.Combobox(
            form_layout, state="readonly", values=get_augmentation_levels()
        )
        self.augmentation_level_combo.set("Легкая")
        self.augmentation_level_combo.grid(row=3, column=1, sticky="ew", pady=5, padx=5)

        form_layout.columnconfigure(1, weight=1)

        actions_group = ttk.LabelFrame(panel, text="Шаг 4: Запуск обучения", padding=10)
        actions_group.grid(row=2, column=0, sticky="ew", pady=5)
        self.btn_start_training = ttk.Button(
            actions_group, text="Обучить с нуля", command=self.start_training, bootstyle=PRIMARY
        )
        self.btn_start_training.pack(fill="x", pady=2)
        self.btn_continue_training = ttk.Button(
            actions_group,
            text="Дообучить модель",
            command=self.continue_training,
            bootstyle=SECONDARY,
        )
        self.btn_continue_training.pack(fill="x", pady=2)

        self.btn_interrupt_training = ttk.Button(
            actions_group,
            text="Прервать обучение",
            command=self.request_training_stop,
            bootstyle=DANGER,
        )
        self.btn_interrupt_training.pack(fill="x", pady=(10, 2))
        self.btn_interrupt_training.config(state="disabled")

        progress_group = ttk.LabelFrame(panel, text="Прогресс", padding=10)
        progress_group.grid(row=3, column=0, sticky="nsew", pady=5)
        progress_group.grid_rowconfigure(5, weight=1)
        progress_group.grid_columnconfigure(0, weight=1)

        self.training_status_label = ttk.Label(progress_group, text="Готов к работе.")
        self.training_status_label.grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=2
        )

        self.progress_bar = ttk.Progressbar(progress_group, mode="determinate")
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)

        # Замена .pack() на .grid() для всех дочерних элементов
        timer_frame = ttk.Frame(progress_group)
        timer_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        timer_frame.columnconfigure(0, weight=1)
        timer_frame.columnconfigure(1, weight=1)

        self.timer_label = ttk.Label(timer_frame, text="Время: 00:00:00")
        self.timer_label.grid(row=0, column=0, sticky="w", padx=5)

        self.eta_label = ttk.Label(timer_frame, text="Осталось: --:--:--")
        self.eta_label.grid(row=0, column=1, sticky="w", padx=5)

        loss_values_frame = ttk.Frame(progress_group)
        loss_values_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        loss_values_frame.columnconfigure(0, weight=1)
        loss_values_frame.columnconfigure(1, weight=1)

        train_loss_frame = ttk.Frame(loss_values_frame)
        train_loss_frame.grid(row=0, column=0, sticky="w", padx=5)

        self.train_loss_label = ttk.Label(train_loss_frame, text="Train Loss: --.----")
        self.train_loss_label.grid(row=0, column=0, sticky="w")

        self.train_loss_tooltip = ToolTip(
            train_loss_frame,
            "**Train Loss (Потери на обучении):**\n"
            "Показывает, насколько хорошо модель предсказывает данные, на которых она УЖЕ обучалась.\n"
            "Низкое значение = модель хорошо 'запомнила' обучающие примеры.",
        )
        self.train_loss_tooltip.grid(row=0, column=1, sticky="w", padx=5)

        val_loss_frame = ttk.Frame(loss_values_frame)
        val_loss_frame.grid(row=0, column=1, sticky="w", padx=5)

        self.val_loss_label = ttk.Label(val_loss_frame, text="Val Loss: --.----")
        self.val_loss_label.grid(row=0, column=0, sticky="w")

        self.val_loss_tooltip = ToolTip(
            val_loss_frame,
            "**Val Loss (Потери на валидации):**\n"
            "Показывает, насколько хорошо модель работает на НОВЫХ данных, которые она НЕ видела.\n"
            "Это главный показатель 'ума' модели.",
        )
        self.val_loss_tooltip.grid(row=0, column=1, sticky="w", padx=5)

        model_state_frame = ttk.Frame(progress_group)
        model_state_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)

        ttk.Label(model_state_frame, text="Состояние модели:").grid(row=0, column=0, sticky="w",
                                                                    padx=(5, 0))

        self.model_state_label = ttk.Label(
            model_state_frame, text="--", bootstyle="default"
        )
        self.model_state_label.grid(row=0, column=1, sticky="w", padx=5)

        self.model_state_tooltip = ToolTip(
            model_state_frame, ""
        )
        self.model_state_tooltip.grid(row=0, column=2, sticky="w")

        ttk.Label(progress_group, text="Детальный лог:").grid(
            row=5, column=0, sticky="w", pady=(10, 0)
        )
        verbose_log_scrollbar = ttk.Scrollbar(progress_group, orient="vertical")
        self.verbose_log_area = tk.Text(
            progress_group,
            state="disabled",
            background="#1A1A1A",
            foreground="gray",
            relief="flat",
            yscrollcommand=verbose_log_scrollbar.set,
            font=("Courier New", 8),
        )
        verbose_log_scrollbar.config(command=self.verbose_log_area.yview)

        self.verbose_log_area.grid(row=6, column=0, sticky="nsew")
        verbose_log_scrollbar.grid(row=6, column=1, sticky="ns")

        return panel

    def _create_tooltips(self):
        self.params_help_tooltip.text = (
            "**Шаг 1: Создание и выбор модели**\n"
            "1. **Создайте модель:** Нажмите 'Создать новую модель' (например, 'Tomato_v1'). Будет создана папка в `models/` с базовым `config.yaml`.\n"
            "2. **Выберите модель:** Укажите папку с вашей моделью.\n\n"
            "**Шаг 2: Выбор датасета**\n"
            "Выберите папку с датасетом (из Шага 1). Путь и классы будут автоматически сохранены в `config.yaml`.\n\n"
            "**Шаг 2.5: (Опционально) Валидационный датасет**\n"
            "Укажите *отдельную* папку с датасетом для валидации. Если не указать, валидация пройдет на данных из Шага 2 (не рекомендуется).\n\n"
            "**Шаг 3 и 4: Настройка и запуск**\n\n"
            "**Параметры:**\n"
            "- **Количество эпох:** Сколько раз модель 'просмотрит' весь датасет. (Больше = дольше учится, но может переобучиться).\n"
            "- **Скорость обучения (LR):** Насколько 'сильно' модель меняет свои веса после каждой ошибки. (Слишком большая = не сойдется, слишком малая = вечное обучение).\n"
            "- **Размер батча:** Сколько изображений модель обрабатывает за один раз. (Больше = стабильнее, но требует больше VRAM).\n"
            "- **Аугментация:** Насколько сильно изменять изображения (повороты, яркость) во время обучения для устойчивости модели.\n\n"
            "**Запуск:**\n"
            "- **Обучить с нуля:** Начать обучение с самого начала.\n"
            "- **Дообучить модель:** Продолжить обучение с лучшей сохраненной точки (из папки `runs`)."
        )

        self.model_state_tooltip.text = (
            "**Анализ состояния обучения в реальном времени:**\n\n"
            "• **Нормальное (Обучается)** (Зеленый/Синий):\n"
            "  Train Loss (▼) и Val Loss (▼) падают. Val Loss близок к Train Loss.\n"
            "  *Решение: Отлично, продолжайте.*\n\n"
            "• **Переобучение (Overfitting)** (Желтый):\n"
            "  Train Loss (▼) падает, а Val Loss (▲) растет. Модель 'зубрит' данные.\n"
            "  *Решение: Остановите обучение. Используйте 'Дообучить' с меньшим LR, увеличьте аугментацию или добавьте больше данных.*\n\n"
            "• **Недообучение (Underfitting)** (Синий):\n"
            "  Train Loss (▼) все еще очень высокий. Модель слишком 'простая'.\n"
            "  *Решение: Дайте модели больше эпох или увеличьте LR.*\n\n"
            "• **Проблема (Loss растет)** (Красный):\n"
            "  Train Loss (▲) и Val Loss (▲) растут. Что-то не так.\n"
            "  *Решение: Остановите. Скорее всего, LR слишком высокий или данные 'битые'.*\n\n"
            "• **Схождение (Плато)** (Зеленый):\n"
            "  Val Loss (━) перестал падать. Модель достигла своего пика.\n"
            "  *Решение: Можно останавливать обучение.*\n"
        )

    def _start_timer(self):
        self.training_start_time = time.time()
        self.current_epoch = 0
        self.epochs_total = int(self.epochs_input.get())
        if self.timer_job:
            self.after_cancel(self.timer_job)
        self.eta_label.config(text="Осталось: --:--:--")
        self._update_elapsed_time()

    def _stop_timer(self):
        if self.timer_job:
            self.after_cancel(self.timer_job)
            self.timer_job = None
        self.training_start_time = None
        self.eta_label.config(text="Осталось: --:--:--")

    def _update_elapsed_time(self):
        if self.training_start_time:
            elapsed = time.time() - self.training_start_time
            self.timer_label.config(
                text=f"Время: {str(datetime.timedelta(seconds=int(elapsed)))}"
            )
            self.timer_job = self.after(1000, self._update_elapsed_time)

    def _update_eta(self):
        if (
                self.training_start_time
                and self.current_epoch > 1
                and self.epochs_total > 0
        ):
            elapsed = time.time() - self.training_start_time
            completed_epochs = self.current_epoch - 1
            time_per_epoch = elapsed / completed_epochs

            remaining_epochs = self.epochs_total - completed_epochs
            eta_seconds = int(remaining_epochs * time_per_epoch)

            self.eta_label.config(text=f"Осталось: {self._humanize_time(eta_seconds)}")

    def _set_ui_busy(self, is_busy: bool):
        for widget in self.interactive_widgets:
            with contextlib.suppress(tk.TclError):
                widget.config(state="disabled" if is_busy else "normal")

        self.btn_interrupt_training.config(state="normal" if is_busy else "disabled")

        if not is_busy:
            self._update_button_states()

    def _update_button_states(self):
        is_busy = self.training_thread and self.training_thread.is_alive()
        if is_busy:
            return

        model_ready = self.model_dir is not None and self.config_data is not None
        dataset_ready = self.dataset_path_from_labeling is not None

        self.btn_start_training.config(
            state="normal" if model_ready and dataset_ready else "disabled"
        )
        self.btn_continue_training.config(
            state="normal" if model_ready and dataset_ready else "disabled"
        )
        self.btn_preview_dataset.config(state="normal" if dataset_ready else "disabled")
        self.btn_interrupt_training.config(state="disabled")

    def create_new_model(self):
        model_name = simpledialog.askstring(
            "Создать новую модель", "Введите имя модели (например, Tomato_Detector_v1):"
        )
        if not model_name:
            return

        model_path = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(model_path):
            messagebox.showerror(
                "Ошибка", f"Папка с моделью '{model_name}' уже существует."
            )
            return

        try:
            os.makedirs(model_path)
            config_path = os.path.join(model_path, CONFIG_FILE)
            config_data = get_default_config()

            config_data["display_name"] = model_name

            save_config(config_data, config_path)
            messagebox.showinfo(
                "Успех",
                f"Создана папка для новой модели: {model_path}\n\nТеперь выберите эту папку.",
            )
            self.set_model_dir(model_path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать папку для модели: {e}")

    def select_model_dir(self):
        path = filedialog.askdirectory(
            title=f"Выберите папку с моделью (содержащую {CONFIG_FILE})"
        )
        if path:
            self.set_model_dir(path)

    def set_model_dir(self, path: str):
        config_path = os.path.join(path, CONFIG_FILE)
        if not os.path.exists(config_path):
            messagebox.showwarning(
                "Внимание", f"В выбранной папке отсутствует файл '{CONFIG_FILE}'."
            )
            self.model_dir = None
            self.config_data = None
        else:
            self.model_dir = path
            config = load_config(config_path)

            config_updated = False
            if "task" not in config:
                config["task"] = {}
                config_updated = True
            if "outputs" not in config["task"]:
                config["task"]["outputs"] = {}
                config_updated = True
            if "main_detection" not in config["task"]["outputs"]:
                config["task"]["outputs"]["main_detection"] = {
                    "type": "detection",
                    "classes": ["background", "plant"],
                }
                config_updated = True

            if "data" not in config:
                config["data"] = {}
            if "val_dataset_path" not in config["data"]:
                config["data"]["val_dataset_path"] = None
                config_updated = True

            if "data_yaml_path" not in config["data"]:
                config["data"]["data_yaml_path"] = None
                config_updated = True

            if config_updated:
                save_config(config, config_path)
                self._log_to_widget(
                    self.log_area,
                    f"INFO: Файл {CONFIG_FILE} был обновлен до новой структуры.",
                )

            self.config_data = config
            self.model_path_label.config(
                text=f"Выбрана модель: {os.path.basename(path)}"
            )
            self.epochs_input.set(
                self.config_data.get("training", {}).get("epochs", 100)
            )
            self.lr_input.delete(0, tk.END)
            self.lr_input.insert(
                0, str(self.config_data.get("training", {}).get("learning_rate", 0.001))
            )
            self.batch_size_input.set(
                self.config_data.get("training", {}).get("batch_size", 8)
            )

            self.val_dataset_path = self.config_data.get("data", {}).get(
                "val_dataset_path", None
            )
            if self.val_dataset_path:
                self.val_dataset_path_label.config(
                    text=f"Путь (Валидация): {os.path.basename(self.val_dataset_path)}"
                )
            else:
                self.val_dataset_path_label.config(
                    text="Валидационный датасет не выбран"
                )

            dataset_yaml_path = self.config_data.get("data", {}).get("data_yaml_path", None)
            if dataset_yaml_path and os.path.exists(dataset_yaml_path):
                # Если в конфиге уже был валидный путь, используем его
                self.set_dataset_path(os.path.dirname(dataset_yaml_path))
            elif self.dataset_path_from_labeling:
                # Иначе, если путь пришел из Вкладки 1, используем его
                self.set_dataset_path(self.dataset_path_from_labeling)

        self._update_button_states()

    def select_val_dataset_dir(self):
        path = filedialog.askdirectory(title="Выберите папку с валидационным датасетом")
        if path:
            self.val_dataset_path = path
            self.val_dataset_path_label.config(
                text=f"Путь (Валидация): {os.path.basename(path)}"
            )
        else:
            self.val_dataset_path = None
            self.val_dataset_path_label.config(text="Валидационный датасет не выбран")

        if self.config_data:
            self.config_data["data"]["val_dataset_path"] = self.val_dataset_path
            config_path = os.path.join(self.model_dir, CONFIG_FILE)
            save_config(self.config_data, config_path)
            self._log_to_widget(
                self.log_area, f"INFO: Путь валидации обновлен в {CONFIG_FILE}."
            )

    # (Он больше не нужен, т.к. мы читаем классы из data.yaml)
    # def _scan_dataset_for_classes(self, dataset_path: str) -> list:
    #     ...

    def set_dataset_path(self, path: str | None):
        """
        Устанавливает путь к датасету, находит data.yaml,
        обновляет конфиг модели (пути и классы).
        """
        self.dataset_path_from_labeling = path

        if not path:
            self.dataset_path_label.config(text=DEFAULT_DATASET_NAME)
            self._update_button_states()
            return

        text = f"Путь к датасету: {os.path.basename(path)}"
        self.dataset_path_label.config(text=text)

        if not self.model_dir:
            self._log_to_widget(self.log_area, "INFO: Датасет выбран. Выберите модель, чтобы обновить конфиг.")
            self._update_button_states()
            return

        try:
            # 1. Найти data.yaml
            data_yaml_path = os.path.join(path, DATA_YAML_FILE)
            if not os.path.exists(data_yaml_path):
                messagebox.showerror(
                    "Ошибка датасета",
                    f"Не удалось найти файл '{DATA_YAML_FILE}' в папке:\n{path}",
                    parent=self
                )
                self.dataset_path_from_labeling = None
                self.dataset_path_label.config(text=DEFAULT_DATASET_NAME)
                self._update_button_states()
                return

            # 2. Загрузить data.yaml, чтобы получить список классов
            with open(data_yaml_path, encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f)

            class_names = data_yaml.get("names")
            if not class_names:
                messagebox.showerror(
                    "Ошибка data.yaml",
                    f"Файл '{DATA_YAML_FILE}' не содержит списка классов ('names').",
                    parent=self
                )
                return

            # 3. Загрузить config.yaml модели
            config_path = os.path.join(self.model_dir, CONFIG_FILE)
            config = load_config(config_path)

            # 4. Обновить config.yaml
            config["data"]["dataset_path"] = path
            config["data"]["data_yaml_path"] = data_yaml_path

            self._log_to_widget(
                self.log_area, f"INFO: Путь к data.yaml в '{CONFIG_FILE}' обновлен."
            )

            # 5. Обновить классы в config.yaml
            if "main_detection" in config.get("task", {}).get("outputs", {}):
                config["task"]["outputs"]["main_detection"]["classes"] = class_names
                self._log_to_widget(
                    self.log_area,
                    f"INFO: Список классов в '{CONFIG_FILE}' обновлен. Найдено классов: {len(class_names)}",
                )
            else:
                self._log_to_widget(self.log_area,
                                    "WARNING: Структура 'main_detection' в конфиге не найдена. Классы не обновлены.")

            # 6. Сохранить config.yaml
            save_config(config, config_path)
            self.config_data = config

        except (OSError, yaml.YAMLError) as e:
            messagebox.showerror(
                "Ошибка data.yaml", f"Не удалось прочитать '{data_yaml_path}': {e}", parent=self
            )
        except Exception as e:
            messagebox.showerror(
                "Ошибка", f"Не удалось обновить '{CONFIG_FILE}': {e}", parent=self
            )

        self._update_button_states()


    def select_dataset_dir(self):
        path = filedialog.askdirectory(title="Выберите папку с готовым датасетом")
        if path:
            self.set_dataset_path(path)

    def preview_dataset(self):
        if not self.dataset_path_from_labeling or not os.path.isdir(
                self.dataset_path_from_labeling
        ):
            messagebox.showwarning(
                "Внимание", "Путь к датасету не указан или не существует."
            )
            return

        images_path = os.path.join(self.dataset_path_from_labeling, IMAGES_DIR)
        if not os.path.isdir(images_path):
            messagebox.showerror(
                "Ошибка", f"Папка с изображениями не найдена по пути:\n{images_path}"
            )
            return

        image_files = sorted(
            [
                f
                for f in os.listdir(images_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )[:10]

        if not image_files:
            messagebox.showinfo(
                "Предпросмотр", "В папке с изображениями нет файлов для предпросмотра."
            )
            return

        if self.preview_window and self.preview_window.winfo_exists():
            self.preview_window.lift()
            return

        self.preview_window = Toplevel(self.master)
        self.preview_window.title(
            f"Предпросмотр датасета ({os.path.basename(self.dataset_path_from_labeling)})"
        )
        self.preview_window.geometry("800x600")

        self.preview_images = []

        main_frame = ttk.Frame(self.preview_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        row, col = 0, 0
        max_cols = 4

        for img_file in image_files:
            try:
                img_path = os.path.join(images_path, img_file)
                img = Image.open(img_path)
                img.thumbnail((200, 200))
                photo_img = ImageTk.PhotoImage(img)
                self.preview_images.append(photo_img)

                img_label = ttk.Label(
                    scrollable_frame, image=photo_img, text=img_file, compound="bottom"
                )
                img_label.grid(row=row, column=col, padx=5, pady=5)

                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
            except Exception as e:
                print(f"Не удалось загрузить изображение {img_file}: {e}")

    def start_training(self, weights_path: str | None = None):
        if (
                not self.config_data
                or not self.dataset_path_from_labeling
                or not self.model_dir
        ):
            messagebox.showerror("Ошибка", "Сначала выберите модель и датасет.")
            return

        self.progress_bar["value"] = 0
        self.log_area.config(state="normal")
        self.log_area.delete("1.0", END)
        self.log_area.config(state="disabled")

        self.verbose_log_area.config(state="normal")
        self.verbose_log_area.delete("1.0", END)
        self.verbose_log_area.config(state="disabled")

        self.train_loss_label.config(text="Train Loss: --.----", bootstyle="default")
        self.val_loss_label.config(text="Val Loss: --.----", bootstyle="default")
        self.model_state_label.config(text="--", bootstyle="default")
        self.prev_train_loss = float("inf")
        self.prev_val_loss = float("inf")
        self.interrupt_event.clear()

        try:
            config_path = os.path.join(self.model_dir, CONFIG_FILE)
            if not os.path.exists(config_path):
                messagebox.showerror(
                    "Ошибка", f"Файл конфигурации не найден: {config_path}"
                )
                return

            current_config = load_config(config_path)

            current_config["training"]["epochs"] = int(self.epochs_input.get())
            current_config["training"]["learning_rate"] = float(self.lr_input.get())
            current_config["training"]["batch_size"] = int(self.batch_size_input.get())
            current_config["training"]["augmentation_level"] = (
                self.augmentation_level_combo.get()
            )

            if not current_config.get("data", {}).get("data_yaml_path"):
                messagebox.showerror(
                    "Ошибка",
                    f"Путь 'data_yaml_path' не установлен в {CONFIG_FILE}.\n"
                    "Пожалуйста, повторно выберите датасет (Шаг 2), чтобы конфиг обновился.",
                )
                return

            current_config["data"]["val_dataset_path"] = self.val_dataset_path
            current_config["training"]["save_dir"] = os.path.join(
                self.model_dir, RUNS_DIR
            )

            save_config(current_config, config_path)
            self.config_data = current_config

        except (ValueError, TypeError) as e:
            messagebox.showerror(
                "Ошибка параметров",
                f"Неверный формат одного из параметров обучения: {e}",
            )
            return

        self._start_timer()
        self.training_thread = CustomTrainingThread(
            self.config_data, weights_path, self.queue, self.interrupt_event
        )
        self.training_thread.start()
        self._set_ui_busy(True)

    def continue_training(self):
        if not self.model_dir:
            messagebox.showwarning(
                "Внимание", "Сначала выберите модель для дообучения."
            )
            return

        weights_path = None
        for root, _dirs, files in os.walk(self.model_dir):
            if BEST_MODEL_FILE in files:
                weights_path = os.path.join(root, BEST_MODEL_FILE)
                break

        if not weights_path:
            messagebox.showwarning(
                "Внимание",
                f"Не найден файл '{BEST_MODEL_FILE}' в папке модели: {self.model_dir}\n\nСначала обучите модель с нуля.",
            )
            return

        if messagebox.askyesno(
                "Дообучение",
                f"Найден файл с весами для дообучения:\n{weights_path}\n\nПродолжить?",
        ):
            self.start_training(weights_path)

    def request_training_stop(self):
        if not self.training_thread or not self.training_thread.is_alive():
            return

        answer = messagebox.askyesnocancel(
            "Прервать обучение?",
            "Вы уверены, что хотите прервать обучение?\n\n"
            "Да (Yes) - Остановить и сохранить текущий прогресс (если возможно).\n"
            "Нет (No) - Отмена, продолжить обучение.\n",
            parent=self,
        )

        if answer is True:
            self.interrupt_event.set()
            self._log_to_widget(self.log_area, "Запрошена остановка обучения...")
            self.training_status_label.config(text="Остановка...")
            self.btn_interrupt_training.config(
                state="disabled"
            )

        elif answer is False:
            pass

    def on_training_finished(
            self, success: bool, message: str, history: dict | None = None
    ):
        self._set_ui_busy(False)
        self._stop_timer()

        if not success:
            self.train_loss_label.config(
                text="Train Loss: --.----", bootstyle="default"
            )
            self.val_loss_label.config(text="Val Loss: --.----", bootstyle="default")
            self.model_state_label.config(text="--", bootstyle="default")

        if success:
            messagebox.showinfo(
                "Успех", f"Обучение успешно завершено!\nМодель сохранена: {message}"
            )

            if self.on_model_trained_callback:
                self.on_model_trained_callback(message)

            if history:
                self.show_loss_plot(history)
        else:
            if "Обучение прервано" in message:
                messagebox.showwarning(
                    "Остановлено", "Обучение было прервано пользователем."
                )
            else:
                messagebox.showerror("Ошибка", f"Ошибка во время обучения: {message}")

        self.training_status_label.config(text="Готов к работе.")
        self.progress_bar["value"] = 0

    def show_loss_plot(self, history: dict):
        if not history.get("train_loss"):
            self._log_to_widget(
                self.log_area,
                "ПРЕДУПРЕЖДЕНИЕ: История обучения пуста, график не будет построен.",
            )
            return

        try:
            plot_window = Toplevel(self)
            plot_window.title("График потерь (Loss)")
            plot_window.geometry("600x450")

            fig, ax = plt.subplots()
            ax.plot(history["train_loss"], label="Training Loss")
            if history.get("val_loss"):
                ax.plot(history["val_loss"], label="Validation Loss", linestyle="--")
            ax.set_xlabel("Эпохи")
            ax.set_ylabel("Потери (Loss)")
            ax.legend()
            ax.grid(True)

            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        except Exception as e:
            self._log_to_widget(self.log_area, f"Не удалось отобразить график: {e}")
