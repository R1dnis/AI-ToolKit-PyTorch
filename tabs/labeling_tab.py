import contextlib
import json
import os
import random

import re
import shutil
import tkinter as tk
import uuid
from tkinter import Listbox, filedialog, messagebox, simpledialog

import ttkbootstrap as ttk
import yaml

from ttkbootstrap.constants import DANGER, END, PRIMARY

from class_database import ClassManager
from constants import (
    AUTOSAVE_FILE,
    DATA_YAML_FILE,
    IMAGES_DIR,
    LABELS_DIR,
    SUPPORTED_IMAGE_FORMATS,
    VAL_IMAGES_DIR,
    VAL_LABELS_DIR,
)
from tooltip import ToolTip
from ui_components import PhotoViewer


# --- Классы Command, UndoStack, natural_sort_key остаются без изменений ---
class Command:
    """Абстрактный базовый класс для реализации паттерна 'Команда' (Undo/Redo)."""

    def undo(self): raise NotImplementedError

    def redo(self): raise NotImplementedError


class AddAnnotationCommand(Command):
    """Команда для добавления аннотации."""

    def __init__(self, tab_ref, image_name, annotation_data):
        self.tab = tab_ref
        self.image_name = image_name
        self.annotation_data = annotation_data

    def redo(self):
        if self.image_name not in self.tab.annotations_map:
            self.tab.annotations_map[self.image_name] = []
        self.tab.annotations_map[self.image_name].append(self.annotation_data)
        self.tab.redraw_scene_from_model()

    def undo(self):
        if self.image_name in self.tab.annotations_map and self.annotation_data in self.tab.annotations_map[
            self.image_name]:
            self.tab.annotations_map[self.image_name].remove(self.annotation_data)
        self.tab.redraw_scene_from_model()


class RemoveAnnotationCommand(Command):
    """Команда для удаления аннотации."""

    def __init__(self, tab_ref, image_name, annotation_data):
        self.tab = tab_ref
        self.image_name = image_name
        self.annotation_data = annotation_data

    def redo(self):
        if self.image_name in self.tab.annotations_map and self.annotation_data in self.tab.annotations_map[
            self.image_name]:
            self.tab.annotations_map[self.image_name].remove(self.annotation_data)
        self.tab.redraw_scene_from_model()

    def undo(self):
        if self.image_name not in self.tab.annotations_map:
            self.tab.annotations_map[self.image_name] = []
        self.tab.annotations_map[self.image_name].append(self.annotation_data)
        self.tab.redraw_scene_from_model()


class MoveResizeAnnotationCommand(Command):
    """Команда для перемещения или изменения размера аннотации."""

    def __init__(self, tab_ref, image_name, annotation_id, old_coords, new_coords):
        self.tab = tab_ref
        self.image_name = image_name
        self.annotation_id = annotation_id
        self.old_coords = old_coords
        self.new_coords = new_coords

    def _apply_coords(self, coords):
        if self.image_name in self.tab.annotations_map:
            for annotation in self.tab.annotations_map[self.image_name]:
                if annotation['id'] == self.annotation_id:
                    annotation['coords'] = coords
                    break
        self.tab.redraw_scene_from_model()

    def redo(self):
        self._apply_coords(self.new_coords)

    def undo(self):
        self._apply_coords(self.old_coords)


class UndoStack:
    """Простая реализация стека для операций Undo/Redo."""

    def __init__(self):
        self._undo_stack = []
        self._redo_stack = []

    def push(self, command):
        self._undo_stack.append(command)
        self._redo_stack.clear()

    def undo(self):
        if self._undo_stack:
            command = self._undo_stack.pop()
            command.undo()
            self._redo_stack.append(command)

    def redo(self):
        if self._redo_stack:
            command = self._redo_stack.pop()
            command.redo()
            self._undo_stack.append(command)

    def clear(self):
        self._undo_stack.clear()
        self._redo_stack.clear()


def natural_sort_key(s):
    """Ключ для естественной сортировки строк с числами."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]




class LabelingTab(ttk.Frame):
    """Основная вкладка приложения для создания и разметки датасетов."""
    settings_provider = None

    def __init__(self, parent=None, class_manager: ClassManager = None, on_dataset_created_callback: callable = None):
        super().__init__(parent)
        self.image_dir = None
        self.current_image_path = None
        self.annotations_map = {}
        self.annotations_save_path = None
        self.image_files = []
        self.current_image_index = -1

        self._autosave_job = None
        self._load_autosave_job = None

        # self.class_colors = {}

        self.undo_stack = UndoStack()
        self.active_drag_command = None

        # Принимаем ClassManager извне (Dependency Injection)
        if class_manager is None:
            # Обратная совместимость или ошибка, если main.py не обновлен
            print("ПРЕДУПРЕЖДЕНИЕ: ClassManager не был передан в LabelingTab. Создание локального экземпляра.")
            self.class_manager = ClassManager()
            self.class_manager.load_from_json()  # (Этап 3) Восстанавливаем, если DI не удался
        else:
            self.class_manager = class_manager

        self.on_dataset_created_callback = on_dataset_created_callback

        self.active_class_filter = None
        self.analysis_tab_ref = None

        self.val_split_var = tk.DoubleVar(value=20.0)

        self._init_ui()
        self._connect_signals()
        self._create_tooltips()
        self.update_class_sets_list()

        # Сохраняем ID таймера
        self._load_autosave_job = self.after(100, self._load_autosave)
        self._autosave_scheduler()

    def _init_ui(self):
        splitter = ttk.PanedWindow(self, orient="horizontal")
        splitter.pack(fill="both", expand=True)

        left_panel = self._create_left_panel()
        center_panel = self._create_center_panel()
        right_panel = self._create_right_panel()

        splitter.add(left_panel, weight=1)
        splitter.add(center_panel, weight=4)
        splitter.add(right_panel, weight=1)

    def _create_left_panel(self):
        panel = ttk.Frame(self, padding=10)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(2, weight=1)  # (8.2) Изменено для списка файлов

        folder_group_text_frame = ttk.Frame(panel)
        folder_group_text_frame.grid(row=0, column=0, sticky="ew")
        ttk.Label(folder_group_text_frame, text="1. Выбор папки и файла", font="-weight bold").grid(row=0, column=0,
                                                                                                    sticky="w")
        self.folder_help_tooltip = ToolTip(folder_group_text_frame, "")
        self.folder_help_tooltip.grid(row=0, column=1, padx=5, sticky="w")  # Добавлено sticky

        # (8.1) Текст изменен
        folder_group = ttk.LabelFrame(panel, text="Папка с изображениями (Источник)", padding=10)
        folder_group.grid(row=1, column=0, sticky="ew", pady=5)
        folder_group.columnconfigure(1, weight=1)
        self.btn_browse_folder = ttk.Button(folder_group, text="Выбрать папку...", command=self.browse_folder)
        self.btn_browse_folder.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.folder_path_label = ttk.Label(folder_group, text="Папка не выбрана", wraplength=250)
        self.folder_path_label.grid(row=0, column=1, sticky="ew")

        # (8.2) Добавлен список файлов
        list_group = ttk.LabelFrame(panel, text="Список файлов", padding=10)
        list_group.grid(row=2, column=0, sticky="nsew", pady=5)
        list_group.grid_rowconfigure(0, weight=1)
        list_group.grid_columnconfigure(0, weight=1)

        list_v_scroll = ttk.Scrollbar(list_group, orient="vertical")
        list_h_scroll = ttk.Scrollbar(list_group, orient="horizontal")

        self.file_list_widget = Listbox(
            list_group,
            yscrollcommand=list_v_scroll.set,
            xscrollcommand=list_h_scroll.set,
            exportselection=False,
            background="#343a40",  # Темный фон
            foreground="white",
            selectbackground="#0d6efd",  # Синий фон выделения
            selectforeground="white"
        )

        list_v_scroll.config(command=self.file_list_widget.yview)
        list_h_scroll.config(command=self.file_list_widget.xview)

        self.file_list_widget.grid(row=0, column=0, sticky="nsew")
        list_v_scroll.grid(row=0, column=1, sticky="ns")
        list_h_scroll.grid(row=1, column=0, columnspan=2, sticky="ew")

        return panel


    def _create_center_panel(self):
        panel = ttk.Frame(self, padding=10)
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        center_title_frame = ttk.Frame(panel)
        center_title_frame.grid(row=0, column=0, sticky="ew")
        ttk.Label(center_title_frame, text="Рабочая область разметки", font="-weight bold").grid(row=0, column=0,
                                                                                                 sticky="w")
        self.viewer_help_tooltip = ToolTip(center_title_frame, "")
        self.viewer_help_tooltip.grid(row=0, column=1, padx=5, sticky="w")  # Добавлено sticky

        self.viewer = PhotoViewer(panel, is_drawing_enabled=True)
        self.viewer.grid(row=1, column=0, sticky="nsew", pady=5)
        self.viewer.set_draw_mode('poly')

        toolbar = ttk.Frame(panel)
        toolbar.grid(row=2, column=0, sticky="ew", pady=5)
        toolbar.columnconfigure(2, weight=1)  # Добавляем вес для растягивания

        draw_mode_frame = ttk.Frame(toolbar)
        draw_mode_frame.grid(row=0, column=0, sticky="w")

        self.btn_undo = ttk.Button(draw_mode_frame, text="Отменить (Undo)", command=self.undo_stack.undo)
        self.btn_undo.pack(side="left",
                           padx=(0, 2))  # .pack() здесь допустим, т.к. родитель (draw_mode_frame) использует .grid()
        self.btn_redo = ttk.Button(draw_mode_frame, text="Повторить (Redo)", command=self.undo_stack.redo)
        self.btn_redo.pack(side="left", padx=(0, 10))  # .pack() здесь допустим

        self.show_all_annotations_var = tk.BooleanVar(value=True)
        self.btn_show_all_annotations = ttk.Checkbutton(toolbar, text="Показать все активные зоны",
                                                        variable=self.show_all_annotations_var,
                                                        command=self.toggle_show_all_annotations,
                                                        bootstyle="square-toggle")
        self.btn_show_all_annotations.grid(row=0, column=1, padx=20)

        nav_frame = ttk.Frame(panel)
        nav_frame.grid(row=3, column=0, sticky="ew", pady=5)
        nav_frame.grid_columnconfigure(1, weight=1)
        self.btn_prev_image = ttk.Button(nav_frame, text="<< Предыдущее", command=self.prev_image)
        self.btn_prev_image.grid(row=0, column=0, sticky="w")
        self.image_counter_label = ttk.Label(nav_frame, text="- / -")
        self.image_counter_label.grid(row=0, column=1, sticky="ew")
        self.btn_next_image = ttk.Button(nav_frame, text="Следующее >>", command=self.next_image)
        self.btn_next_image.grid(row=0, column=2, sticky="e")
        nav_frame.columnconfigure(1, weight=1)

        self.status_label = ttk.Label(panel, text="Выберите папку с изображениями и json-файлами разметки.",
                                      wraplength=400)
        self.status_label.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        return panel


    def _create_right_panel(self):
        panel = ttk.Frame(self, padding=10)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(2, weight=1)

        class_title_frame = ttk.Frame(panel)
        class_title_frame.grid(row=0, column=0, sticky="ew")
        ttk.Label(class_title_frame, text="2. Управление классами", font="-size 12 -weight bold").grid(row=0, column=0,
                                                                                                       sticky="w")
        self.class_help_tooltip = ToolTip(class_title_frame, "")
        self.class_help_tooltip.grid(row=0, column=1, padx=5, sticky="w")  # Добавлено sticky

        class_set_group = ttk.LabelFrame(panel, text="Набор классов", padding=10)
        class_set_group.grid(row=1, column=0, sticky="ew", pady=5)
        class_set_group.columnconfigure(0, weight=1)  # Добавлено
        self.class_set_combo = ttk.Combobox(class_set_group, state="readonly")
        self.class_set_combo.grid(row=0, column=0, sticky="ew", pady=5)
        btn_frame = ttk.Frame(class_set_group)
        btn_frame.grid(row=1, column=0, sticky="ew")
        btn_frame.columnconfigure(0, weight=1)  # Добавлено
        btn_frame.columnconfigure(1, weight=1)  # Добавлено
        btn_frame.columnconfigure(2, weight=1)  # Добавлено

        self.btn_new_set = ttk.Button(btn_frame, text="Новый", command=self.create_new_class_set)
        self.btn_new_set.grid(row=0, column=0, sticky="ew", padx=2)
        self.btn_rename_set = ttk.Button(btn_frame, text="Переименовать", command=self.rename_class_set)
        self.btn_rename_set.grid(row=0, column=1, sticky="ew", padx=2)
        self.btn_delete_set = ttk.Button(btn_frame, text="Удалить", command=self.delete_class_set)
        self.btn_delete_set.grid(row=0, column=2, sticky="ew", padx=2)

        tree_container = ttk.LabelFrame(panel, text="Список классов для разметки", padding=10)
        tree_container.grid(row=2, column=0, sticky="nsew", pady=5)
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)

        tree_v_scroll = ttk.Scrollbar(tree_container, orient="vertical")
        tree_h_scroll = ttk.Scrollbar(tree_container, orient="horizontal")
        self.class_tree_widget = ttk.Treeview(tree_container, columns=("class_name",), show="tree",
                                              yscrollcommand=tree_v_scroll.set,
                                              xscrollcommand=tree_h_scroll.set)
        tree_v_scroll.config(command=self.class_tree_widget.yview)
        tree_h_scroll.config(command=self.class_tree_widget.xview)
        self.class_tree_widget.heading("#0", text="Название класса")
        tree_container.bind("<Configure>", self._on_tree_configure)
        self.class_tree_widget.grid(row=0, column=0, sticky="nsew")
        tree_v_scroll.grid(row=0, column=1, sticky="ns")
        tree_h_scroll.grid(row=1, column=0, columnspan=2, sticky="ew")

        save_title_frame = ttk.Frame(panel)
        save_title_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(save_title_frame, text="3. Сохранение", font="-weight bold").grid(row=0, column=0, sticky="w")
        self.save_help_tooltip = ToolTip(save_title_frame, "")
        self.save_help_tooltip.grid(row=0, column=1, padx=5, sticky="w")  # Добавлено sticky

        save_group = ttk.LabelFrame(panel, text="Путь для сохранения готового датасета", padding=10)
        save_group.grid(row=4, column=0, sticky="ew", pady=5)
        save_group.columnconfigure(1, weight=1)  # Добавлено
        self.btn_set_save_path = ttk.Button(save_group, text="Указать папку...", command=self.browse_save_path)
        self.btn_set_save_path.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.save_path_label = ttk.Label(save_group, text="Путь не указан", wraplength=250)
        self.save_path_label.grid(row=0, column=1, sticky="ew")

        split_group = ttk.LabelFrame(panel, text="Разделение датасета", padding=10)
        split_group.grid(row=5, column=0, sticky="ew", pady=5)
        split_group.columnconfigure(1, weight=1)

        ttk.Label(split_group, text="Процент валидации (%):").grid(row=0, column=0, sticky="w", pady=2, padx=(0, 5))
        self.val_split_spinbox = ttk.Spinbox(split_group, from_=0.0, to=100.0, textvariable=self.val_split_var,
                                             width=8)
        self.val_split_spinbox.grid(row=0, column=1, sticky="w", pady=2)

        self.btn_save_annotations = ttk.Button(panel, text="Сохранить датасет", bootstyle=PRIMARY)
        self.btn_save_annotations.grid(row=6, column=0, sticky="ew", pady=5)
        self.btn_delete_selected = ttk.Button(panel, text="Удалить выделенные аннотации", bootstyle=DANGER,
                                              command=self.delete_selected)
        self.btn_delete_selected.grid(row=7, column=0, sticky="ew", pady=10)
        return panel


    def _on_tree_configure(self, event):
        new_width = event.width - 20
        self.class_tree_widget.column("#0", width=new_width)

    def _create_tooltips(self):
        self.folder_help_tooltip.text = (
            "**Шаг 1: Выбор папки и файла**\n\n"
            "1. Нажмите 'Выбрать папку...' и укажите директорию, где лежат изображения (`.jpg`, `.png`...).\n"
            "2. В 'Списке файлов' выберите изображение для разметки.\n\n"
            "**Автосохранение:** Ваш прогресс разметки автоматически сохраняется. Если приложение закроется, при следующем запуске оно предложит восстановить сессию."
        )
        self.viewer_help_tooltip.text = (
            "**Рабочая область**\n\n"
            "1. **Выберите класс** справа.\n"
            "2. **Начните рисовать полигон** на изображении, кликая левой кнопкой мыши для добавления вершин.\n"
            "3. **Завершите полигон**, кликнув правой кнопкой мыши.\n\n"
            "**Управление:**\n"
            "- **Клик ЛКМ по полигону:** Выделить.\n"
            "- **Двойной клик ЛКМ по ребру:** Добавить вершину.\n"
            "- **Перетаскивание ЛКМ:** Переместить полигон или вершину.\n"
            "- **Колесо мыши:** Масштабировать.\n"
            "- **Средняя кнопка мыши (колесо):** Панорамировать.\n"
            "- **Кнопки 'Отменить'/'Повторить':** Отменяют или повторяют последнее действие.\n"
            "- **Горячие клавиши:**\n"
            "  - **Ctrl+Z:** Отменить\n"
            "  - **Ctrl+Shift+Z / Ctrl+Y:** Повторить\n"
            "  - **Delete:** Удалить выделенную аннотацию"
        )
        self.class_help_tooltip.text = (
            "**Шаг 2: Управление классами**\n\n"
            "Классы - это категории объектов для разметки (например, 'лист', 'стебель', 'плод').\n\n"
            "1. **Создайте 'Набор классов'** для вашего проекта.\n"
            "2. **Кликните Правой Кнопкой Мыши** по списку для добавления, переименования или удаления классов."
        )
        self.save_help_tooltip.text = (
            "**Шаг 3: Сохранение датасета**\n\n"
            "Когда все изображения размечены, укажите папку и сохраните результат.\n\n"
            "Укажите процент данных для **валидации** (например, 20%).\n\n"
            f"Программа создаст структуру (`{IMAGES_DIR}/`, `{LABELS_DIR}/`, `{VAL_IMAGES_DIR}/`, `{VAL_LABELS_DIR}/`) и файл `{DATA_YAML_FILE}`, готовый для обучения. После успешного сохранения файл автосохранения будет удален."
        )

    def _update_image_counter(self):
        if self.image_files:
            total = len(self.image_files)
            current = self.current_image_index + 1
            self.image_counter_label.config(text=f"{current} / {total}")
        else:
            self.image_counter_label.config(text="- / -")

    def _connect_signals(self):
        self.viewer.bind_event('annotation_added', self.on_annotation_added)
        self.viewer.bind_event('annotation_modified', self.on_annotation_modified)
        self.viewer.bind_event('annotation_press', self.on_annotation_press)
        self.btn_save_annotations.config(command=self.save_annotations)
        self.class_tree_widget.bind("<Button-3>", self.open_class_menu)
        self.class_tree_widget.bind("<<TreeviewSelect>>", self.on_class_selected_for_view_filter)

        # (8.2) Привязка выбора файла в списке
        self.file_list_widget.bind("<<ListboxSelect>>", self.on_file_selected_from_listbox)

        self.bind_all("<Control-z>", lambda e: self.undo_stack.undo())
        self.bind_all("<Control-Shift-Z>", lambda e: self.undo_stack.redo())
        self.bind_all("<Control-y>", lambda e: self.undo_stack.redo())
        self.bind_all("<Delete>", lambda e: self.delete_selected())
        self.class_set_combo.bind("<<ComboboxSelected>>", self.on_class_set_selected)


    def _autosave_scheduler(self):
        self._autosave_data()
        self._autosave_job = self.after(5000, self._autosave_scheduler)

    def _autosave_data(self):
        if not self.image_dir:
            return
        try:
            data_to_save = {
                'image_dir': self.image_dir,
                'current_image_path': self.current_image_path,
                'annotations': self.annotations_map,
            }
            with open(AUTOSAVE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка автосохранения: {e}")

    def _load_autosave(self):
        self._load_autosave_job = None

        if os.path.exists(AUTOSAVE_FILE):
            if messagebox.askyesno("Восстановление сессии",
                                   "Найден файл предыдущей сессии. Хотите восстановить прогресс?"):
                try:
                    with open(AUTOSAVE_FILE, encoding='utf-8') as f:
                        saved_data = json.load(f)

                    self.annotations_map = saved_data.get('annotations', {})
                    saved_dir = saved_data.get('image_dir')
                    saved_image = saved_data.get('current_image_path')

                    if saved_dir and os.path.isdir(saved_dir):
                        self.on_directory_selected(saved_dir, restore_path=saved_image)
                    self.status_label.config(text="Предыдущая сессия успешно восстановлена.")

                except (OSError, json.JSONDecodeError) as e:
                    messagebox.showerror(
                        "Ошибка восстановления",
                        f"Не удалось загрузить данные из файла автосохранения '{AUTOSAVE_FILE}'.\n\n"
                        f"Ошибка: {e}\n\n"
                        "Файл будет удален, чтобы приложение могло запуститься."
                    )
                    try:
                        os.remove(AUTOSAVE_FILE)
                    except OSError as remove_e:
                        messagebox.showerror("Критическая ошибка",
                                             f"Не удалось удалить поврежденный файл автосохранения: {remove_e}")
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось загрузить данные из файла автосохранения: {e}")
                    with contextlib.suppress(OSError):
                        os.remove(AUTOSAVE_FILE)
            else:
                try:
                    os.remove(AUTOSAVE_FILE)
                except OSError as e:
                    messagebox.showwarning("Ошибка", f"Не удалось удалить файл автосохранения: {e}")


    def destroy(self):
        """Очистка таймеров перед уничтожением виджета."""
        if self._autosave_job:
            self.after_cancel(self._autosave_job)
        if self._load_autosave_job:
            self.after_cancel(self._load_autosave_job)
        super().destroy()

    # ---------------------------------------------------------------

    def browse_folder(self):
        path = filedialog.askdirectory(title="Выберите папку с изображениями и .json разметкой")
        if path:
            if self.image_dir and self.image_dir != path and os.path.exists(AUTOSAVE_FILE):
                with contextlib.suppress(OSError):
                    os.remove(AUTOSAVE_FILE)
            self.on_directory_selected(path)

    def on_directory_selected(self, path, restore_path=None):
        if os.path.isdir(path):
            self.image_dir = path
            self.folder_path_label.config(text=os.path.basename(path))
            try:
                all_files = os.listdir(self.image_dir)
                image_files_unsorted = [f for f in all_files if f.lower().endswith(SUPPORTED_IMAGE_FORMATS)]

                file_limit = 10000
                if len(image_files_unsorted) > file_limit:
                    if not messagebox.askyesno(
                            "Предупреждение о производительности",
                            f"Найдено более {len(image_files_unsorted)} изображений.\n\n"
                            f"Загрузка такого большого списка может занять много времени и замедлить работу интерфейса.\n\n"
                            f"Хотите загрузить только первые {file_limit} файлов?"
                    ):
                        # Пользователь выбрал "Нет" (загрузить все)
                        self.status_label.config(text=f"Загрузка {len(image_files_unsorted)} файлов...")
                        self.update_idletasks()  # Обновляем UI перед возможным фризом
                    else:
                        # Пользователь выбрал "Да" (обрезать список)
                        image_files_unsorted = image_files_unsorted[:file_limit]

                self.image_files = sorted(image_files_unsorted, key=natural_sort_key)

                # (8.2) Обновляем Listbox
                self._update_file_listbox()

                self.status_label.config(text=f"Найдено изображений: {len(self.image_files)}")

                if self.image_files:
                    target_index = 0
                    if restore_path and os.path.basename(restore_path) in self.image_files:
                        try:
                            target_index = self.image_files.index(os.path.basename(restore_path))
                        except ValueError:
                            target_index = 0
                    self.load_image_by_index(target_index)
                else:
                    self.current_image_index = -1
                    self.current_image_path = None
                    self.viewer.clear_canvas()
                    self._update_image_counter()
                    self.status_label.config(text="Изображения не найдены в выбранной папке.")
            except PermissionError:
                self.status_label.config(text=f"Ошибка доступа к папке: {path}")
                messagebox.showerror("Ошибка доступа", f"Нет прав на чтение содержимого папки:\n{path}")
            except Exception as e:
                messagebox.showerror("Ошибка загрузки папки", f"Не удалось прочитать папку: {e}")


    def _update_file_listbox(self):
        """Очищает и заполняет Listbox списком файлов."""
        self.file_list_widget.delete(0, END)
        if not self.image_files:
            self.file_list_widget.insert(END, "Файлы не найдены")
            return

        for file_name in self.image_files:
            self.file_list_widget.insert(END, file_name)


    def on_file_selected_from_listbox(self, event):
        """Обработчик выбора файла из Listbox."""
        selected_indices = self.file_list_widget.curselection()
        if not selected_indices:
            return

        new_index = selected_indices[0]
        if new_index != self.current_image_index:
            self.load_image_by_index(new_index)


    def load_image_by_index(self, index):
        if not (0 <= index < len(self.image_files)):
            return

        self.current_image_index = index
        image_name = self.image_files[self.current_image_index]
        self.current_image_path = os.path.join(self.image_dir, image_name)

        self.viewer.set_photo(self.current_image_path)
        self.undo_stack.clear()
        self.after(50, self.redraw_scene_from_model)
        self._update_image_counter()

        # (8.2) Синхронизация Listbox
        try:
            self.file_list_widget.selection_clear(0, END)
            self.file_list_widget.selection_set(index)
            self.file_list_widget.activate(index)
            self.file_list_widget.see(index)
        except tk.TclError:
            # Может произойти, если виджет еще не полностью готов
            print(f"Не удалось обновить выделение в Listbox для индекса {index}")


    def redraw_scene_from_model(self):
        image_name = os.path.basename(self.current_image_path) if self.current_image_path else None
        if not image_name:
            self.viewer.redraw_annotations_from_model([])
            return

        annotations_for_image = self.annotations_map.get(image_name, [])
        if self.active_class_filter:
            display_annotations = [ann for ann in annotations_for_image if
                                   ann['class_name'] == self.active_class_filter]
        else:
            display_annotations = annotations_for_image

        all_class_colors = self.class_manager.get_class_colors()
        for ann_data in display_annotations:
            class_name = ann_data['class_name']
            ann_data['color'] = all_class_colors.get(class_name, '#FF0000')  # Красный по умолчанию

        self.viewer.redraw_annotations_from_model(display_annotations)

    def on_annotation_added(self, annotation_type, coords):
        selected_item_id = self.class_tree_widget.focus()
        if not selected_item_id:
            messagebox.showwarning("Внимание", "Пожалуйста, сначала выберите класс в списке справа.")
            self.redraw_scene_from_model()
            return
        class_name = self.class_tree_widget.item(selected_item_id, 'text')
        image_name = os.path.basename(self.current_image_path)
        annotation_data = {'id': str(uuid.uuid4()), 'class_name': class_name, 'type': annotation_type, 'coords': coords}
        command = AddAnnotationCommand(self, image_name, annotation_data)
        command.redo()
        self.undo_stack.push(command)

    def on_annotation_modified(self, annotation_id, new_coords):
        if self.active_drag_command and self.active_drag_command.annotation_id == annotation_id:

            # (Фикс 8.3) START: Немедленно применяем изменение в модели данных,
            # иначе оно будет потеряно при переключении изображения.
            image_name = os.path.basename(self.current_image_path)
            if image_name in self.annotations_map:
                for ann in self.annotations_map[image_name]:
                    if ann['id'] == annotation_id:
                        ann['coords'] = new_coords
                        break
            # (Фикс 8.3) END

            # Теперь, когда модель обновлена, мы можем безопасно
            # положить команду в стек отмены.
            self.active_drag_command.new_coords = new_coords
            if self.active_drag_command.old_coords != self.active_drag_command.new_coords:
                self.undo_stack.push(self.active_drag_command)

        self.active_drag_command = None


    def on_annotation_press(self, annotation_id):
        image_name = os.path.basename(self.current_image_path)
        old_coords = None
        for ann in self.annotations_map.get(image_name, []):
            if ann['id'] == annotation_id:
                old_coords = ann['coords']
                break
        if old_coords is not None:
            self.active_drag_command = MoveResizeAnnotationCommand(self, image_name, annotation_id, old_coords,
                                                                   old_coords)

    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.load_image_by_index(self.current_image_index + 1)

    def prev_image(self):
        if self.current_image_index > 0:
            self.load_image_by_index(self.current_image_index - 1)

    def open_class_menu(self, event):
        menu = tk.Menu(self, tearoff=0)
        item_id = self.class_tree_widget.identify_row(event.y)
        if item_id:
            self.class_tree_widget.selection_set(item_id)
            menu.add_command(label="Добавить подкласс", command=lambda: self.add_class(parent_item_id=item_id))
            menu.add_command(label="Переименовать", command=lambda: self.rename_class(item_id))
            menu.add_separator()
            menu.add_command(label="Удалить", command=lambda: self.delete_class(item_id))
        else:
            menu.add_command(label="Добавить класс", command=self.add_class)
        menu.tk_popup(event.x_root, event.y_root)

    def add_class(self, parent_item_id=None):
        active_set = self.class_set_combo.get()
        if not active_set:
            messagebox.showwarning("Внимание", "Сначала создайте или выберите набор классов.")
            return
        text = simpledialog.askstring("Новый класс", "Введите название класса:")
        if text:
            parent_name = self.class_tree_widget.item(parent_item_id, 'text') if parent_item_id else None
            success = self.class_manager.add_class(active_set, text, parent_name)
            if success:
                self.update_class_tree()
            else:
                error_subject = "Подкласс" if parent_name else "Класс"
                messagebox.showerror("Ошибка создания",
                                     f"{error_subject} с именем '{text}' уже существует в этом наборе.")

    def rename_class(self, item_id):
        active_set = self.class_set_combo.get()
        old_name = self.class_tree_widget.item(item_id, 'text')
        new_name = simpledialog.askstring("Переименовать", "Новое имя:", initialvalue=old_name)

        if new_name and new_name != old_name:
            success = self.class_manager.rename_class(active_set, old_name, new_name)
            if success:
                for annotations in self.annotations_map.values():
                    for ann in annotations:
                        if ann['class_name'] == old_name:
                            ann['class_name'] = new_name
                self.update_class_tree()
                self.redraw_scene_from_model()
            else:
                messagebox.showerror("Ошибка переименования",
                                     f"Класс с именем '{new_name}' уже существует в этом наборе.")

    def delete_class(self, item_id):
        active_set = self.class_set_combo.get()
        class_name = self.class_tree_widget.item(item_id, 'text')
        if messagebox.askyesno("Удалить", f"Удалить класс '{class_name}' и все его разметки?"):
            self.class_manager.delete_class(active_set, class_name)
            for image_name in list(self.annotations_map.keys()):
                self.annotations_map[image_name] = [a for a in self.annotations_map[image_name] if
                                                    a['class_name'] != class_name]
            self.update_class_tree()
            self.redraw_scene_from_model()

    def delete_selected(self):
        selected_ids = self.viewer.get_selected_ids()
        if not selected_ids:
            return
        annotation_id = selected_ids[0]
        image_name = os.path.basename(self.current_image_path)
        annotation_data = None
        if image_name in self.annotations_map:
            for ann in self.annotations_map[image_name]:
                if ann['id'] == annotation_id:
                    annotation_data = ann
                    break
        if annotation_data:
            command = RemoveAnnotationCommand(self, image_name, annotation_data)
            command.redo()
            self.undo_stack.push(command)

    def get_class_names(self):
        active_set = self.class_set_combo.get()
        if not active_set:
            return []
        return self.class_manager.get_all_class_names(active_set)

    # (Логика перенесена в class_manager.get_class_colors())
    # def _update_class_colors(self):
    #     ...

    def save_annotations(self):
        if not self.annotations_save_path:
            messagebox.showerror("Ошибка", "Не выбран путь для сохранения датасета.")
            return
        class_names = self.get_class_names()
        if not class_names:
            messagebox.showwarning("Ошибка", "Список классов пуст. Добавьте классы перед сохранением.")
            return

        try:
            train_labels_dir = os.path.join(self.annotations_save_path, LABELS_DIR)
            train_images_dir = os.path.join(self.annotations_save_path, IMAGES_DIR)
            os.makedirs(train_labels_dir, exist_ok=True)
            os.makedirs(train_images_dir, exist_ok=True)

            val_labels_dir = os.path.join(self.annotations_save_path, VAL_LABELS_DIR)
            val_images_dir = os.path.join(self.annotations_save_path, VAL_IMAGES_DIR)
            os.makedirs(val_labels_dir, exist_ok=True)
            os.makedirs(val_images_dir, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Ошибка создания папок", f"Не удалось создать структуру папок датасета:\n{e}")
            return

        val_percentage = self.val_split_var.get() / 100.0
        if not (0.0 <= val_percentage <= 1.0):
            messagebox.showerror("Ошибка", "Процент валидации должен быть от 0 до 100.")
            return

        all_image_files = self.image_files.copy()
        random.seed(42)
        random.shuffle(all_image_files)

        val_count = int(len(all_image_files) * val_percentage)
        val_files = set(all_image_files[:val_count])
        train_files = set(all_image_files[val_count:])

        if not train_files:
            messagebox.showwarning("Внимание", "Нет файлов для обучающего набора. Проверьте процент валидации.")
        if val_percentage > 0 and not val_files:
            messagebox.showwarning("Внимание",
                                   "Нет файлов для валидационного набора, хотя процент > 0. Возможно, изображений слишком мало.")

        progress_popup = self._create_progress_popup("Сохранение...", len(self.image_files))
        self.update_idletasks()

        try:
            for index, image_name in enumerate(self.image_files):
                progress_popup.update_progress(index + 1)
                self.update_idletasks()
                source_img_path = os.path.join(self.image_dir, image_name)
                if not os.path.exists(source_img_path):
                    print(f"Предупреждение: Файл изображения не найден, пропуск: {source_img_path}")
                    continue

                base_filename_without_ext, _ = os.path.splitext(image_name)
                new_base_filename = image_name

                if image_name in val_files:
                    dest_img_dir = val_images_dir
                    dest_label_dir = val_labels_dir
                else:
                    dest_img_dir = train_images_dir
                    dest_label_dir = train_labels_dir

                dest_img_path = os.path.join(dest_img_dir, new_base_filename)

                # (9.2) Обертываем I/O операции
                try:
                    shutil.copy2(source_img_path, dest_img_path)
                except OSError as e:
                    print(f"Не удалось скопировать изображение {image_name}: {e}")
                    # Пропускаем этот файл, но продолжаем
                    continue

                annotations_data_list = self.annotations_map.get(image_name, [])
                label_path = os.path.join(dest_label_dir, f"{base_filename_without_ext}.json")

                cleaned_annotations = []
                for ann in annotations_data_list:
                    ann_copy = ann.copy()
                    ann_copy.pop('id', None)
                    ann_copy.pop('color', None)
                    cleaned_annotations.append(ann_copy)

                full_data = {"image_path": new_base_filename,
                             "annotations": cleaned_annotations}

                with open(label_path, 'w', encoding='utf-8') as f:
                    json.dump(full_data, f, indent=4, ensure_ascii=False)

            final_class_names = sorted(class_names)
            if "background" in final_class_names:
                final_class_names.remove("background")
            final_class_names.insert(0, "background")

            data_yaml = {'path': os.path.abspath(self.annotations_save_path),
                         'train': IMAGES_DIR,
                         'val': VAL_IMAGES_DIR,
                         'nc': len(final_class_names),  # (Этап 11) Используем обновленный список
                         'names': final_class_names}  # (Этап 11) Используем обновленный список

            with open(os.path.join(self.annotations_save_path, DATA_YAML_FILE), 'w', encoding='utf-8') as f:
                yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

        except OSError as e:
            # (9.2) Ловим ошибки записи/копирования (например, диск полон)
            if progress_popup:
                progress_popup.destroy()
            messagebox.showerror("Ошибка сохранения",
                                 f"Произошла ошибка I/O во время сохранения датасета:\n{e}\n\n"
                                 "Возможно, закончилось место на диске или нет прав на запись.")
            return  # Прерываем выполнение
        except Exception as e:
            # (9.2) Ловим другие непредвиденные ошибки
            if progress_popup:
                progress_popup.destroy()
            messagebox.showerror("Неизвестная ошибка",
                                 f"Произошла непредвиденная ошибка:\n{e}")
            return  # Прерываем выполнение
        finally:
            if progress_popup:
                progress_popup.destroy()

        if os.path.exists(AUTOSAVE_FILE):
            try:
                os.remove(AUTOSAVE_FILE)
            except OSError as e:
                print(f"Не удалось удалить файл автосохранения: {e}")

        messagebox.showinfo("Успех", "Датасет успешно сохранен (Train/Val).")

        if self.on_dataset_created_callback:
            self.on_dataset_created_callback(self.annotations_save_path)


    def _create_progress_popup(self, title, max_value):
        popup = tk.Toplevel(self)
        popup.title(title)
        popup.transient(self)
        popup.grab_set()
        popup.geometry("300x100")
        ttk.Label(popup, text="Пожалуйста, подождите...").pack(pady=10)
        progress = ttk.Progressbar(popup, length=280, mode='determinate', maximum=max_value)
        progress.pack(pady=5)

        def update_progress(value):
            progress['value'] = value
            popup.update_idletasks()

        popup.update_progress = update_progress
        return popup

    def browse_save_path(self):
        path = filedialog.askdirectory(title="Выберите папку для сохранения датасета")
        if path:
            self.set_annotations_save_path(path, emit_signal=False)

    def set_annotations_save_path(self, path: str, emit_signal=True):
        if path and os.path.isdir(path):
            self.annotations_save_path = path
            self.save_path_label.config(text=path)
            self.btn_save_annotations.config(state="normal")

            if emit_signal and self.on_dataset_created_callback:
                self.on_dataset_created_callback(path)

        elif path:
            messagebox.showwarning("Неверный путь", f"Указанный путь не является существующей директорией:\n{path}")
            self.btn_save_annotations.config(state="disabled")
        else:
            self.annotations_save_path = None
            self.save_path_label.config(text="Путь не указан")
            self.btn_save_annotations.config(state="disabled")

    def update_class_sets_list(self):
        sets = self.class_manager.get_class_set_names()
        self.class_set_combo['values'] = sets
        if sets:
            self.class_set_combo.set(sets[0])
            self.update_class_tree()
        else:
            self.class_set_combo.set('')
            self.update_class_tree()

    def on_class_set_selected(self, event):
        self.update_class_tree()

    def update_class_tree(self):
        self.class_tree_widget.delete(*self.class_tree_widget.get_children())
        active_set = self.class_set_combo.get()
        if not active_set:
            return
        class_structure = self.class_manager.get_class_set(active_set)

        def insert_items(parent_id, classes):
            for class_name, details in classes.items():
                new_id = self.class_tree_widget.insert(parent_id, "end", text=class_name, open=True)
                if details and 'children' in details and details['children']:
                    insert_items(new_id, details['children'])

        insert_items("", class_structure)

        # _update_class_colors() был удален, но нам все равно нужно
        # перерисовать сцену, если классы (а значит и цвета) изменились.
        self.redraw_scene_from_model()

    def on_class_selected_for_view_filter(self, event):
        selected_item_id = self.class_tree_widget.focus()
        if selected_item_id:
            class_name = self.class_tree_widget.item(selected_item_id, 'text')
            self.active_class_filter = class_name
            self.show_all_annotations_var.set(False)
        else:
            self.active_class_filter = None
            self.show_all_annotations_var.set(True)
        self.redraw_scene_from_model()

    def toggle_show_all_annotations(self):
        if self.show_all_annotations_var.get():
            self.active_class_filter = None
            selection = self.class_tree_widget.selection()
            if selection:
                self.class_tree_widget.selection_remove(selection)
        else:
            selected_item_id = self.class_tree_widget.focus()
            if selected_item_id:
                self.active_class_filter = self.class_tree_widget.item(selected_item_id, 'text')
            else:
                self.show_all_annotations_var.set(True)
                self.active_class_filter = None
        self.redraw_scene_from_model()

    def show_all_annotations(self):
        self.active_class_filter = None
        self.show_all_annotations_var.set(True)
        selection = self.class_tree_widget.selection()
        if selection:
            self.class_tree_widget.selection_remove(selection)
        self.redraw_scene_from_model()

    def create_new_class_set(self):
        set_name = simpledialog.askstring("Новый набор классов", "Введите имя нового набора:")
        if set_name:
            success = self.class_manager.add_class_set(set_name)
            if success:
                self.update_class_sets_list()
                self.class_set_combo.set(set_name)
                self.update_class_tree()
            else:
                messagebox.showerror("Ошибка", f"Набор с именем '{set_name}' уже существует.")

    def rename_class_set(self):
        old_name = self.class_set_combo.get()
        if not old_name:
            return
        new_name = simpledialog.askstring("Переименовать набор", "Введите новое имя:", initialvalue=old_name)
        if new_name and new_name != old_name:
            success = self.class_manager.rename_class_set(old_name, new_name)
            if success:
                self.update_class_sets_list()
                self.class_set_combo.set(new_name)
            else:
                messagebox.showerror("Ошибка", f"Набор с именем '{new_name}' уже существует.")

    def delete_class_set(self):
        set_name = self.class_set_combo.get()
        if not set_name:
            return
        if messagebox.askyesno("Удалить набор", f"Вы уверены, что хотите удалить набор классов '{set_name}'?"):
            classes_to_delete = self.class_manager.get_all_class_names(set_name)
            success = self.class_manager.delete_class_set(set_name)
            if success:
                for image_name in list(self.annotations_map.keys()):
                    self.annotations_map[image_name] = [
                        ann for ann in self.annotations_map[image_name]
                        if ann.get('class_name') not in classes_to_delete
                    ]
                self.update_class_sets_list()
                if self.class_set_combo['values']:
                    self.class_set_combo.set(self.class_set_combo['values'][0])
                else:
                    self.class_set_combo.set('')
                self.update_class_tree()
                self.redraw_scene_from_model()
            else:
                messagebox.showerror("Ошибка", f"Не удалось удалить набор '{set_name}'.")
