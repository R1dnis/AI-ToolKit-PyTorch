import math
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk, ImageOps


RESIZE_HANDLE_SIZE = 8
MIN_BBOX_SIZE = 5


class PhotoViewer(ttk.Frame):
    """
    Интерактивный виджет для просмотра изображений и создания/редактирования аннотаций.
    (Этап 12) Внедрена "Честная камера" для исправления дрейфа зума.
    """

    def __init__(self, parent, is_drawing_enabled=False):
        super().__init__(parent)
        self.parent = parent
        self.canvas = tk.Canvas(self, bg="gray20", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.image_id = None
        self.image = None  # Оригинал (full-res)
        self.photo_image = None  # Отображаемый кроп

        self.scale_factor = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        # Эти переменные больше не нужны для новой математики,
        # но оставим их, если вдруг понадобятся для дебага.
        # self.viewport_img_x1 = 0
        # self.viewport_img_y1 = 0
        # self.viewport_scale_x = 1.0
        # self.viewport_scale_y = 1.0

        self._zoom_job = None
        self.render_job = None  # (Этап 10) Debouncer

        self._is_drawing_enabled = is_drawing_enabled
        self.item_map = {}  # {ann_id: {"data": {...}, "canvas_items": [poly_id, text_bg, text_id]}}
        self.selected_item_id = None
        self.resize_handles = {}
        self.action_state = {
            "action": None,
            "start_pos": None,
            "item_id": None,
            "handle": None,
            "temp_coords": None,
        }

        self.draw_mode = "rect"
        self.current_poly_points = []
        self.current_poly_preview_id = None

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right_mouse_press)

        self.canvas.bind("<Double-Button-1>", self.on_add_vertex)

        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)

        self.canvas.bind("<ButtonPress-2>", self.start_pan_v2)
        self.canvas.bind("<B2-Motion>", self.pan_image_v2)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan_v2)

        self.canvas.bind("<Motion>", self.update_cursor)

        self.event_handlers = {}

    def bind_event(self, event_name, handler):
        self.event_handlers[event_name] = handler

    def _fire_event(self, event_name, *args, **kwargs):
        if event_name in self.event_handlers:
            self.event_handlers[event_name](*args, **kwargs)

    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        """Преобразует координаты холста (вид) в координаты изображения (мир)."""
        # Формула: world = (screen - offset) / scale
        img_x = (canvas_x - self.offset_x) / self.scale_factor
        img_y = (canvas_y - self.offset_y) / self.scale_factor
        return img_x, img_y

    def _image_to_canvas_coords(self, img_x, img_y):
        """Преобразует координаты изображения (мир) в координаты холста (вид)."""
        # Формула: screen = (world * scale) + offset
        canvas_x = (img_x * self.scale_factor) + self.offset_x
        canvas_y = (img_y * self.scale_factor) + self.offset_y
        return canvas_x, canvas_y


    def set_drawing_enabled(self, enabled: bool):
        self._is_drawing_enabled = enabled
        if not enabled and self.selected_item_id:
            self.deselect_item()

    def set_draw_mode(self, mode):
        self.draw_mode = mode
        self._cleanup_drawing_state()

    def set_photo(self, image_path: str):
        self.clear_canvas()
        if image_path and hasattr(self, "canvas"):
            try:
                self.image = Image.open(image_path)
                self.image = ImageOps.exif_transpose(self.image)
                self.after(10, self.fit_to_screen)
            except Exception as e:
                print(f"Error loading image: {e}")
                raise

    def fit_to_screen(self):
        if not self.image or not self.canvas.winfo_width() > 1:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.image.size

        if img_width > 0 and img_height > 0:
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            self.scale_factor = min(scale_w, scale_h)

            self.offset_x = (canvas_width - (img_width * self.scale_factor)) / 2
            self.offset_y = (canvas_height - (img_height * self.scale_factor)) / 2

            self._schedule_render(delay=0, high_quality=True, fit_screen=True)

    def add_annotation(self, ann_data):
        item_id = ann_data["id"]
        self.item_map[item_id] = {
            "data": ann_data,
            "canvas_items": []  # [poly, text_bg, text]
        }
        self._draw_single_annotation(item_id)
        return ann_data

    def remove_annotation(self, item_id):
        if item_id in self.item_map:
            self._delete_annotation_graphics(item_id)
            del self.item_map[item_id]
            if self.selected_item_id == item_id:
                self.selected_item_id = None

    def update_annotation_coords(self, item_id, new_coords_image_space):
        if item_id in self.item_map:
            self.item_map[item_id]["data"]["coords"] = new_coords_image_space
            self._update_single_annotation_coords(item_id)
            if self.selected_item_id == item_id:
                self._draw_resize_handles()

    def redraw_annotations_from_model(self, annotations_list):
        """(Этап 10) Полностью очищает и перерисовывает все с нуля."""
        current_selection = self.selected_item_id

        self.canvas.delete("annotation")
        self.canvas.delete("handle")
        self.item_map.clear()
        self.resize_handles.clear()

        for ann_data in annotations_list:
            self.add_annotation(ann_data.copy())

        if current_selection and current_selection in self.item_map:
            self.selected_item_id = current_selection
            self._stylize_selection(current_selection, True)
            self._draw_resize_handles()

    def get_selected_ids(self):
        return [self.selected_item_id] if self.selected_item_id else []

    def clear_canvas(self):
        if self._zoom_job:
            self.after_cancel(self._zoom_job)
            self._zoom_job = None
        if self.render_job:
            self.after_cancel(self.render_job)
            self.render_job = None

        self.canvas.delete("all")
        self.image_id = None
        self.image = None
        self.photo_image = None

        self.scale_factor = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        self.item_map.clear()
        self.selected_item_id = None
        self.resize_handles.clear()
        self._cleanup_drawing_state()

    def _cleanup_drawing_state(self):
        if self.current_poly_preview_id:
            self.canvas.delete(self.current_poly_preview_id)
        self.current_poly_points = []
        self.current_poly_preview_id = None
        self.action_state = {
            "action": None,
            "start_pos": None,
            "item_id": None,
            "handle": None,
            "temp_coords": None,
        }

    def _schedule_render(self, delay=15, high_quality=False, fit_screen=False):
        """(Этап 10) Отменяет предыдущий рендер и планирует новый."""
        if self.render_job:
            self.after_cancel(self.render_job)

        self.render_job = self.after(
            delay,
            lambda: self._render_viewport(high_quality, fit_screen)
        )

    def _render_viewport(self, high_quality=False, fit_screen=False):
        """
        Рендерит только видимую часть изображения.
        Теперь корректно обрабатывает ситуации, когда изображение меньше холста
        или видно лишь частично.
        """
        self.render_job = None
        if not self.image:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 1 or ch < 1: return

        # 1. Определяем видимую область в координатах ИЗОБРАЖЕНИЯ
        # (Какие пиксели картинки видны на холсте от (0,0) до (cw,ch)?)
        vis_x1 = (0 - self.offset_x) / self.scale_factor
        vis_y1 = (0 - self.offset_y) / self.scale_factor
        vis_x2 = (cw - self.offset_x) / self.scale_factor
        vis_y2 = (ch - self.offset_y) / self.scale_factor

        # 2. Находим пересечение видимой области с реальным изображением
        img_w, img_h = self.image.size
        crop_x1 = max(0, min(img_w, int(vis_x1)))
        crop_y1 = max(0, min(img_h, int(vis_y1)))
        crop_x2 = max(0, min(img_w, int(vis_x2) + 1))  # +1 для запаса на округление
        crop_y2 = max(0, min(img_h, int(vis_y2) + 1))

        # 3. Если пересечения нет, прячем изображение и выходим
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            if self.image_id:
                self.canvas.itemconfig(self.image_id, state='hidden')
            return

        # 4. Вырезаем видимую часть
        crop = self.image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # 5. Вычисляем, куда на холсте поставить этот кроп.
        # Используем нашу стандартную формулу _image_to_canvas_coords
        canvas_crop_x1 = (crop_x1 * self.scale_factor) + self.offset_x
        canvas_crop_y1 = (crop_y1 * self.scale_factor) + self.offset_y

        # 6. Вычисляем целевой размер кропа на экране
        # (Можно использовать crop.size * scale, но для точности лучше разницу координат)
        canvas_crop_x2 = (crop_x2 * self.scale_factor) + self.offset_x
        canvas_crop_y2 = (crop_y2 * self.scale_factor) + self.offset_y
        target_w = max(1, int(canvas_crop_x2 - canvas_crop_x1))
        target_h = max(1, int(canvas_crop_y2 - canvas_crop_y1))

        # 7. Ресайз
        resample_method = Image.Resampling.LANCZOS if high_quality else Image.Resampling.NEAREST
        # Используем NEAREST для максимальной скорости при зуме/пане,
        # BILINEAR может все еще лагать на старом железе при огромных картинках.
        # Если NEAREST слишком пиксельный, верните BILINEAR.

        resized_crop = crop.resize((target_w, target_h), resample_method)
        self.photo_image = ImageTk.PhotoImage(resized_crop)

        # 8. Отображение
        if self.image_id:
            self.canvas.itemconfig(self.image_id, image=self.photo_image, state='normal')
            self.canvas.coords(self.image_id, canvas_crop_x1, canvas_crop_y1)
        else:
            self.image_id = self.canvas.create_image(
                canvas_crop_x1, canvas_crop_y1, anchor="nw", image=self.photo_image
            )
        self.canvas.tag_lower(self.image_id)

        # 9. Обновление аннотаций
        if fit_screen:
            self.redraw_annotations_from_model(list(self.item_map.values()))
        else:
            self._update_all_coords()


    def _draw_single_annotation(self, item_id, temp_coords=None):
        """(Этап 10) Создает объекты canvas ОДИН РАЗ."""
        if item_id not in self.item_map:
            return

        ann_data = self.item_map[item_id]["data"]
        ann_type = ann_data["type"]
        coords = temp_coords if temp_coords is not None else ann_data["coords"]
        class_name = ann_data["class_name"]
        color = ann_data.get("color", "red")
        is_selected = item_id == self.selected_item_id

        self._delete_annotation_graphics(item_id)

        canvas_items = []
        poly_id, text_bg_id, text_id = None, None, None
        cx1, cy1 = None, None  # Точка привязки для текста

        if ann_type == "rect" and coords and len(coords) == 4:
            x, y, w, h = coords
            cx1, cy1 = self._image_to_canvas_coords(x, y)
            cx2, cy2 = self._image_to_canvas_coords(x + w, y + h)

            poly_id = self.canvas.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline=color, width=2,
                tags=("annotation", f"ann_{item_id}", item_id),
                dash=(4, 4) if is_selected else (),
            )
            canvas_items.append(poly_id)
            self.canvas.tag_bind(poly_id, "<ButtonPress-1>", lambda e: self._on_annotation_click(e, item_id))

        elif ann_type == "poly" and coords:
            scaled_points = [self._image_to_canvas_coords(p[0], p[1]) for p in coords]

            if len(scaled_points) > 1:
                poly_id = self.canvas.create_polygon(
                    scaled_points, outline=color, fill="", width=2,
                    tags=("annotation", f"ann_{item_id}", item_id),
                    dash=(4, 4) if is_selected else (),
                )
                canvas_items.append(poly_id)
                self.canvas.tag_bind(poly_id, "<ButtonPress-1>", lambda e: self._on_annotation_click(e, item_id))

                if scaled_points:
                    cx1, cy1 = scaled_points[0]

        if cx1 is not None:
            text_id = self.canvas.create_text(
                cx1, cy1 - 2, anchor="sw", text=class_name, fill="white",
                font=("Segoe UI", 10, "bold"),
                tags=("annotation", f"ann_{item_id}", item_id),
            )
            bbox = self.canvas.bbox(text_id)
            if bbox:
                text_bg_id = self.canvas.create_rectangle(
                    bbox, fill=color, outline="",
                    tags=("annotation", f"ann_{item_id}", item_id),
                )
                self.canvas.tag_lower(text_bg_id, text_id)
                canvas_items.extend([text_bg_id, text_id])

                self.canvas.tag_bind(text_id, "<ButtonPress-1>", lambda e: self._on_annotation_click(e, item_id))
                self.canvas.tag_bind(text_bg_id, "<ButtonPress-1>", lambda e: self._on_annotation_click(e, item_id))

        self.item_map[item_id]["canvas_items"] = canvas_items

    def _on_annotation_click(self, event, item_id):
        if not self._is_drawing_enabled:
            return

        if self.selected_item_id == item_id:
            canvas_x, canvas_y = event.x, event.y
            self.action_state["start_pos"] = (canvas_x, canvas_y)
            self.action_state["action"] = "move"
            self.action_state["item_id"] = item_id
            self._fire_event("annotation_press", item_id)
            self._set_annotations_stipple(active_item_id=item_id, stipple="gray50")
        else:
            self.select_item(item_id)

    def _delete_annotation_graphics(self, item_id):
        if item_id in self.item_map:
            for canvas_id in self.item_map[item_id].get("canvas_items", []):
                if canvas_id:
                    self.canvas.delete(canvas_id)
            self.item_map[item_id]["canvas_items"] = []

    def _toggle_other_annotations(self, show=True):
        state = tk.NORMAL if show else tk.HIDDEN
        for item_id_key, item_val in self.item_map.items():
            if item_id_key != self.selected_item_id:
                for canvas_id in item_val.get("canvas_items", []):
                    if canvas_id:
                        self.canvas.itemconfig(canvas_id, state=state)

    def _set_annotations_stipple(self, active_item_id=None, stipple=""):
        for item_id_key, item_val in self.item_map.items():
            is_active = item_id_key == active_item_id
            poly_id = item_val.get("canvas_items", [None])[0]
            if poly_id:
                self.canvas.itemconfig(poly_id, stipple="" if is_active or not stipple else stipple)

            text_items = item_val.get("canvas_items", [None, None, None])[1:]
            for text_id in text_items:
                if text_id and not is_active:
                    self.canvas.itemconfig(text_id, state=tk.HIDDEN if stipple else tk.NORMAL)

    def _draw_resize_handles(self):
        self.canvas.delete("handle")
        self.resize_handles.clear()
        if not self.selected_item_id or not self._is_drawing_enabled:
            return

        ann_data = self.item_map.get(self.selected_item_id)
        if not ann_data:
            return

        if ann_data["data"]["type"] == "rect":
            x, y, w, h = ann_data["data"]["coords"]
            cx1, cy1 = self._image_to_canvas_coords(x, y)
            cx2, cy2 = self._image_to_canvas_coords(x + w, y + h)

            coords = {"nw": (cx1, cy1), "ne": (cx2, cy1), "sw": (cx1, cy2), "se": (cx2, cy2)}
            for name, (px, py) in coords.items():
                handle_id = self.canvas.create_rectangle(
                    px - RESIZE_HANDLE_SIZE / 2, py - RESIZE_HANDLE_SIZE / 2,
                    px + RESIZE_HANDLE_SIZE / 2, py + RESIZE_HANDLE_SIZE / 2,
                    fill="white", outline="black", tags=("handle",),
                )
                self.resize_handles[name] = handle_id

        elif ann_data["data"]["type"] == "poly":
            points = ann_data["data"]["coords"]
            for i, (px, py) in enumerate(points):
                cx, cy = self._image_to_canvas_coords(px, py)
                handle_id = self.canvas.create_oval(
                    cx - RESIZE_HANDLE_SIZE / 2, cy - RESIZE_HANDLE_SIZE / 2,
                    cx + RESIZE_HANDLE_SIZE / 2, cy + RESIZE_HANDLE_SIZE / 2,
                    fill="white", outline="black", tags=("handle",),
                )
                self.resize_handles[i] = handle_id

    def on_mouse_wheel(self, event):
        if not self.image:
            return
        if self._zoom_job:
            self.after_cancel(self._zoom_job)
            self._zoom_job = None

        if event.num == 4 or event.delta > 0:
            zoom_factor = 1.1
        elif event.num == 5 or event.delta < 0:
            zoom_factor = 1 / 1.1
        else:
            return

        self.zoom_center = (event.x, event.y)
        self.target_scale = self.scale_factor * zoom_factor
        self.target_scale = max(0.01, min(self.target_scale, 50.0))  # Расширенные лимиты зума

        self._zoom_job = self.after(15, self._apply_zoom)

    def _apply_zoom(self):
        """(Этап 10) Применяет "зум к курсору", вычисляя новый offset."""
        if not self.image or not self.zoom_center:
            return

        cx, cy = self.zoom_center

        # 1. Где была точка под курсором в "мире" (Image space)
        img_x, img_y = self._canvas_to_image_coords(cx, cy)

        # 2. Обновляем масштаб
        self.scale_factor = self.target_scale

        # 3. Вычисляем новый offset, чтобы (img_x, img_y) осталась под (cx, cy)
        # New_Canvas_X = (World_X * New_Scale) + New_Offset
        # CX = (img_x * self.scale_factor) + New_Offset
        # New_Offset = CX - (img_x * self.scale_factor)
        self.offset_x = cx - (img_x * self.scale_factor)
        self.offset_y = cy - (img_y * self.scale_factor)

        # 4. (Этап 10) Планируем рендер
        self._schedule_render(delay=0)  # Немедленный рендер (LQ)

        self._zoom_job = None
        self.zoom_center = None


    def start_pan(self, event):
        pass

    def pan_image(self, event):
        pass

    def start_pan_v2(self, event):
        self.action_state["action"] = "pan"
        self.action_state["start_pos"] = (event.x, event.y)
        self.action_state["start_offset"] = (self.offset_x, self.offset_y)

    def pan_image_v2(self, event):
        if self.action_state["action"] != "pan":
            return

        start_x, start_y = self.action_state["start_pos"]
        start_off_x, start_off_y = self.action_state["start_offset"]

        dx, dy = event.x - start_x, event.y - start_y

        # Двигаем "камеру"
        self.offset_x = start_off_x + dx
        self.offset_y = start_off_y + dy

        self._schedule_render(delay=5)

    def end_pan_v2(self, event):
        if self.action_state["action"] == "pan":
            self._schedule_render(delay=50, high_quality=True)

        self.action_state["action"] = None


    def on_mouse_press(self, event):
        if not self.image or not self._is_drawing_enabled:
            return

        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.action_state["start_pos"] = (canvas_x, canvas_y)

        # Проверяем, не кликнули ли мы на ручку изменения размера
        handle_under_cursor = None
        if self.selected_item_id:
            for name, handle_id in self.resize_handles.items():
                if self.is_inside(handle_id, canvas_x, canvas_y):
                    handle_under_cursor = name
                    break

        if handle_under_cursor is not None:
            self.action_state.update(
                {
                    "action": "resize",
                    "item_id": self.selected_item_id,
                    "handle": handle_under_cursor,
                }
            )
            self.action_state["start_pos_img"] = self._canvas_to_image_coords(canvas_x, canvas_y)
            self._fire_event("annotation_press", self.selected_item_id)
            self._set_annotations_stipple(active_item_id=self.selected_item_id, stipple="gray50")
            return

        # Проверяем, не кликнули ли мы на аннотацию
        items_under_cursor = self.canvas.find_overlapping(canvas_x - 1, canvas_y - 1, canvas_x + 1, canvas_y + 1)
        for item in items_under_cursor:
            tags = self.canvas.gettags(item)
            if "annotation" in tags:
                self._on_annotation_click(event, tags[2])  # tags[2] это item_id
                return

        # Если кликнули на пустое место, снимаем выделение и начинаем рисование
        self.deselect_item()

        img_x, img_y = self._canvas_to_image_coords(canvas_x, canvas_y)

        if self.draw_mode == "poly":
            self._toggle_other_annotations(show=False)
            self.action_state["action"] = "draw_poly"
            self.current_poly_points.append((img_x, img_y))
            self.canvas.delete("poly_preview")

            if len(self.current_poly_points) > 1:
                scaled_points = [self._image_to_canvas_coords(p[0], p[1]) for p in self.current_poly_points]
                self.current_poly_preview_id = self.canvas.create_line(
                    *sum(scaled_points, ()),
                    fill="red", width=2, dash=(4, 4), tags="poly_preview",
                )
            return

        if self.draw_mode == "rect":
            self._toggle_other_annotations(show=False)
            self.action_state["action"] = "draw_rect"
            self.action_state["start_pos_img"] = (img_x, img_y)
            self.action_state["item_id"] = self.canvas.create_rectangle(
                canvas_x, canvas_y, canvas_x, canvas_y,
                outline="red", dash=(4, 4), width=2,
            )

    def on_right_mouse_press(self, event):
        if self.draw_mode == "poly" and self.current_poly_points:
            if len(self.current_poly_points) > 2:
                self._fire_event("annotation_added", "poly", self.current_poly_points)
            self._cleanup_drawing_state()
            self.canvas.delete("poly_preview")
            self._toggle_other_annotations(show=True)

    def on_mouse_move(self, event):
        if not self.action_state.get("action"):
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        start_x_canvas, start_y_canvas = self.action_state["start_pos"]

        action = self.action_state["action"]
        item_id = self.action_state["item_id"]

        if action == "draw_rect":
            self.canvas.coords(item_id, start_x_canvas, start_y_canvas, canvas_x, canvas_y)

        elif action == "move" and item_id:
            dx_canvas, dy_canvas = canvas_x - start_x_canvas, canvas_y - start_y_canvas

            canvas_items = self.item_map[item_id].get("canvas_items", [])
            for cid in canvas_items:
                self.canvas.move(cid, dx_canvas, dy_canvas)
            for hid in self.resize_handles.values():
                self.canvas.move(hid, dx_canvas, dy_canvas)

            self.action_state["start_pos"] = (canvas_x, canvas_y)

            start_img_x, start_img_y = self._canvas_to_image_coords(start_x_canvas, start_y_canvas)
            current_img_x, current_img_y = self._canvas_to_image_coords(canvas_x, canvas_y)
            img_dx = current_img_x - start_img_x
            img_dy = current_img_y - start_img_y

            # Обновляем temp_coords, если они есть, иначе берем из item_map
            original_coords = self.action_state.get("temp_coords") or self.item_map[item_id]["data"]["coords"]

            if self.item_map[item_id]["data"]["type"] == "rect":
                x, y, w, h = original_coords
                temp_coords = (x + img_dx, y + img_dy, w, h)
            elif self.item_map[item_id]["data"]["type"] == "poly":
                temp_coords = [(p[0] + img_dx, p[1] + img_dy) for p in original_coords]

            self.action_state["temp_coords"] = temp_coords

        elif action == "resize" and item_id:
            img_mouse_x, img_mouse_y = self._canvas_to_image_coords(canvas_x, canvas_y)
            handle = self.action_state["handle"]
            original_coords = self.item_map[item_id]["data"]["coords"]

            if self.item_map[item_id]["data"]["type"] == "rect":
                x, y, w, h = original_coords
                img_x1, img_y1, img_x2, img_y2 = x, y, x + w, y + h
                if "n" in str(handle): img_y1 = img_mouse_y
                if "s" in str(handle): img_y2 = img_mouse_y
                if "w" in str(handle): img_x1 = img_mouse_x
                if "e" in str(handle): img_x2 = img_mouse_x

                new_img_x, new_img_y = min(img_x1, img_x2), min(img_y1, img_y2)
                new_img_w, new_img_h = abs(img_x1 - img_x2), abs(img_y1 - img_y2)

                if new_img_w > MIN_BBOX_SIZE and new_img_h > MIN_BBOX_SIZE:
                    temp_coords = (new_img_x, new_img_y, new_img_w, new_img_h)
                    self.action_state["temp_coords"] = temp_coords
                    self._update_single_annotation_coords(item_id, temp_coords)
                    self._draw_resize_handles()

            elif self.item_map[item_id]["data"]["type"] == "poly":
                temp_coords = list(original_coords)
                temp_coords[handle] = (img_mouse_x, img_mouse_y)
                self.action_state["temp_coords"] = temp_coords
                self._update_single_annotation_coords(item_id, temp_coords)
                self._draw_resize_handles()

    def on_mouse_release(self, event):
        action = self.action_state.get("action")
        if not action or action == "draw_poly":
            return

        self.canvas.delete("poly_preview")
        self._toggle_other_annotations(show=True)
        self._set_annotations_stipple(active_item_id=self.selected_item_id)

        if action == "draw_rect":
            rect_id = self.action_state.get("item_id")
            if rect_id:
                self.canvas.delete(rect_id)  # Удаляем превью

                start_img_x, start_img_y = self.action_state["start_pos_img"]
                end_canvas_x = self.canvas.canvasx(event.x)
                end_canvas_y = self.canvas.canvasy(event.y)
                end_img_x, end_img_y = self._canvas_to_image_coords(end_canvas_x, end_canvas_y)

                img_x1, img_y1 = min(start_img_x, end_img_x), min(start_img_y, end_img_y)
                img_w, img_h = abs(start_img_x - end_img_x), abs(start_img_y - end_img_y)

                if img_w > MIN_BBOX_SIZE and img_h > MIN_BBOX_SIZE:
                    self._fire_event("annotation_added", "rect", (img_x1, img_y1, img_w, img_h))

        elif action in ["move", "resize"]:
            item_id = self.action_state.get("item_id")
            temp_coords = self.action_state.get("temp_coords")
            if item_id in self.item_map and temp_coords is not None:
                self.update_annotation_coords(item_id, temp_coords)
                self._fire_event("annotation_modified", item_id, temp_coords)
            elif item_id in self.item_map:
                self._update_single_annotation_coords(item_id)

        self.action_state = {
            "action": None,
            "start_pos": None,
            "start_pos_img": None,
            "item_id": None,
            "handle": None,
            "temp_coords": None,
        }

    def update_cursor(self, event):
        if not self._is_drawing_enabled:
            self.canvas.config(cursor="")
            return

        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Обновляем превью полигона при рисовании
        if self.action_state.get("action") == "draw_poly" and self.current_poly_points:
            self.canvas.delete("poly_preview")
            scaled_points = [self._image_to_canvas_coords(p[0], p[1]) for p in self.current_poly_points]
            preview_points = scaled_points + [(canvas_x, canvas_y)]
            if len(preview_points) > 1:
                self.current_poly_preview_id = self.canvas.create_line(
                    *sum(preview_points, ()),
                    fill="red", width=2, dash=(4, 4), tags="poly_preview",
                )

        # Проверяем курсор над ручками изменения размера
        for name, handle_id in self.resize_handles.items():
            if self.is_inside(handle_id, canvas_x, canvas_y):
                if isinstance(name, str):
                    if name in ["nw", "se"]:
                        self.canvas.config(cursor="top_left_corner")
                    elif name in ["ne", "sw"]:
                        self.canvas.config(cursor="bottom_left_corner")
                    elif name in ["n", "s"]:
                        self.canvas.config(cursor="sb_v_double_arrow")
                    elif name in ["w", "e"]:
                        self.canvas.config(cursor="sb_h_double_arrow")
                else:
                    self.canvas.config(cursor="crosshair")
                return

        # Проверяем курсор над аннотациями
        items_under_cursor = self.canvas.find_overlapping(canvas_x - 1, canvas_y - 1, canvas_x + 1, canvas_y + 1)
        for item in items_under_cursor:
            tags = self.canvas.gettags(item)
            if "annotation" in tags:
                self.canvas.config(cursor="fleur")
                return

        # Курсор по умолчанию
        self.canvas.config(cursor="crosshair")

    def select_item(self, item_id):
        if self.selected_item_id == item_id:
            return
        self.deselect_item()
        self.selected_item_id = item_id
        self._stylize_selection(item_id, True)
        self._draw_resize_handles()
        self._fire_event("annotation_selected", item_id)

    def deselect_item(self):
        if not self.selected_item_id:
            return
        old_id = self.selected_item_id
        self.selected_item_id = None
        self.canvas.delete("handle")
        self.resize_handles.clear()
        if old_id in self.item_map:
            self._stylize_selection(old_id, False)
        self._fire_event("annotation_deselected", old_id)

    def _stylize_selection(self, item_id: str, is_selected: bool):
        if item_id not in self.item_map:
            return

        poly_id = self.item_map[item_id].get("canvas_items", [None])[0]
        if poly_id:
            self.canvas.itemconfig(poly_id, dash=(4, 4) if is_selected else ())

    def _update_single_annotation_coords(self, item_id, temp_coords=None):
        """(Этап 10) Быстро обновляет геометрию ОДНОГО объекта."""
        if item_id not in self.item_map:
            return

        canvas_items = self.item_map[item_id].get("canvas_items", [])
        if not canvas_items:
            self._draw_single_annotation(item_id, temp_coords)
            return

        ann_data = self.item_map[item_id]["data"]
        ann_type = ann_data["type"]
        coords = temp_coords if temp_coords is not None else ann_data["coords"]

        poly_id = canvas_items[0]
        text_bg_id = canvas_items[1] if len(canvas_items) > 1 else None
        text_id = canvas_items[2] if len(canvas_items) > 2 else None

        cx1, cy1 = None, None  # Точка привязки для текста

        if ann_type == "rect" and coords and len(coords) == 4:
            x, y, w, h = coords
            cx1, cy1 = self._image_to_canvas_coords(x, y)
            cx2, cy2 = self._image_to_canvas_coords(x + w, y + h)
            self.canvas.coords(poly_id, cx1, cy1, cx2, cy2)

        elif ann_type == "poly" and coords:
            scaled_points = [self._image_to_canvas_coords(p[0], p[1]) for p in coords]
            if len(scaled_points) > 1:
                self.canvas.coords(poly_id, *sum(scaled_points, ()))
                if scaled_points:
                    cx1, cy1 = scaled_points[0]

        if text_id and cx1 is not None:
            self.canvas.coords(text_id, cx1, cy1 - 2)
            if text_bg_id:
                bbox = self.canvas.bbox(text_id)
                if bbox:
                    self.canvas.coords(text_bg_id, *bbox)

    def _update_all_coords(self):
        """(Этап 10) Супер-быстрый апдейт, использует self.canvas.coords() для всех."""
        for item_id in self.item_map:
            self._update_single_annotation_coords(item_id)
        self._draw_resize_handles()

    def is_inside(self, item_id, x, y):
        coords = self.canvas.coords(item_id)
        if not coords:
            return False
        return coords[0] <= x <= coords[2] and coords[1] <= y <= coords[3]

    def _get_closest_segment_to_point(self, point, polygon_coords):
        min_dist = float("inf")
        insert_index = -1
        closest_point_on_segment = None

        px, py = point

        for i in range(len(polygon_coords)):
            p1 = polygon_coords[i]
            p2 = polygon_coords[(i + 1) % len(polygon_coords)]
            x1, y1 = p1
            x2, y2 = p2

            dx, dy = x2 - x1, y2 - y1

            if dx == 0 and dy == 0:
                dist = math.hypot(px - x1, py - y1)
                t = 0
            else:
                t = ((px - x1) * dx + (py - y1) * dy) / (dx ** 2 + dy ** 2)
                t = max(0, min(1, t))

                closest_x = x1 + t * dx
                closest_y = y1 + t * dy
                dist = math.hypot(px - closest_x, py - closest_y)

            if dist < min_dist:
                min_dist = dist
                insert_index = i + 1
                closest_point_on_segment = (closest_x, closest_y)

        return insert_index, closest_point_on_segment

    def on_add_vertex(self, event):
        if not self.selected_item_id or not self._is_drawing_enabled:
            return

        ann_data = self.item_map.get(self.selected_item_id)
        if not ann_data or ann_data["data"]["type"] != "poly":
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        img_x, img_y = self._canvas_to_image_coords(canvas_x, canvas_y)

        old_coords = ann_data["data"]["coords"]
        new_coords = list(old_coords)

        insert_index, _ = self._get_closest_segment_to_point((img_x, img_y), new_coords)

        if insert_index != -1:
            new_coords.insert(insert_index, (img_x, img_y))

            self._fire_event("annotation_press", self.selected_item_id)
            self.action_state["temp_coords"] = new_coords
            self._fire_event("annotation_modified", self.selected_item_id, new_coords)

            self.update_annotation_coords(self.selected_item_id, new_coords)
            self.select_item(self.selected_item_id)