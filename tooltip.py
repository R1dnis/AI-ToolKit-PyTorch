import tkinter as tk

import ttkbootstrap as ttk



class ToolTip(ttk.Frame):
    """
    Кастомный Tooltip (всплывающая подсказка).
    Отображается при наведении на 'button'.
    """

    def __init__(self, button, text="", delay=500, follow=True, **kwargs):
        super().__init__(button.master, **kwargs)
        self.button = button
        self._text = text
        self.delay = delay
        self.follow = follow
        self.tooltip_window = None
        self.id = None
        self.x = self.y = 0

        # Привязки событий
        self.button.bind("<Enter>", self.schedule_tooltip)
        self.button.bind("<Leave>", self.hide_tooltip)
        self.button.bind("<ButtonPress>", self.hide_tooltip)
        if self.follow:
            self.button.bind("<Motion>", self.move_tooltip)

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, new_text):
        self._text = new_text
        if self.tooltip_window:
            self.hide_tooltip()
            self.show_tooltip()

    def schedule_tooltip(self, event=None):
        self.id = self.button.after(self.delay, self.show_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return

        # Геометрия
        x, y, x_offset, y_offset = 0, 0, 10, 10
        self.tooltip_window = tw = tk.Toplevel(self.button)
        tw.wm_overrideredirect(True)

        try:
            # Используем стиль Toplevel из ttkbootstrap
            style = ttk.Style.get_instance()
            theme_name = style.theme.name

            # Адаптация под темную/светлую тему
            if "dark" in theme_name or "superhero" in theme_name:
                bg = "#2b2b2b"
                fg = "#ffffff"
            else:
                bg = "#ffffff"
                fg = "#000000"

            tw.configure(background=bg, borderwidth=1, relief="solid")

        except Exception:
            # Фоллбэк, если что-то не так с темой
            bg = "#2b2b2b"
            fg = "#ffffff"
            tw.configure(background="#2b2b2b", borderwidth=1, relief="solid")

        label = ttk.Label(
            tw,
            text=self.text,
            background=bg,
            foreground=fg,
            wraplength=500,
            justify="left",
            padding=(10, 10),
            font=("Segoe UI", 9)
        )
        label.pack()

        # Расчет позиции
        button_x = self.button.winfo_rootx()
        button_y = self.button.winfo_rooty()
        button_width = self.button.winfo_width()



        # По умолчанию справа
        x = button_x + button_width + x_offset
        y = button_y + y_offset

        # Коррекция, если уходит за экран
        screen_width = self.button.winfo_screenwidth()
        screen_height = self.button.winfo_screenheight()

        tw.update_idletasks()  # Обновляем, чтобы получить winfo_width/height
        tooltip_width = tw.winfo_width()
        tooltip_height = tw.winfo_height()

        if x + tooltip_width > screen_width:
            x = button_x - tooltip_width - x_offset

        if y + tooltip_height > screen_height:
            y = button_y - tooltip_height - y_offset

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        tw.wm_geometry(f"+{int(x)}+{int(y)}")

    def hide_tooltip(self, event=None):
        if self.id:
            self.button.after_cancel(self.id)
            self.id = None
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def move_tooltip(self, event):
        if self.tooltip_window:
            pass
