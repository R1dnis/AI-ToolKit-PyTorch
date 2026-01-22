import json
import os

from constants import CLASS_SETS_FILE



class ClassManager:
    """
    Централизованно управляет всеми наборами классов (class_sets.json).
    Отвечает за CRUD операции над наборами и самими классами.
    """

    def __init__(self, filepath=CLASS_SETS_FILE):
        self.filepath = filepath
        self.class_sets: dict[str, dict] = {}
        self.is_dirty = False

    def load_from_json(self):
        """Загружает базу классов из JSON-файла."""
        if not os.path.exists(self.filepath):
            print(f"INFO: Файл {self.filepath} не найден, будет создан новый.")
            self.class_sets = {}
            self.save_to_json()  # Создаем дефолтный пустой файл
            return

        try:
            with open(self.filepath, encoding="utf-8") as f:
                self.class_sets = json.load(f)
            self.is_dirty = False
        except json.JSONDecodeError as e:
            print(f"CRITICAL: Файл {self.filepath} поврежден.")
            raise ValueError(
                f"Файл набора классов '{os.path.basename(self.filepath)}' поврежден и не может быть прочитан."
            ) from e
        except OSError as e:
            print(f"CRITICAL: Ошибка чтения {self.filepath}: {e}")
            raise OSError(
                f"Не удалось прочитать файл '{os.path.basename(self.filepath)}'. Проверьте права доступа."
            ) from e

    def save_to_json(self):
        """Сохраняет текущее состояние базы классов в JSON-файл."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.class_sets, f, indent=4, ensure_ascii=False)
            self.is_dirty = False
            print(f"INFO: ClassManager сохранил изменения в {self.filepath}")
        except OSError as e:
            print(f"CRITICAL: Ошибка сохранения {self.filepath}: {e}")
            raise OSError(
                f"Не удалось сохранить файл '{os.path.basename(self.filepath)}'. Проверьте права доступа или место на диске."
            ) from e

    def get_class_set_names(self) -> list[str]:
        """Возвращает отсортированный список имен наборов классов."""
        return sorted(self.class_sets.keys())

    def get_class_set(self, set_name: str) -> dict | None:
        """Возвращает структуру (словарь) указанного набора классов."""
        return self.class_sets.get(set_name)

    def add_class_set(self, set_name: str) -> bool:
        """Добавляет новый пустой набор классов. Возвращает False, если имя занято."""
        if set_name in self.class_sets:
            return False
        self.class_sets[set_name] = {}
        self.is_dirty = True
        return True

    def delete_class_set(self, set_name: str) -> bool:
        """Удаляет набор классов по имени."""
        if set_name in self.class_sets:
            del self.class_sets[set_name]
            self.is_dirty = True
            return True
        return False

    def rename_class_set(self, old_name: str, new_name: str) -> bool:
        """Переименовывает набор. Возвращает False, если новое имя занято."""
        if new_name in self.class_sets or old_name not in self.class_sets:
            return False
        self.class_sets[new_name] = self.class_sets.pop(old_name)
        self.is_dirty = True
        return True

    def _find_class_recursive(self, class_dict, class_name):
        """Вспомогательный рекурсивный поиск класса по имени."""
        if class_name in class_dict:
            return class_dict, class_dict
        for details in class_dict.values():
            if "children" in details and details["children"]:
                found, parent = self._find_class_recursive(
                    details["children"], class_name
                )
                if found:
                    return found, parent
        return None, None

    def add_class(
        self, set_name: str, class_name: str, parent_name: str | None = None
    ) -> bool:
        """Добавляет новый класс в набор. Если указан parent_name, добавляет как подкласс."""
        if set_name not in self.class_sets:
            return False

        # Проверка на дубликат в любом месте набора
        if self.get_class_details(set_name, class_name) is not None:
            return False  # Имя уже занято

        new_class_data = {"color": None, "children": {}}

        if parent_name:
            parent_details, _ = self._find_class_recursive(
                self.class_sets[set_name], parent_name
            )
            if parent_details:
                parent_details["children"][class_name] = new_class_data
            else:
                return False  # Родитель не найден
        else:
            self.class_sets[set_name][class_name] = new_class_data

        # --- ИЗМЕНЕНИЕ: (Этап 3) Убран вызов save_to_json() ---
        self.is_dirty = True
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
        return True

    def delete_class(self, set_name: str, class_name: str) -> bool:
        """Удаляет класс (и всех его потомков) из набора."""
        if set_name not in self.class_sets:
            return False
        details, parent_dict = self._find_class_recursive(
            self.class_sets[set_name], class_name
        )
        if details is not None and parent_dict is not None:
            del parent_dict[class_name]
            self.is_dirty = True
            return True
        return False

    def rename_class(self, set_name: str, old_name: str, new_name: str) -> bool:
        """Переименовывает класс."""
        if set_name not in self.class_sets or old_name == new_name:
            return False

        # Проверка, что новое имя не занято
        if self.get_class_details(set_name, new_name) is not None:
            return False

        details, parent_dict = self._find_class_recursive(
            self.class_sets[set_name], old_name
        )
        if details is not None and parent_dict is not None:
            # Сохраняем данные (включая потомков) и удаляем старую запись
            class_data = parent_dict.pop(old_name)
            # Добавляем с новым именем
            parent_dict[new_name] = class_data
            self.is_dirty = True
            return True
        return False

    def get_class_details(self, set_name: str, class_name: str) -> dict | None:
        """Получает детали класса (включая цвет и потомков) по имени."""
        if set_name not in self.class_sets:
            return None
        details, _ = self._find_class_recursive(self.class_sets[set_name], class_name)
        return details

    def get_all_class_names(self, set_name: str) -> list[str]:
        """Возвращает плоский список всех имен классов в наборе (включая вложенные)."""
        if set_name not in self.class_sets:
            return []

        names = []

        def collect_names(class_dict):
            for name, details in class_dict.items():
                names.append(name)
                if "children" in details and details["children"]:
                    collect_names(details["children"])

        collect_names(self.class_sets[set_name])
        return names

    def get_class_colors(self, set_name: str = None) -> dict[str, str]:
        """
        Возвращает словарь {class_name: hex_color} для *всех* наборов,
        если set_name не указан, или для конкретного набора.
        Пытается назначить уникальные цвета, если они не заданы (пока не реализовано).
        """
        # TODO: Внедрить генерацию цветов, если они None
        colors = {}
        sets_to_scan = (
            self.class_sets.items()
            if set_name is None
            else [(set_name, self.class_sets.get(set_name, {}))]
        )

        def collect_colors(class_dict):
            for name, details in class_dict.items():
                # TODO: Назначать сгенерированный цвет, если details['color'] is None
                colors[name] = details.get("color") or "#FF0000"  # Красный по умолчанию
                if "children" in details and details["children"]:
                    collect_colors(details["children"])

        for _, class_data in sets_to_scan:
            if class_data:
                collect_colors(class_data)

        return colors
