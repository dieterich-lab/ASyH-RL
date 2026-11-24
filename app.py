#!/usr/bin/env python
# the app with ascii text based gui 
# it runs the pipeline implemented in the main script `main.py`
# a user can choose a model they prefer and specify values of following 
# parameters: NUM_EPISODES, SHARE_TRAIN, THRESH_CONT_COL
# also in the ascii gui a user has an option to alter paths `train_data_path`, `val_data_path`
# and the path for a produced file `output_csv`
# last but not least - a user can tweak the parameter space (stored in the var. modelParamSpace)

import ast
import copy
import curses
import traceback

import main as pipeline

ESC = 27
ENTER_KEYS = (curses.KEY_ENTER, 10, 13)
BACKSPACE_KEYS = (curses.KEY_BACKSPACE, 127, 8)

MODEL_NAMES = [m.__name__ for m in pipeline.MODELS_AVAIL]

# (label, key, kind); kind: 'model' | 'int' | 'float' | 'str' | 'space' | 'run' | 'quit'
FIELDS = [
    ("Model", "model_num", "model"),
    ("NUM_EPISODES", "num_episodes", "int"),
    ("SHARE_TRAIN", "share_train", "float"),
    ("THRESH_CONT_COL", "thresh_cont_col", "int"),
    ("train_data_path", "train_data_path", "str"),
    ("val_data_path", "val_data_path", "str"),
    ("output_csv", "output_csv", "str"),
    ("Parameter space", "param_space", "space"),
    ("[ Run pipeline ]", None, "run"),
    ("[ Quit ]", None, "quit"),
]


def default_config():
    return {
        "model_num": 0,
        "num_episodes": pipeline.NUM_EPISODES,
        "share_train": pipeline.SHARE_TRAIN,
        "thresh_cont_col": pipeline.THRESH_CONT_COL,
        "train_data_path": pipeline.TRAIN_DATA_PATH,
        "val_data_path": pipeline.VAL_DATA_PATH,
        "output_csv": "",  # empty -> auto-generated name from main.py
        "param_space": copy.deepcopy(pipeline.DEFAULT_PARAM_SPACE),
    }


def parse_value(kind, text):
    """Convert text to the field type; raises ValueError on invalid input."""
    if kind == "int":
        value = int(text)
        if value <= 0:
            raise ValueError("must be a positive integer")
        return value
    if kind == "float":
        value = float(text)
        if not 0.0 < value < 1.0:
            raise ValueError("must be in the open range (0, 1)")
        return value
    return text.strip()


def parse_values_list(text):
    value = ast.literal_eval(text.strip())
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError("expected a non-empty list, e.g. [25, 50, 100]")
    return list(value)


# role -> (fg, bg) for 256-color terminals and for 8-color fallback; -1 = terminal default
PALETTE_256 = {
    "border": (67, -1),
    "header": (255, 24),
    "label": (110, -1),
    "value": (180, -1),
    "sel": (255, 60),
    "run": (108, -1),
    "quit": (174, -1),
    "error": (255, 131),
    "footer": (246, -1),
}
PALETTE_8 = {
    "border": (curses.COLOR_BLUE, -1),
    "header": (curses.COLOR_WHITE, curses.COLOR_BLUE),
    "label": (curses.COLOR_CYAN, -1),
    "value": (curses.COLOR_YELLOW, -1),
    "sel": (curses.COLOR_BLACK, curses.COLOR_CYAN),
    "run": (curses.COLOR_GREEN, -1),
    "quit": (curses.COLOR_RED, -1),
    "error": (curses.COLOR_WHITE, curses.COLOR_RED),
    "footer": (curses.COLOR_WHITE, -1),
}


class Screen:
    def __init__(self, stdscr):
        self.scr = stdscr
        curses.curs_set(0)
        stdscr.keypad(True)
        self.status = ""
        self.attrs = self._init_colors()

    @staticmethod
    def _init_colors():
        fallback = {"header": curses.A_BOLD, "sel": curses.A_REVERSE, "error": curses.A_REVERSE}
        if not curses.has_colors():
            return {role: fallback.get(role, 0) for role in PALETTE_8}
        curses.start_color()
        try:
            curses.use_default_colors()
        except curses.error:
            pass
        palette = PALETTE_256 if curses.COLORS >= 256 else PALETTE_8
        attrs = {}
        for i, (role, (fg, bg)) in enumerate(palette.items(), start=1):
            curses.init_pair(i, fg, bg)
            attrs[role] = curses.color_pair(i)
        for role in ("header", "sel"):
            attrs[role] |= curses.A_BOLD
        return attrs

    def attr(self, role):
        return self.attrs.get(role, 0)

    def draw_frame(self, title, footer):
        self.scr.erase()
        h, w = self.scr.getmaxyx()
        border = self.attr("border")
        rule = "+" + "-" * (w - 3) + "+"
        self.scr.addnstr(0, 0, rule, w - 1, border)
        self.scr.addnstr(1, 0, "|", 1, border)
        self.scr.addnstr(1, 1, (" " + title).ljust(w - 3), w - 3, self.attr("header"))
        self.scr.addnstr(1, w - 2, "|", 1, border)
        self.scr.addnstr(2, 0, rule, w - 1, border)
        self.scr.addnstr(h - 3, 0, rule, w - 1, border)
        self.scr.addnstr(h - 2, 0, "|", 1, border)
        self.scr.addnstr(h - 2, 1, (" " + footer).ljust(w - 3), w - 3, self.attr("footer"))
        self.scr.addnstr(h - 2, w - 2, "|", 1, border)
        self.scr.addnstr(h - 1, 0, self.status.ljust(w - 1), w - 1, self.attr("error") if self.status else 0)

    def draw_segments(self, y, segments, selected):
        """Draw (text, role) pieces on row y; the selected row uses a single highlight style."""
        _, w = self.scr.getmaxyx()
        sel = self.attr("sel")
        if selected:
            self.scr.addnstr(y, 0, " " * (w - 1), w - 1, sel)
        x = 0
        for text, role in segments:
            if x >= w - 1:
                break
            self.scr.addnstr(y, x, text, w - 1 - x, sel if selected else self.attr(role))
            x += len(text)

    def prompt(self, label, initial=""):
        """Single-line editor on the bottom area. Returns text, or None if cancelled with Esc."""
        buf = list(initial)
        pos = len(buf)
        curses.curs_set(1)
        try:
            while True:
                h, w = self.scr.getmaxyx()
                y = h - 5
                self.scr.move(y, 0)
                self.scr.clrtoeol()
                self.scr.addnstr(y - 1, 0, (label + " (Enter=ok, Esc=cancel)").ljust(w - 1), w - 1, self.attr("header"))
                prefix = "> "
                avail = max(1, w - len(prefix) - 2)
                start = max(0, pos - avail + 1)
                shown = "".join(buf)[start:start + avail]
                self.scr.addnstr(y, 0, " " * (w - 1), w - 1)
                self.scr.addnstr(y, 0, prefix, w - 1, self.attr("border") | curses.A_BOLD)
                self.scr.addnstr(y, len(prefix), shown, max(0, w - 1 - len(prefix)), self.attr("value"))
                self.scr.move(y, len(prefix) + pos - start)
                self.scr.refresh()
                ch = self.scr.getch()
                if ch == ESC:
                    return None
                if ch in ENTER_KEYS:
                    return "".join(buf)
                if ch in BACKSPACE_KEYS:
                    if pos > 0:
                        del buf[pos - 1]
                        pos -= 1
                elif ch == curses.KEY_DC:
                    if pos < len(buf):
                        del buf[pos]
                elif ch == curses.KEY_LEFT:
                    pos = max(0, pos - 1)
                elif ch == curses.KEY_RIGHT:
                    pos = min(len(buf), pos + 1)
                elif ch == curses.KEY_HOME:
                    pos = 0
                elif ch == curses.KEY_END:
                    pos = len(buf)
                elif 32 <= ch < 256:
                    buf.insert(pos, chr(ch))
                    pos += 1
        finally:
            curses.curs_set(0)

    def list_menu(self, title, footer, rows, draw_row, handler):
        """Generic selectable list. `handler(idx, key)` returns True to exit the menu."""
        sel = 0
        while True:
            self.draw_frame(title, footer)
            h, w = self.scr.getmaxyx()
            visible = max(1, h - 7)
            top = max(0, sel - visible + 1)
            for i, row in enumerate(rows()[top:top + visible]):
                idx = top + i
                self.draw_segments(3 + i, draw_row(idx, row), idx == sel)
            self.scr.refresh()
            key = self.scr.getch()
            n = len(rows())
            if key == curses.KEY_UP:
                sel = (sel - 1) % n
            elif key == curses.KEY_DOWN:
                sel = (sel + 1) % n
            elif key == curses.KEY_RESIZE:
                continue
            else:
                self.status = ""
                if handler(sel, key):
                    return
                n = len(rows())
                sel = min(sel, max(0, n - 1))


def edit_param_space(ui, space):
    keys = lambda: list(space.keys())

    def draw_row(idx, name):
        return [(f" {name:<24} ", "label"), (repr(space[name]), "value")]

    def handler(idx, key):
        names = keys()
        if key in (ESC, ord("q")):
            return True
        if key in ENTER_KEYS and names:
            name = names[idx]
            text = ui.prompt(f"Values for '{name}' (python list)", repr(space[name]))
            if text is not None:
                try:
                    space[name] = parse_values_list(text)
                except (ValueError, SyntaxError) as e:
                    ui.status = f"Invalid: {e}"
        elif key == ord("a"):
            name = ui.prompt("New parameter name")
            if name and name.strip():
                text = ui.prompt(f"Values for '{name.strip()}' (python list)", "[]")
                if text is not None:
                    try:
                        space[name.strip()] = parse_values_list(text)
                    except (ValueError, SyntaxError) as e:
                        ui.status = f"Invalid: {e}"
        elif key in (ord("d"), curses.KEY_DC) and names:
            del space[names[idx]]
        elif key == ord("r"):
            space.clear()
            space.update(copy.deepcopy(pipeline.DEFAULT_PARAM_SPACE))
            ui.status = "Parameter space reset to defaults"
        return False

    ui.list_menu(
        "Parameter space (modelParamSpace)",
        "Up/Down move | Enter edit | a add | d delete | r reset | Esc/q back",
        keys, draw_row, handler,
    )


def run_menu(stdscr, cfg):
    """Returns True to run the pipeline with `cfg`, False when the user quits."""
    ui = Screen(stdscr)
    result = {"run": False}

    def draw_row(idx, field):
        label, key, kind = field
        if kind == "model":
            val = f"< {MODEL_NAMES[cfg['model_num']]} >"
        elif kind == "space":
            val = f"{len(cfg['param_space'])} parameters (Enter to edit)"
        elif kind in ("run", "quit"):
            return [(f" {label}", kind)]
        elif key == "output_csv" and not cfg[key]:
            val = "(auto)"
        else:
            val = str(cfg[key])
        return [(f" {label:<18} ", "label"), (val, "value")]

    def handler(idx, key):
        label, name, kind = FIELDS[idx]
        if key in (ESC, ord("q")):
            return True
        if kind == "model" and key in (curses.KEY_LEFT, curses.KEY_RIGHT):
            step = 1 if key == curses.KEY_RIGHT else -1
            cfg["model_num"] = (cfg["model_num"] + step) % len(MODEL_NAMES)
            return False
        if key not in ENTER_KEYS:
            return False
        if kind == "model":
            cfg["model_num"] = (cfg["model_num"] + 1) % len(MODEL_NAMES)
        elif kind in ("int", "float", "str"):
            text = ui.prompt(label, "" if cfg[name] is None else str(cfg[name]))
            if text is None:
                return False
            try:
                cfg[name] = parse_value(kind, text) if text.strip() or name != "output_csv" else ""
            except ValueError as e:
                ui.status = f"Invalid value for {label}: {e}"
        elif kind == "space":
            edit_param_space(ui, cfg["param_space"])
        elif kind == "run":
            if not cfg["param_space"]:
                ui.status = "Parameter space is empty"
                return False
            result["run"] = True
            return True
        elif kind == "quit":
            return True
        return False

    ui.list_menu(
        "Generative model optimization (RL-driven)",
        "Up/Down move | Enter edit/select | Left/Right change model | Esc/q quit",
        lambda: FIELDS, draw_row, handler,
    )
    return result["run"]


def run_pipeline(cfg):
    # main.py reads these module-level constants at call time
    pipeline.NUM_EPISODES = cfg["num_episodes"]
    pipeline.SHARE_TRAIN = cfg["share_train"]
    pipeline.THRESH_CONT_COL = cfg["thresh_cont_col"]
    model = pipeline.MODELS_AVAIL[cfg["model_num"]]
    pipeline.invoke_model(
        model,
        train_data_path=cfg["train_data_path"],
        val_data_path=cfg["val_data_path"],
        output_csv=cfg["output_csv"] or None,
        param_space=cfg["param_space"],
    )


def main():
    cfg = default_config()
    while True:
        if not curses.wrapper(run_menu, cfg):
            return
        try:
            run_pipeline(cfg)
        except SystemExit:
            print("The pipeline exited early.")
        except Exception:
            traceback.print_exc()
        input("\nPress Enter to return to the menu...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
