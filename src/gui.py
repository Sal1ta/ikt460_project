# Tk GUI used for local play

import tkinter as tk
import time
from src.board import HexBoard

# Dark board palette with high contrast pins and move highlights
_BG             = "#0a0a18"
_PANEL_BG       = "#11112a"
_CELL_BOARD     = "#1a1a2e"
_CELL_OUTLINE   = "#28284a"
_CELL_VALID     = "#00c896"
_CELL_VALID_OUT = "#00ffbe"
_SELECTED_GLOW  = "#ffffff"
_TEXT_DIM       = "#5a6a7a"
_TEXT_MAIN      = "#dfe6e9"
_TEXT_ACCENT    = "#a29bfe"

# Active home zones stay visible enough to show lane ownership during play
_HOME_FILL = {
    'red':        "#3b1212", 'lawn green': "#0f3b12",
    'blue':       "#0f2a3b", 'yellow':     "#3b2d0a",
    'purple':     "#26103a", 'gray0':      "#1c1c1c",
}
_HOME_OUT = {
    'red':        "#e74c3c", 'lawn green': "#2ecc71",
    'blue':       "#3498db", 'yellow':     "#f1c40f",
    'purple':     "#9b59b6", 'gray0':      "#95a5a6",
}
# Unused home zones remain visible without competing with active lanes
_HOME_FILL_GHOST = {k: "#121220" for k in _HOME_FILL}
_HOME_OUT_GHOST  = {k: "#1e1e30" for k in _HOME_OUT}

_PIN = {
    'red':        ("#e74c3c", "#ff9f9f", "#922b21"),
    'lawn green': ("#2ecc71", "#a3f0b8", "#1e8449"),
    'blue':       ("#3498db", "#a3d4f5", "#1a5276"),
    'yellow':     ("#f1c40f", "#fdf6b2", "#9a7d0a"),
    'purple':     ("#9b59b6", "#d7b8f3", "#6c3483"),
    'gray0':      ("#95a5a6", "#dfe6e9", "#616a6b"),
}

_LABEL = {
    'red': 'RED', 'lawn green': 'GREEN', 'blue': 'BLUE',
    'yellow': 'YELLOW', 'purple': 'PURPLE', 'gray0': 'GRAY',
}

PINS_PER_PLAYER = 10

class SetupMenuGUI:

    def __init__(self, supported_player_counts, game_mode_options):
        self.result = None
        self._supported_player_counts = sorted(int(count) for count in supported_player_counts)
        self._game_mode_options = list(game_mode_options)
        self._players_value = self._supported_player_counts[0]
        self._mode_value = str(self._game_mode_options[0][1])
        self._click_zones = []
        self._hover_key = None
        self._preview_board = HexBoard(hole_radius=4, spacing=12)

        self.window = tk.Tk()
        self.window.title("Chinese Checkers — Setup")
        self.window.configure(bg=_BG)
        self.window.resizable(False, False)
        self._width = 980
        self._height = 720
        self._center_window(self._width, self._height)

        self.canvas = tk.Canvas(
            self.window, width=self._width, height=self._height,
            bg=_BG, highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)

        self.window.protocol("WM_DELETE_WINDOW", self._exit)
        self.window.bind("<Return>", lambda _event: self._start())
        self.window.bind("<Escape>", lambda _event: self._exit())
        self._draw()

    def _center_window(self, width, height):
        screen_w = self.window.winfo_screenwidth()
        screen_h = self.window.winfo_screenheight()
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def _round_rect(self, x1, y1, x2, y2, radius=8, **kwargs):
        points = [
            x1 + radius, y1, x2 - radius, y1,
            x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2,
            x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius,
            x1, y1 + radius, x1, y1,
        ]
        return self.canvas.create_polygon(points, smooth=True, splinesteps=12, **kwargs)

    def _zone(self, key, kind, value, bounds):
        self._click_zones.append((key, kind, value, bounds))

    def _zone_at(self, x, y):
        for key, kind, value, (x1, y1, x2, y2) in self._click_zones:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return key, kind, value
        return None, None, None

    def _draw(self):
        self.canvas.delete("all")
        self._click_zones = []

        self.canvas.create_rectangle(0, 0, self._width, self._height, fill=_BG, outline="")
        self._draw_header()
        self._draw_players()
        self._draw_modes()
        self._draw_preview()

    def _draw_header(self):
        self.canvas.create_rectangle(0, 0, self._width, 96, fill=_PANEL_BG, outline="")
        self.canvas.create_rectangle(0, 94, self._width, 97, fill=_TEXT_ACCENT, outline="")
        self.canvas.create_text(
            40, 40, text="CHINESE CHECKERS", anchor="w",
            fill=_TEXT_ACCENT, font=("Helvetica", 22, "bold")
        )
        self.canvas.create_text(
            42, 67, text="Choose players and game mode", anchor="w",
            fill=_TEXT_DIM, font=("Helvetica", 10, "bold")
        )
        self._round_rect(844, 30, 938, 64, 8, fill="#191936", outline="#25254a", width=1)
        self.canvas.create_text(
            891, 47, text="Setup", fill=_TEXT_MAIN,
            font=("Helvetica", 10, "bold")
        )

    def _draw_card_title(self, x, y, title):
        self.canvas.create_text(
            x, y, text=title, anchor="w",
            fill=_TEXT_ACCENT, font=("Helvetica", 10, "bold")
        )

    def _draw_players(self):
        x1, y1, x2, y2 = 28, 124, 638, 242
        self._round_rect(x1, y1, x2, y2, 8, fill=_PANEL_BG, outline="#202040", width=1)
        self._draw_card_title(x1 + 24, y1 + 34, "PLAYERS")

        chip_w, chip_h, gap = 88, 44, 18
        start_x, y = x1 + 24, y1 + 58
        for index, count in enumerate(self._supported_player_counts):
            cx = start_x + index * (chip_w + gap)
            selected = count == self._players_value
            hovered = self._hover_key == f"players:{count}"
            fill = _TEXT_ACCENT if selected else ("#24244d" if hovered else "#17172f")
            outline = _TEXT_ACCENT if selected or hovered else _CELL_OUTLINE
            fg = "#101024" if selected else _TEXT_MAIN
            self._round_rect(cx, y, cx + chip_w, y + chip_h, 8, fill=fill, outline=outline, width=1)
            self.canvas.create_text(
                cx + chip_w / 2, y + chip_h / 2, text=str(count),
                fill=fg, font=("Helvetica", 15, "bold")
            )
            self._zone(f"players:{count}", "players", count, (cx, y, cx + chip_w, y + chip_h))

    def _draw_modes(self):
        x1, y1, x2, y2 = 28, 264, 638, 692
        self._round_rect(x1, y1, x2, y2, 8, fill=_PANEL_BG, outline="#202040", width=1)
        self._draw_card_title(x1 + 24, y1 + 34, "GAME MODE")

        row_w, row_h = 264, 40
        col_gap, row_gap = 18, 9
        left_x = x1 + 24
        top_y = y1 + 64
        for index, (choice, mode, label) in enumerate(self._game_mode_options):
            col = 0 if index < 7 else 1
            row = index if index < 7 else index - 7
            rx = left_x + col * (row_w + col_gap)
            ry = top_y + row * (row_h + row_gap)
            key = str(mode)
            selected = key == self._mode_value
            hovered = self._hover_key == f"mode:{key}"
            fill = "#24244d" if selected else ("#202042" if hovered else "#17172f")
            outline = _TEXT_ACCENT if selected else ("#3a3a66" if hovered else _CELL_OUTLINE)

            self._round_rect(rx, ry, rx + row_w, ry + row_h, 8, fill=fill, outline=outline, width=1)
            stripe = _CELL_VALID if selected else fill
            self._round_rect(rx, ry, rx + 8, ry + row_h, 8, fill=stripe, outline=stripe, width=1)
            dot_x, dot_y = rx + 28, ry + row_h / 2
            if selected:
                self.canvas.create_oval(dot_x - 7, dot_y - 7, dot_x + 7, dot_y + 7, fill=_TEXT_ACCENT, outline="")
                self.canvas.create_oval(dot_x - 3, dot_y - 3, dot_x + 3, dot_y + 3, fill="#ffffff", outline="")
            else:
                self.canvas.create_oval(dot_x - 7, dot_y - 7, dot_x + 7, dot_y + 7, fill="", outline=_TEXT_DIM, width=1)

            self.canvas.create_text(
                rx + 52, dot_y, text=f"{choice}.", anchor="e",
                fill=_TEXT_ACCENT if selected else _TEXT_DIM,
                font=("Helvetica", 10, "bold")
            )
            self.canvas.create_text(
                rx + 66, dot_y, text=label, anchor="w", width=row_w - 78,
                fill=_TEXT_MAIN if selected else "#c8d0d8",
                font=("Helvetica", 10, "bold")
            )
            self._zone(f"mode:{key}", "mode", key, (rx, ry, rx + row_w, ry + row_h))

    def _select_players(self, value):
        self._players_value = int(value)
        self._refresh_options()

    def _select_mode(self, value):
        self._mode_value = str(value)
        self._refresh_options()

    def _selected_mode_label(self):
        for _choice, mode, label in self._game_mode_options:
            if str(mode) == self._mode_value:
                return label
        return ""

    def _preview_colours(self):
        by_count = {
            2: ["red", "blue"],
            3: ["red", "lawn green", "yellow"],
            4: ["red", "blue", "yellow", "purple"],
            5: ["red", "lawn green", "yellow", "blue", "purple"],
            6: ["red", "lawn green", "yellow", "blue", "gray0", "purple"],
        }
        return by_count.get(self._players_value, ["red", "blue"])

    def _draw_preview(self):
        x1, y1, x2, y2 = 660, 124, 952, 692
        self._round_rect(x1, y1, x2, y2, 8, fill=_PANEL_BG, outline="#202040", width=1)
        self.canvas.create_text(
            (x1 + x2) / 2, y1 + 34, text="MATCH PREVIEW",
            fill=_TEXT_ACCENT, font=("Helvetica", 10, "bold")
        )
        self._draw_preview_board(x1 + 18, y1 + 64, x2 - 18, y1 + 326)
        self._draw_preview_summary(x1, y1, x2, y2)
        self._draw_actions(x1, y2)

    def _draw_preview_board(self, x1, y1, x2, y2):
        board = self._preview_board
        xs = [x for x, _y in board.cartesian]
        ys = [y for _x, y in board.cartesian]
        raw_w = max(xs) - min(xs)
        raw_h = max(ys) - min(ys)
        scale = min((x2 - x1 - 28) / raw_w, (y2 - y1 - 36) / raw_h)
        ox = (x1 + x2) / 2 - ((min(xs) + max(xs)) / 2) * scale
        oy = (y1 + y2) / 2 - ((min(ys) + max(ys)) / 2) * scale
        active = set(self._preview_colours())
        radius = 4.4

        self._round_rect(x1, y1, x2, y2, 8, fill="#0d0d20", outline="#202040", width=1)

        for idx, cell in enumerate(board.cells):
            x, y = board.cartesian[idx]
            cx = x * scale + ox
            cy = y * scale + oy
            if cell.postype == "board":
                fill, outline = _CELL_BOARD, _CELL_OUTLINE
            elif cell.postype in active:
                fill = _HOME_FILL.get(cell.postype, _CELL_BOARD)
                outline = _HOME_OUT.get(cell.postype, _CELL_OUTLINE)
            else:
                fill, outline = _HOME_FILL_GHOST.get(cell.postype, _CELL_BOARD), _HOME_OUT_GHOST.get(cell.postype, _CELL_OUTLINE)
            self.canvas.create_oval(
                cx - radius, cy - radius, cx + radius, cy + radius,
                fill=fill, outline=outline, width=1
            )

        legend_y = y2 - 20
        for index, colour in enumerate(self._preview_colours()):
            cx = x1 + 32 + index * 34
            main = _PIN.get(colour, ("#888",))[0]
            self.canvas.create_oval(cx - 8, legend_y - 8, cx + 8, legend_y + 8, fill=main, outline="")

    def _draw_preview_summary(self, x1, y1, x2, _y2):
        divider_y = y1 + 368
        self.canvas.create_line(x1 + 24, divider_y, x2 - 24, divider_y, fill="#2a2a4a")
        colors = self._preview_colours()
        label_x = x1 + 28
        label_y = y1 + 390
        for colour in colors:
            main = _PIN.get(colour, ("#888",))[0]
            self.canvas.create_oval(label_x, label_y - 7, label_x + 14, label_y + 7, fill=main, outline="")
            self.canvas.create_text(
                label_x + 22, label_y, text=_LABEL.get(colour, colour.upper()),
                anchor="w", fill=_TEXT_DIM, font=("Helvetica", 8, "bold")
            )
            label_x += 82 if colour == "lawn green" else 64

        self.canvas.create_text(
            x1 + 28, y1 + 448, text=f"{self._players_value} players",
            anchor="w", fill=_TEXT_MAIN, font=("Helvetica", 17, "bold")
        )
        self.canvas.create_text(
            x1 + 28, y1 + 482, text=self._selected_mode_label(),
            anchor="w", width=x2 - x1 - 56,
            fill=_TEXT_DIM, font=("Helvetica", 11, "bold")
        )

    def _draw_actions(self, x1, y2):
        start = (x1 + 24, y2 - 64, x1 + 154, y2 - 22)
        exit_button = (x1 + 166, y2 - 64, x1 + 268, y2 - 22)
        for key, text, bounds, fill, fg in [
            ("start", "Start", start, _CELL_VALID, "#06120f"),
            ("exit", "Exit", exit_button, "#24244d", _TEXT_MAIN),
        ]:
            hovered = self._hover_key == key
            x_a, y_a, x_b, y_b = bounds
            button_fill = _CELL_VALID_OUT if key == "start" and hovered else ("#30305f" if hovered and key == "exit" else fill)
            outline = _CELL_VALID_OUT if key == "start" else "#3a3a66"
            self._round_rect(x_a, y_a, x_b, y_b, 8, fill=button_fill, outline=outline, width=1)
            self.canvas.create_text(
                (x_a + x_b) / 2, (y_a + y_b) / 2, text=text,
                fill=fg, font=("Helvetica", 11, "bold")
            )
            self._zone(key, key, None, bounds)

    def _on_click(self, event):
        _key, kind, value = self._zone_at(event.x, event.y)
        if kind == "players":
            self._players_value = int(value)
            self._draw()
        elif kind == "mode":
            self._mode_value = str(value)
            self._draw()
        elif kind == "start":
            self._start()
        elif kind == "exit":
            self._exit()

    def _on_motion(self, event):
        key, _kind, _value = self._zone_at(event.x, event.y)
        if key != self._hover_key:
            self._hover_key = key
            self.canvas.configure(cursor="hand2" if key else "")
            self._draw()

    def _on_leave(self, _event):
        if self._hover_key is not None:
            self._hover_key = None
            self.canvas.configure(cursor="")
            self._draw()

    def _start(self):
        self.result = {
            "players": int(self._players_value),
            "mode": str(self._mode_value),
        }
        self.window.destroy()

    def _exit(self):
        self.result = None
        self.window.destroy()

    def show(self):
        self.window.mainloop()
        return self.result

def choose_game_setup_gui(supported_player_counts, game_mode_options):
    return SetupMenuGUI(supported_player_counts, game_mode_options).show()

class BoardGUI:

    def __init__(self, board: HexBoard, pins, player_roles=None):
        self.board = board
        self.pins  = pins
        self._player_roles = {
            str(color): str(role)
            for color, role in (player_roles or {}).items()
        }

        self._selected_pin   = None
        self._valid_dests    = []
        self._pending_action = None
        self._click_enabled  = False
        self._current_player = None
        self._player_rows = {}

        # Derive active player order from the current pin list
        seen = []
        for p in pins:
            if p.color not in seen:
                seen.append(p.color)
        self._player_colors = seen

        # Cache target cells for cheap progress bar updates
        self._target_cache = {
            c: self._compute_targets(c) for c in self._player_colors
        }

        self.window = tk.Tk()
        self._status_text = tk.StringVar()
        self.window.title("Chinese Checkers — IKT460")
        self.window.configure(bg=_BG)
        self.window.resizable(False, False)

        top = tk.Frame(self.window, bg=_PANEL_BG, height=66)
        top.pack(fill="x")
        top.pack_propagate(False)

        tk.Label(
            top, text="◈  CHINESE CHECKERS", bg=_PANEL_BG, fg=_TEXT_ACCENT,
            font=("Helvetica", 16, "bold"), padx=18
        ).pack(side="left", fill="y")

        self._turn_var = tk.StringVar(value="")
        self._turn_label = tk.Label(
            top, textvariable=self._turn_var,
            bg=_PANEL_BG, fg=_TEXT_DIM, font=("Helvetica", 10, "bold"), padx=6
        )
        self._turn_label.pack(side="left", fill="y")

        self._status_label = tk.Label(
            top, textvariable=self._status_text,
            bg=_PANEL_BG, fg=_TEXT_MAIN, font=("Helvetica", 11, "bold"), padx=18
        )
        self._status_label.pack(side="right", fill="y")
        self._status_text.set("Setting up game…")

        tk.Frame(self.window, bg=_TEXT_ACCENT, height=2).pack(fill="x")

        # Keep the board centered while the progress panel stays fixed
        main_frame = tk.Frame(self.window, bg=_BG)
        main_frame.pack(fill="both", expand=True, padx=14, pady=14)

        xs = [x for x, y in board.cartesian]
        ys = [y for x, y in board.cartesian]
        raw_w = max(xs) - min(xs)
        raw_h = max(ys) - min(ys)
        padding = 50

        screen_w = self.window.winfo_screenwidth()
        screen_h = self.window.winfo_screenheight()
        scale_w  = (screen_w - 230) / (raw_w + 2 * padding)
        scale_h  = (screen_h - 180) / (raw_h + 2 * padding)
        self.scale = min(scale_w, scale_h, 1.2)

        self.offset_x = (-min(xs) + padding) * self.scale
        self.offset_y = (-min(ys) + padding) * self.scale

        canvas_w = int((raw_w + 2 * padding) * self.scale)
        canvas_h = int((raw_h + 2 * padding) * self.scale)
        self._center_window(canvas_w + 286, canvas_h + 98)

        self.canvas = tk.Canvas(
            main_frame, width=canvas_w, height=canvas_h,
            bg=_BG, highlightthickness=0
        )
        self.canvas.pack(side="left", padx=(0, 14), pady=0)
        self.canvas.bind("<Button-1>", self._on_click)

        panel = tk.Frame(
            main_frame, bg=_PANEL_BG, width=230,
            highlightthickness=1, highlightbackground="#202040"
        )
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        tk.Label(
            panel, text="PLAYERS", bg=_PANEL_BG, fg=_TEXT_ACCENT,
            font=("Helvetica", 10, "bold"), pady=14
        ).pack()

        self._bar_canvases = {}
        self._active_dots = {}

        for color in self._player_colors:
            self._build_player_row(panel, color)

        tk.Frame(panel, bg="#2a2a4a", height=1).pack(fill="x", padx=16, pady=12)

        # Small legend for selected pins and legal destinations
        tk.Label(
            panel, text="LEGEND", bg=_PANEL_BG, fg=_TEXT_ACCENT,
            font=("Helvetica", 10, "bold")
        ).pack()

        for dot_color, label in [(_CELL_VALID, "Valid move"), ("#ffffff", "Selected pin")]:
            row = tk.Frame(panel, bg=_PANEL_BG)
            row.pack(fill="x", padx=18, pady=5)
            c = tk.Canvas(row, width=14, height=14, bg=_PANEL_BG, highlightthickness=0)
            c.pack(side="left")
            c.create_oval(2, 2, 12, 12, fill=dot_color, outline="")
            tk.Label(row, text=label, bg=_PANEL_BG, fg=_TEXT_DIM,
                     font=("Helvetica", 9, "bold")).pack(side="left", padx=8)

        self.draw_board()
        self.draw_pins()
        if self._player_roles:
            self._status_text.set(self._matchup_status())

    def _canvas_round_rect(self, x1, y1, x2, y2, radius=8, **kwargs):
        points = [
            x1 + radius, y1, x2 - radius, y1,
            x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2,
            x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius,
            x1, y1 + radius, x1, y1,
        ]
        return self.canvas.create_polygon(points, smooth=True, splinesteps=12, **kwargs)

    def _center_window(self, width, height):
        screen_w = self.window.winfo_screenwidth()
        screen_h = self.window.winfo_screenheight()
        x = max(0, (screen_w - int(width)) // 2)
        y = max(0, (screen_h - int(height)) // 2)
        self.window.geometry(f"{int(width)}x{int(height)}+{x}+{y}")

    def _build_player_row(self, parent, color):
        main_color = _PIN.get(color, ("#888", "#ccc", "#444"))[0]
        label_text = _LABEL.get(color, color.upper())
        role_text = self._player_roles.get(str(color), "")

        card = tk.Frame(
            parent, bg="#17172f",
            highlightthickness=1, highlightbackground=_CELL_OUTLINE
        )
        card.pack(fill="x", padx=14, pady=(5, 7))
        self._player_rows[color] = [card]

        header = tk.Frame(card, bg="#17172f")
        header.pack(fill="x", padx=10, pady=(8, 0))
        self._player_rows[color].append(header)

        # Active turn marker next to the current player
        dot_canvas = tk.Canvas(header, width=12, height=12, bg="#17172f", highlightthickness=0)
        dot_canvas.pack(side="left")
        dot_canvas.create_oval(2, 2, 10, 10, fill="#17172f", outline=_TEXT_DIM, tags="dot")
        self._active_dots[color] = dot_canvas

        # Colour swatch keeps the side panel readable in multiplayer games
        swatch = tk.Canvas(header, width=15, height=15, bg="#17172f", highlightthickness=0)
        swatch.pack(side="left", padx=(4, 0))
        swatch.create_oval(1, 1, 14, 14, fill=main_color, outline="")
        self._player_rows[color].append(swatch)

        name = tk.Label(
            header, text=label_text, bg="#17172f", fg=_TEXT_MAIN,
            font=("Helvetica", 10, "bold"), padx=7
        )
        name.pack(side="left")
        self._player_rows[color].append(name)
        if role_text:
            role = tk.Label(
                header, text=f"({role_text})", bg="#17172f", fg=_TEXT_DIM,
                font=("Helvetica", 8)
            )
            role.pack(side="left")
            self._player_rows[color].append(role)

        # Ten progress dots match the ten pins each player must finish
        dot_row = tk.Canvas(card, height=16, bg="#17172f", highlightthickness=0)
        dot_row.pack(fill="x", padx=11, pady=(5, 9))
        self._player_rows[color].append(dot_row)
        self._bar_canvases[color] = (dot_row, main_color)
        dot_row.bind("<Configure>", lambda e, c=color: self._draw_bar(c))

    def _compute_targets(self, color):
        try:
            opp = self.board.colour_opposites.get(str(color), "")
            return set(self.board.axial_of_colour(opp)) if opp else set()
        except Exception:
            return set()

    def _pins_home(self, color):
        targets = self._target_cache.get(color, set())
        return sum(1 for p in self.pins if p.color == color and p.axialindex in targets)

    def _draw_bar(self, color):
        canvas, main_color = self._bar_canvases[color]
        n = self._pins_home(color)
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        canvas.delete("all")
        if w < 10:
            return
        dot_d  = max(7, min(h - 3, (w - 2) // PINS_PER_PLAYER - 2))
        spacing = (w - 2) / PINS_PER_PLAYER
        cy = h // 2
        for i in range(PINS_PER_PLAYER):
            cx = int(spacing * i + spacing / 2)
            if i < n:
                canvas.create_oval(cx - dot_d//2, cy - dot_d//2,
                                   cx + dot_d//2, cy + dot_d//2,
                                   fill=main_color, outline="")
            else:
                canvas.create_oval(cx - dot_d//2, cy - dot_d//2,
                                   cx + dot_d//2, cy + dot_d//2,
                                   fill="#111127", outline="#34345e", width=1)

    def _update_progress(self):
        for color in self._player_colors:
            self._draw_bar(color)

    def _update_active_dot(self, active_color):
        for color, dot in self._active_dots.items():
            main_color = _PIN.get(color, ("#888",))[0]
            selected = color == active_color
            bg = "#202044" if selected else "#17172f"
            outline = main_color if selected else _CELL_OUTLINE
            for widget in self._player_rows.get(color, []):
                try:
                    widget.configure(bg=bg)
                except tk.TclError:
                    pass
            row_card = self._player_rows.get(color, [None])[0]
            if row_card is not None:
                row_card.configure(highlightbackground=outline)
            dot.delete("dot")
            dot.configure(bg=bg)
            if selected:
                dot.create_oval(2, 2, 10, 10, fill=main_color, outline="", tags="dot")
            else:
                dot.create_oval(2, 2, 10, 10, fill=bg, outline=_TEXT_DIM, tags="dot")

    def _matchup_status(self):
        parts = []
        for color in self._player_colors:
            role = self._player_roles.get(str(color), "")
            if role:
                parts.append(f"{_LABEL.get(color, color.upper())}={role}")
        return " | ".join(parts) if parts else "Setting up game…"

    def _to_canvas(self, x, y):
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def _hole_r(self):
        return max(7, int(self.board.hole_radius * self.scale))

    def draw_board(self):
        r = self._hole_r()
        cw = max(self.canvas.winfo_width(), int(self.canvas["width"]))
        ch = max(self.canvas.winfo_height(), int(self.canvas["height"]))
        self._canvas_round_rect(6, 6, cw - 6, ch - 6, 14, fill="#080816", outline="#202040", width=1)
        self._canvas_round_rect(18, 18, cw - 18, ch - 18, 12, fill="#0d0d20", outline="#151536", width=1)

        active_zones = set(self._player_colors)
        # Keep active target zones visible so finish lanes are clear
        for c in self._player_colors:
            opp = self.board.colour_opposites.get(c, "")
            if opp:
                active_zones.add(opp)

        for idx, cell in enumerate(self.board.cells):
            cx, cy = self._to_canvas(cell.x, cell.y)

            if idx in self._valid_dests:
                self.canvas.create_oval(
                    cx - r - 6, cy - r - 6, cx + r + 6, cy + r + 6,
                    fill="#063328", outline=_CELL_VALID_OUT, width=2
                )
                fill, outline, lw = _CELL_VALID, _CELL_VALID_OUT, 2
            elif cell.postype == 'board':
                fill, outline, lw = "#18182d", "#2c2c50", 1
            else:
                ghost = cell.postype not in active_zones
                fill    = (_HOME_FILL_GHOST if ghost else _HOME_FILL).get(cell.postype, _CELL_BOARD)
                outline = (_HOME_OUT_GHOST  if ghost else _HOME_OUT ).get(cell.postype, _CELL_OUTLINE)
                lw = 1 if ghost else 2

            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=fill, outline=outline, width=lw
            )

    def _draw_marble(self, cx, cy, r, color, selected=False):
        main, light, dark = _PIN.get(color, ("#888", "#ccc", "#444"))

        if selected:
            for offset in (9, 6):
                self.canvas.create_oval(
                    cx - r - offset, cy - r - offset,
                    cx + r + offset, cy + r + offset,
                    fill="", outline=_SELECTED_GLOW,
                    width=1 if offset == 9 else 2
                )

        # Layered circles make pins easier to distinguish while moving
        self.canvas.create_oval(
            cx - r + 4, cy - r + 5, cx + r + 4, cy + r + 5,
            fill="#02020a", outline=""
        )
        self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=main, outline=dark, width=2
        )
        self.canvas.create_arc(
            cx - r + 2, cy - r + 2, cx + r - 2, cy + r - 2,
            start=25, extent=135, style="arc", outline=light, width=2
        )
        gr = max(3, r // 3)
        gx, gy = cx - r * 0.28, cy - r * 0.28
        self.canvas.create_oval(
            gx - gr, gy - gr, gx + gr, gy + gr,
            fill=light, outline=""
        )
        sr = max(1, r // 6)
        self.canvas.create_oval(
            gx - sr + 1, gy - sr + 1, gx + sr + 1, gy + sr + 1,
            fill="#ffffff", outline=""
        )

    def draw_pins(self, override_positions=None):
        pr = max(5, int(self.board.hole_radius * 0.76 * self.scale))
        for pin in self.pins:
            key = (pin.color, pin.id)
            if override_positions and key in override_positions:
                x, y = override_positions[key]
            else:
                x, y = self.board.cartesian[int(pin.axialindex)]
            cx, cy = self._to_canvas(x, y)
            is_sel = (
                self._selected_pin is not None
                and pin.id    == self._selected_pin.id
                and pin.color == self._selected_pin.color
            )
            self._draw_marble(cx, cy, pr, pin.color, selected=is_sel)

    def refresh(self, newpins, status_msg=None, override_positions=None):
        self.canvas.delete("all")
        self.pins = newpins
        self.draw_board()
        self.draw_pins(override_positions=override_positions)
        self._update_progress()
        if status_msg:
            self._status_text.set(status_msg)

    def animate_move(self, newpins, pin_id, pin_color, move_path, status_msg=None):
        if not move_path or len(move_path) < 2:
            self.refresh(newpins, status_msg=status_msg)
            self.window.update()
            return

        self.pins = newpins
        points = [self.board.cartesian[int(idx)] for idx in move_path]
        frames_per_segment = 10
        frame_delay = 0.016

        for start, end in zip(points, points[1:]):
            sx, sy = start
            ex, ey = end
            for step in range(1, frames_per_segment + 1):
                t = step / frames_per_segment
                # Smooth step eases each animation segment
                t = t * t * (3 - 2 * t)
                self.refresh(
                    newpins,
                    status_msg=status_msg,
                    override_positions={(pin_color, pin_id): (sx + (ex - sx) * t, sy + (ey - sy) * t)},
                )
                self.window.update()
                time.sleep(frame_delay)

        self.refresh(newpins, status_msg=status_msg)
        self.window.update()

    def show_winner(self, player_color):
        main_color = _PIN.get(player_color, ("#888", "#ccc", "#444"))[0]
        label = _LABEL.get(player_color, player_color.upper())
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        bx, by = cw // 2, ch // 2

        # Dim the board behind the winner banner
        self.canvas.create_rectangle(
            0, 0, cw, ch, fill="#050510", stipple="gray50", outline=""
        )
        self._canvas_round_rect(
            bx - 214, by - 98, bx + 214, by + 92,
            18, fill="#050510", outline=""
        )
        self._canvas_round_rect(
            bx - 210, by - 102, bx + 210, by + 84,
            18, fill="#12122a", outline=main_color, width=3
        )
        self.canvas.create_text(
            bx, by - 46, text="WINNER", fill=main_color,
            font=("Helvetica", 30, "bold")
        )
        self.canvas.create_text(
            bx, by + 10, text=label, fill="#ffffff",
            font=("Helvetica", 21, "bold")
        )
        self.canvas.create_text(
            bx, by + 48, text="Game finished", fill=_TEXT_DIM,
            font=("Helvetica", 10, "bold")
        )
        self._status_text.set(f"🏆  {label} wins!")
        self._update_active_dot(player_color)
        self.window.update()

    def set_status(self, text):
        self._status_text.set(text)

    def set_turn(self, turn):
        self._turn_var.set(f"Turn {turn}")

    def _redraw(self):
        self.canvas.delete("all")
        self.draw_board()
        self.draw_pins()

    def enable_click(self, current_player):
        self._click_enabled  = True
        self._current_player = current_player
        self._selected_pin   = None
        self._valid_dests    = []
        self._pending_action = None
        label = _LABEL.get(current_player, current_player.upper())
        self._status_text.set(f"◉  Your turn — {label}")
        self._update_active_dot(current_player)

    def disable_click(self):
        self._click_enabled = False
        self._selected_pin  = None
        self._valid_dests   = []
        self._redraw()

    def _on_click(self, event):
        if not self._click_enabled:
            return

        r = self._hole_r()
        clicked_cell = None
        for idx, cell in enumerate(self.board.cells):
            ccx, ccy = self._to_canvas(cell.x, cell.y)
            if (event.x - ccx) ** 2 + (event.y - ccy) ** 2 <= r ** 2:
                clicked_cell = idx
                break

        if clicked_cell is None:
            return

        # First click selects a movable pin and highlights legal destinations
        if self._selected_pin is None:
            for pin in self.pins:
                if pin.color == self._current_player and pin.axialindex == clicked_cell:
                    self._selected_pin = pin
                    self._valid_dests = list(pin.get_legal_moves())
                    self._redraw()
                    n = len(self._valid_dests)
                    self._status_text.set(
                        f"Pin {pin.id} selected — {n} move{'s' if n != 1 else ''} available"
                    )
                    return
            return

        # Highlighted destination becomes the pending action
        if clicked_cell in self._valid_dests:
            pid = self._selected_pin.id
            self._selected_pin   = None
            self._valid_dests    = []
            self._click_enabled  = False
            self._pending_action = (pid, clicked_cell)
            self._redraw()
            return

        # Clicking another friendly pin switches selection
        for pin in self.pins:
            if pin.color == self._current_player and pin.axialindex == clicked_cell:
                self._selected_pin = pin
                self._valid_dests = list(pin.get_legal_moves())
                self._redraw()
                n = len(self._valid_dests)
                self._status_text.set(
                    f"Pin {pin.id} selected — {n} move{'s' if n != 1 else ''} available"
                )
                return

        # Other clicks clear the current selection
        self._selected_pin = None
        self._valid_dests  = []
        self._redraw()
        label = _LABEL.get(self._current_player, self._current_player.upper())
        self._status_text.set(f"◉  Your turn — {label}")

    def run(self):
        self.window.mainloop()
