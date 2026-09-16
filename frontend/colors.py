class Colors:
    # Backgrounds
    BG_1 = "#06101C"        # window background
    BG_2 = "#1F2A38"        # card / sidebar background
    BG_3 = "#141D26"        # recessed surface (inputs, plot canvases)
    BORDER = "#2A3949"      # hairline borders

    # Text
    TEXT = "#FFFFFF"
    TEXT_DIM = "#9AA7B4"    # secondary ink
    TEXT_MUTED = "#5C6773"  # muted / axis labels

    # Accent (promoted from ExperimentWindow's old hardcoded start-button blue)
    ACCENT = "#87CEEB"
    ACCENT_HOVER = "#A8DDF2"
    ACCENT_PRESSED = "#6BBBDD"

    # Status
    GOOD = "#3ED598"
    BAD = "#FF5C7A"
    WARN = "#FE8A27"

    # Plot chrome
    GRID = "#22303E"

    # Gradients
    GRADIENT_1_START = "#CA0095"
    GRADIENT_1_END = "#992B96"

    GRADIENT_2_START = "#F03282"
    GRADIENT_2_END = "#FE8A27"


class Fonts:
    # QSS font-family fallback list (same syntax as CSS) — tried in order
    # against what's actually installed. Applied app-wide via main.py's
    # global QWidget stylesheet; per-widget QFont() calls elsewhere only
    # set size/bold, so this family choice cascades to them too.
    FAMILY = 'Roboto, Inter, "Helvetica Neue", "Arial Nova", Arial, sans-serif'
