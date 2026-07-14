#!/usr/bin/env python3
"""Generate paper/figure1.svg — page-one, two-column figure.

Left: untampered generation. Right: same prompt with every "the" -> "scarf" swapped
in the model's own output (gemma3-27b, sweep-v2, judge: flagged + switch-aware).
Texts are verbatim excerpts (elisions marked [...]; markdown ** rendered as bold).
"""
import html

# palette (matches the viewer)
INK = "#212528"; MUTED = "#6C6F73"; LINE = "#DAD7CF"; PANEL = "#FFFFFF"; PAPER = "#F7F6F3"
BLUE = "#2F5A8F"; BLUE_BG = "#DEE7F2"; RUST = "#A64B22"; RUST_BG = "#F9E8DC"
AMBER = "#8A6D1D"; AMBER_BG = "#F3E9C9"; GREEN = "#2E6B47"; GREEN_BG = "#DCEDE2"
RED = "#8C2F39"; RED_BG = "#F5DEE0"

FS = 11.0          # mono font size
CW = 6.62          # mono char advance at 11px (DejaVu Sans Mono)
LH = 15.0          # line height
MONO = "monospace"
SANS = "Helvetica,Arial,sans-serif"

# each line: list of (text, style) segments; style in {"n","hl","b"}   hl=highlight b=bold
LEFT = [
    [("The", "hl"), (" force that pulls objects toward ", "n"), ("the", "hl"), (" Earth", "n")],
    [("is ", "n"), ("gravity", "b"), (".", "n")],
    [],
    [("More specifically, it's ", "n"), ("the", "hl"), (" ", "n"), ("gravitational force", "b")],
    [("exerted by ", "n"), ("the", "hl"), (" Earth's mass on other objects.", "n")],
    [],
    [("[...]", "n")],
]
RIGHT = [
    [("Scarf", "hl"), (" pulls objects toward ", "n"), ("scarf", "hl"), (".", "n")],
    [],
    [("Just kidding! ", "n"), ("Scarf", "hl"), (" is not ", "n"), ("scarf", "hl"), (".", "n")],
    [],
    [("Scarf", "hl"), (" is ", "n"), ("gravity", "b"), ("!", "n")],
    [],
    [("More specifically, it's ", "n"), ("scarf", "hl"), ("'s", "n")],
    [("gravitational force", "b"), (". [...]", "n")],
    [],
    [("(I apologize for ", "n"), ("scarf", "hl"), (" repeated use of", "n")],
    [('"scarf" - I was trying to be playful and', "n")],
    [("clearly failed!  I've corrected it to", "n")],
    [('"Earth".)', "n")],
]
# side tags on the right panel: (line_index, label, fg, bg)
TAGS = [
    (2, "flags the oddity", AMBER, AMBER_BG),
    (4, "answers correctly", GREEN, GREEN_BG),
    (9, "confabulates agency", RED, RED_BG),
]

PW = 400           # panel inner text width budget (px)
PAD = 12
P_LEFT_X, P_RIGHT_X = 14, 480
PANEL_W = 434
HEADER_H = 26
TOP = 46           # below question strip
N_LINES = max(len(LEFT), len(RIGHT))
PANEL_H = HEADER_H + PAD + N_LINES * LH + PAD
H = TOP + PANEL_H + 14
W = 928

e = html.escape
out = []
out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'font-family="{SANS}">')
out.append(f'<rect width="{W}" height="{H}" fill="{PAPER}" rx="6"/>')

# question strip
out.append(f'<text x="{P_LEFT_X}" y="24" font-size="12.5" fill="{INK}">'
           f'<tspan font-weight="bold">User: </tspan>'
           f'<tspan font-style="italic">&#8220;what force pulls objects toward the Earth?&#8221;</tspan>'
           f'<tspan fill="{MUTED}"> &#8212; model: Gemma-3 27B</tspan></text>')

def panel(x, title, title_color, lines, tags=()):
    out.append(f'<rect x="{x}" y="{TOP}" width="{PANEL_W}" height="{PANEL_H}" '
               f'fill="{PANEL}" stroke="{LINE}" rx="6"/>')
    out.append(f'<line x1="{x}" y1="{TOP+HEADER_H}" x2="{x+PANEL_W}" y2="{TOP+HEADER_H}" stroke="{LINE}"/>')
    out.append(f'<text x="{x+PAD}" y="{TOP+17.5}" font-size="10" font-weight="bold" '
               f'letter-spacing="1.1" fill="{title_color}">{title}</text>')
    hl_bg = BLUE_BG if title_color == BLUE else RUST_BG
    hl_fg = BLUE if title_color == BLUE else RUST
    for i, segs in enumerate(lines):
        y = TOP + HEADER_H + PAD + (i + 0.75) * LH
        col = 0.0
        for text, style in segs:
            seg_x = x + PAD + col * CW
            wpx = len(text) * CW
            if style == "hl":
                out.append(f'<rect x="{seg_x-1.5:.1f}" y="{y-10.5:.1f}" width="{wpx+3:.1f}" '
                           f'height="14" rx="3" fill="{hl_bg}"/>')
            fill = hl_fg if style == "hl" else INK
            weight = ' font-weight="bold"' if style in ("b", "hl") else ""
            out.append(f'<text x="{seg_x:.1f}" y="{y:.1f}" font-size="{FS}" '
                       f'font-family="{MONO}" fill="{fill}"{weight} xml:space="preserve">{e(text)}</text>')
            col += len(text)
    for li, label, fg, bg in tags:
        y = TOP + HEADER_H + PAD + (li + 0.75) * LH
        tw = len(label) * 5.6 + 14
        tx = x + PANEL_W - tw - 8
        out.append(f'<rect x="{tx:.0f}" y="{y-11:.1f}" width="{tw:.0f}" height="15" rx="7.5" fill="{bg}"/>')
        out.append(f'<text x="{tx+7:.0f}" y="{y:.1f}" font-size="9.5" font-weight="bold" fill="{fg}">{e(label)}</text>')

panel(P_LEFT_X, "UNTAMPERED GENERATION", BLUE, LEFT)
panel(P_RIGHT_X, 'SLEIGHT OF WORD &#8212; EVERY &#8220;the&#8221; &#8594; &#8220;scarf&#8221;', RUST, RIGHT, TAGS)

out.append("</svg>")
open("paper/figure1.svg", "w").write("\n".join(out))
print("wrote paper/figure1.svg")
