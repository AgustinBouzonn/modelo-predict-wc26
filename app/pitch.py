"""Render de canchas de fútbol en SVG con avatares de jugadores (fotos de Wikipedia)."""
from __future__ import annotations

from app.teams_visual import team_color


def _last_name(name: str) -> str:
    parts = str(name).split()
    return (parts[-1] if parts else str(name))[:12]


def _num(v) -> str:
    try:
        return str(int(v))
    except (ValueError, TypeError):
        return ""


def _is_light(hex_color: str) -> bool:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return False
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) > 150


def _stripes(W, H, n=10):
    out = [f'<rect width="{W}" height="{H}" fill="#15803d"/>']
    step = H / n
    for i in range(0, n, 2):
        out.append(f'<rect y="{i*step:.1f}" width="{W}" height="{step:.1f}" fill="#fff" fill-opacity="0.045"/>')
    return "".join(out)


def _lines(W, H):
    s = 'stroke="#ffffff" stroke-opacity="0.7" fill="none" stroke-width="2"'
    cx = W / 2
    pw, ph = W * 0.52, H * 0.16
    gw, gh = W * 0.26, H * 0.06
    return "".join([
        f'<rect x="7" y="7" width="{W-14}" height="{H-14}" rx="4" {s}/>',
        f'<line x1="7" y1="{H/2}" x2="{W-7}" y2="{H/2}" {s}/>',
        f'<circle cx="{cx}" cy="{H/2}" r="{W*0.13:.0f}" {s}/>',
        f'<circle cx="{cx}" cy="{H/2}" r="2.5" fill="#fff" fill-opacity="0.8"/>',
        f'<rect x="{cx-pw/2:.0f}" y="7" width="{pw:.0f}" height="{ph:.0f}" {s}/>',
        f'<rect x="{cx-gw/2:.0f}" y="7" width="{gw:.0f}" height="{gh:.0f}" {s}/>',
        f'<rect x="{cx-pw/2:.0f}" y="{H-7-ph:.0f}" width="{pw:.0f}" height="{ph:.0f}" {s}/>',
        f'<rect x="{cx-gw/2:.0f}" y="{H-7-gh:.0f}" width="{gw:.0f}" height="{gh:.0f}" {s}/>',
        f'<circle cx="{cx}" cy="{ph*0.72:.0f}" r="2" fill="#fff" fill-opacity="0.8"/>',
        f'<circle cx="{cx}" cy="{H-ph*0.72:.0f}" r="2" fill="#fff" fill-opacity="0.8"/>',
    ])


def _avatar(cx, cy, color, num, name, img, uid):
    """Avatar circular: foto del jugador (si hay) o color del equipo + dorsal."""
    txt = "#0f172a" if _is_light(color) else "#ffffff"
    r = 16
    sh = f'<ellipse cx="{cx:.0f}" cy="{cy+20:.0f}" rx="13" ry="3.5" fill="#000" fill-opacity="0.25"/>'
    pill = (f'<rect x="{cx-32:.0f}" y="{cy+18:.0f}" width="64" height="14" rx="3" fill="#0f172a" fill-opacity="0.66"/>'
            f'<text x="{cx:.0f}" y="{cy+28:.0f}" text-anchor="middle" font-size="9.5" fill="#f8fafc" font-weight="600">{_last_name(name)}</text>')
    if img:
        body = (
            f'<defs><clipPath id="{uid}"><circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r}"/></clipPath></defs>'
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r+1.5}" fill="{color}"/>'
            f'<image href="{img}" x="{cx-r:.0f}" y="{cy-r:.0f}" width="{2*r}" height="{2*r}" '
            f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{uid})"/>'
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r}" fill="none" stroke="#ffffff" stroke-width="1.6"/>'
            f'<circle cx="{cx+11:.0f}" cy="{cy+11:.0f}" r="7.5" fill="{color}" stroke="#fff" stroke-width="1"/>'
            f'<text x="{cx+11:.0f}" y="{cy+14:.0f}" text-anchor="middle" font-size="8" font-weight="800" fill="{txt}">{num}</text>')
    else:
        body = (
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r}" fill="{color}" stroke="#ffffff" stroke-width="1.6"/>'
            f'<text x="{cx:.0f}" y="{cy+5:.0f}" text-anchor="middle" font-size="13" font-weight="800" fill="{txt}">{num}</text>')
    return sh + body + pill


def pitch_match_svg(home_xi, away_xi, home, away, W=440, H=660):
    ch, ca = team_color(home), team_color(away)
    nodes = []
    if home_xi is not None and not home_xi.empty:
        for i, r in enumerate(home_xi.itertuples(index=False)):
            nodes.append(_avatar(r.slot_x * W, (0.5 + r.slot_y * 0.5) * H, ch,
                                  _num(getattr(r, "number", "")), r.player,
                                  getattr(r, "player_img", ""), f"h{i}"))
    if away_xi is not None and not away_xi.empty:
        for i, r in enumerate(away_xi.itertuples(index=False)):
            nodes.append(_avatar((1 - r.slot_x) * W, (0.5 - r.slot_y * 0.5) * H, ca,
                                  _num(getattr(r, "number", "")), r.player,
                                  getattr(r, "player_img", ""), f"a{i}"))
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:500px;border-radius:14px;'
            f'box-shadow:0 6px 24px rgba(0,0,0,.35)">{_stripes(W,H)}{_lines(W,H)}{"".join(nodes)}</svg>')


def pitch_team_svg(xi, team, W=380, H=500):
    color = team_color(team)
    nodes = []
    if xi is not None and not xi.empty:
        for i, r in enumerate(xi.itertuples(index=False)):
            nodes.append(_avatar(r.slot_x * W, r.slot_y * H, color,
                                  _num(getattr(r, "number", "")), r.player,
                                  getattr(r, "player_img", ""), f"t{i}"))
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:420px;border-radius:14px;'
            f'box-shadow:0 6px 24px rgba(0,0,0,.35)">{_stripes(W,H)}{_lines(W,H)}{"".join(nodes)}</svg>')
