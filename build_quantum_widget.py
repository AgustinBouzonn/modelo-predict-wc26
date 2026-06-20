"""Genera el demo interactivo del Predictor Cuantico a partir de
data/processed/quantum_demo.json.

Produce:
  - quantum_predictor.html         (pagina autonoma para abrir en el navegador / CV)
  - results_widget_fragment.html   (fragmento para incrustar en el chat)

El demo usa banderas SVG reales (flag-icons via CDN) porque los emoji de
bandera no se renderizan en Windows.
"""

import json

with open("data/processed/quantum_demo.json", encoding="utf-8") as f:
    D = json.load(f)

# Codigos ISO 3166-1 alpha-2 para flag-icons (England/Scotland: subdivisiones GB)
ISO = {
    "Algeria": "dz", "Argentina": "ar", "Australia": "au", "Austria": "at",
    "Belgium": "be", "Bosnia and Herzegovina": "ba", "Brazil": "br",
    "Cabo Verde": "cv", "Canada": "ca", "Colombia": "co", "Croatia": "hr",
    "Curaçao": "cw", "Czech Republic": "cz", "Côte d'Ivoire": "ci",
    "DR Congo": "cd", "Ecuador": "ec", "Egypt": "eg", "England": "gb-eng",
    "France": "fr", "Germany": "de", "Ghana": "gh", "Haiti": "ht",
    "Iran": "ir", "Iraq": "iq", "Japan": "jp", "Jordan": "jo",
    "Korea Republic": "kr", "Mexico": "mx", "Morocco": "ma", "Netherlands": "nl",
    "New Zealand": "nz", "Norway": "no", "Panama": "pa", "Paraguay": "py",
    "Portugal": "pt", "Qatar": "qa", "Saudi Arabia": "sa", "Scotland": "gb-sct",
    "Senegal": "sn", "South Africa": "za", "Spain": "es", "Sweden": "se",
    "Switzerland": "ch", "Tunisia": "tn", "Turkey": "tr", "United States": "us",
    "Uruguay": "uy", "Uzbekistan": "uz",
}

# Nombres en espanol para mostrar
ES = {
    "Algeria": "Argelia", "Australia": "Australia", "Austria": "Austria",
    "Belgium": "Bélgica", "Bosnia and Herzegovina": "Bosnia", "Brazil": "Brasil",
    "Cabo Verde": "Cabo Verde", "Canada": "Canadá", "Colombia": "Colombia",
    "Croatia": "Croacia", "Curaçao": "Curazao", "Czech Republic": "Chequia",
    "Côte d'Ivoire": "Costa de Marfil", "DR Congo": "RD Congo", "Ecuador": "Ecuador",
    "Egypt": "Egipto", "England": "Inglaterra", "France": "Francia",
    "Germany": "Alemania", "Ghana": "Ghana", "Haiti": "Haití", "Iran": "Irán",
    "Iraq": "Irak", "Japan": "Japón", "Jordan": "Jordania",
    "Korea Republic": "Corea del Sur", "Mexico": "México", "Morocco": "Marruecos",
    "Netherlands": "Países Bajos", "New Zealand": "Nueva Zelanda", "Norway": "Noruega",
    "Panama": "Panamá", "Paraguay": "Paraguay", "Portugal": "Portugal",
    "Qatar": "Catar", "Saudi Arabia": "Arabia Saudita", "Scotland": "Escocia",
    "Senegal": "Senegal", "South Africa": "Sudáfrica", "Spain": "España",
    "Sweden": "Suecia", "Switzerland": "Suiza", "Tunisia": "Túnez",
    "Turkey": "Turquía", "United States": "Estados Unidos", "Uruguay": "Uruguay",
    "Uzbekistan": "Uzbekistán", "Argentina": "Argentina",
}

# --- Downsample de la grilla 60x60 -> 20x20 para un embed liviano ---
G = D["grid"]
STEP = 3
sel = list(range(0, G, STEP))
gx = [D["gx"][i] for i in sel]
gy = [D["gy"][i] for i in sel]
z = [[D["z"][i][j] for j in sel] for i in sel]
zmax = max(abs(v) for row in z for v in row)

teams = []
for t in D["teams"]:
    teams.append({
        "n": ES.get(t["name"], t["name"]),
        "iso": ISO.get(t["name"], "un"),
        "elo": t["elo"],
        "form": t["form_pts"],
    })
teams.sort(key=lambda d: d["n"])

DATA = {
    "teams": teams, "gx": gx, "gy": gy, "z": z, "zmax": round(zmax, 4),
    "acc": D["test_acc"], "n": D["n_matches"],
}
data_js = json.dumps(DATA, ensure_ascii=False)

BODY = """<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/flag-icons@7.2.3/css/flag-icons.min.css">
<h2 class="sr-only">Predictor cuántico de partidos: elegí dos selecciones y el modelo cuántico predice el ganador.</h2>
<div style="padding:1rem 0; font-family:var(--font-sans);">

  <div style="display:grid; grid-template-columns:1fr auto 1fr; gap:12px; align-items:end; margin-bottom:1.25rem;">
    <div>
      <label style="font-size:13px; color:var(--color-text-secondary);">Local</label>
      <select id="home" style="width:100%; margin-top:4px;"></select>
    </div>
    <div style="padding-bottom:8px; font-size:14px; color:var(--color-text-tertiary);">vs</div>
    <div>
      <label style="font-size:13px; color:var(--color-text-secondary);">Visitante</label>
      <select id="away" style="width:100%; margin-top:4px;"></select>
    </div>
  </div>

  <div style="display:grid; grid-template-columns:1fr auto 1fr; gap:12px; align-items:center; margin-bottom:1rem;">
    <button id="guessH" class="guess" data-side="H" style="display:flex; flex-direction:column; align-items:center; gap:6px; padding:12px;">
      <span id="flagH" class="fi" style="width:72px; height:54px; border-radius:4px; background-size:cover;"></span>
      <span id="nameH" style="font-size:14px; font-weight:500;"></span>
      <span id="eloH" style="font-size:12px; color:var(--color-text-tertiary);"></span>
    </button>
    <div style="font-size:18px; color:var(--color-text-tertiary);">VS</div>
    <button id="guessA" class="guess" data-side="A" style="display:flex; flex-direction:column; align-items:center; gap:6px; padding:12px;">
      <span id="flagA" class="fi" style="width:72px; height:54px; border-radius:4px; background-size:cover;"></span>
      <span id="nameA" style="font-size:14px; font-weight:500;"></span>
      <span id="eloA" style="font-size:12px; color:var(--color-text-tertiary);"></span>
    </button>
  </div>

  <p style="text-align:center; font-size:13px; color:var(--color-text-secondary); margin:0 0 12px;">
    Tocá la bandera de quien creés que gana, luego medí el qubit.
  </p>

  <div style="display:flex; gap:8px; justify-content:center; margin-bottom:1.25rem;">
    <button id="measure" style="font-weight:500;"><i class="ti ti-atom-2" aria-hidden="true"></i> Medir el qubit</button>
    <button id="random"><i class="ti ti-dice-5" aria-hidden="true"></i> Partido al azar</button>
  </div>

  <div id="result" style="visibility:hidden;">
    <div style="display:flex; justify-content:space-between; font-size:13px; color:var(--color-text-secondary); margin-bottom:4px;">
      <span id="lblH"></span><span id="lblA"></span>
    </div>
    <div style="position:relative; height:26px; border-radius:var(--border-radius-md); overflow:hidden; background:var(--color-background-secondary);">
      <div id="bar" style="height:100%; width:50%; background:var(--color-text-info); transition:width .6s ease;"></div>
      <div id="barpct" style="position:absolute; top:0; left:0; right:0; height:100%; display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:500;"></div>
    </div>
    <p id="verdict" style="text-align:center; font-size:15px; margin:12px 0 0;"></p>
  </div>

  <div style="margin-top:1.5rem;">
    <p style="font-size:13px; color:var(--color-text-secondary); margin:0 0 6px;">Frontera de decisión del clasificador cuántico (eje X: ventaja de Elo · eje Y: ventaja de forma). El punto es el partido actual.</p>
    <canvas id="map" width="640" height="260" style="width:100%; border-radius:var(--border-radius-md); border:0.5px solid var(--color-border-tertiary);" role="img" aria-label="Mapa de la frontera de decisión cuántica con el partido actual marcado"></canvas>
  </div>

  <div id="score" style="text-align:center; font-size:13px; color:var(--color-text-secondary); margin-top:1rem;"></div>

  <p style="text-align:center; font-size:12px; color:var(--color-text-tertiary); margin-top:1rem;">
    Clasificador cuántico variacional (PennyLane) entrenado sobre __N__ partidos · precisión en test __ACC__%
  </p>
</div>

<script>
const DATA = __DATA__;
const teams = DATA.teams;
const $ = id => document.getElementById(id);
let guess = null, score = {rounds:0, hits:0};

function fill(sel, def){
  teams.forEach((t,i)=>{ const o=document.createElement('option'); o.value=i; o.textContent=t.n; sel.appendChild(o); });
  sel.value = def;
}
function bilinear(ex, fy){
  const gx=DATA.gx, gy=DATA.gy, z=DATA.z;
  const cx=Math.max(gx[0], Math.min(gx[gx.length-1], ex));
  const cy=Math.max(gy[0], Math.min(gy[gy.length-1], fy));
  let i=0; while(i<gx.length-2 && gx[i+1]<cx) i++;
  let j=0; while(j<gy.length-2 && gy[j+1]<cy) j++;
  const tx=(cx-gx[i])/(gx[i+1]-gx[i]), ty=(cy-gy[j])/(gy[j+1]-gy[j]);
  const a=z[j][i], b=z[j][i+1], c=z[j+1][i], d=z[j+1][i+1];
  return a*(1-tx)*(1-ty)+b*tx*(1-ty)+c*(1-tx)*ty+d*tx*ty;
}
function setFlag(span, iso){ span.className='fi fi-'+iso; }

function refresh(){
  const h=teams[$('home').value], a=teams[$('away').value];
  setFlag($('flagH'),h.iso); setFlag($('flagA'),a.iso);
  $('nameH').textContent=h.n; $('nameA').textContent=a.n;
  $('eloH').textContent='Elo '+Math.round(h.elo); $('eloA').textContent='Elo '+Math.round(a.elo);
  $('lblH').textContent=h.n; $('lblA').textContent=a.n;
  draw(h,a,null);
}

function predict(h,a){
  const raw=bilinear(h.elo-a.elo, h.form-a.form);
  let p=0.5+0.5*raw/DATA.zmax; p=Math.max(0.02,Math.min(0.98,p));
  return p; // prob de que gane el local
}

function reveal(){
  const h=teams[$('home').value], a=teams[$('away').value];
  const p=predict(h,a), pct=Math.round(p*100);
  $('result').style.visibility='visible';
  $('bar').style.width=pct+'%';
  $('barpct').textContent=pct+'% '+h.n+'  ·  '+(100-pct)+'% '+a.n;
  const win = p>=0.5 ? h : a, conf=Math.round(Math.max(p,1-p)*100);
  $('verdict').innerHTML='Predicción cuántica: gana <b>'+win.n+'</b> ('+conf+'% de confianza)';
  if(guess){
    score.rounds++;
    const qSide = p>=0.5 ? 'H':'A';
    if(guess===qSide) score.hits++;
    $('score').innerHTML='<i class="ti ti-trophy" aria-hidden="true"></i> Coincidencias con el oráculo cuántico: '+score.hits+' / '+score.rounds;
    guess=null; document.querySelectorAll('.guess').forEach(b=>b.style.borderColor='');
  }
  draw(h,a,p);
}

function measure(){
  const btn=$('measure'); btn.disabled=true;
  let n=0; const iv=setInterval(()=>{
    const r=Math.random()*100;
    $('result').style.visibility='visible';
    $('bar').style.width=r+'%'; $('barpct').textContent='midiendo función de onda…';
    $('verdict').textContent='⟨Z⟩ colapsando…';
    if(++n>8){ clearInterval(iv); btn.disabled=false; reveal(); }
  },90);
}

function draw(h,a,p){
  const cv=$('map'), ctx=cv.getContext('2d'), W=cv.width, H=cv.height;
  const gx=DATA.gx, gy=DATA.gy, z=DATA.z, nx=gx.length, ny=gy.length;
  const cw=W/nx, ch=H/ny;
  for(let j=0;j<ny;j++) for(let i=0;i<nx;i++){
    const v=z[j][i]/DATA.zmax; // -1..1
    const t=(v+1)/2;
    const r=Math.round(220-120*t), g=Math.round(70+90*t), b=Math.round(70+150*t);
    ctx.fillStyle='rgb('+r+','+g+','+b+')';
    ctx.fillRect(i*cw, H-(j+1)*ch, cw+1, ch+1);
  }
  const xr=gx[nx-1]-gx[0], yr=gy[ny-1]-gy[0];
  if(h&&a){
    const ex=h.elo-a.elo, fy=h.form-a.form;
    const px=(Math.max(gx[0],Math.min(gx[nx-1],ex))-gx[0])/xr*W;
    const py=H-(Math.max(gy[0],Math.min(gy[ny-1],fy))-gy[0])/yr*H;
    ctx.beginPath(); ctx.arc(px,py,9,0,7); ctx.fillStyle='#fff'; ctx.fill();
    ctx.lineWidth=3; ctx.strokeStyle='#111'; ctx.stroke();
  }
}

fill($('home'), teams.findIndex(t=>t.n==='Argentina'));
fill($('away'), teams.findIndex(t=>t.n==='Brasil'));
$('home').onchange=refresh; $('away').onchange=refresh;
$('measure').onclick=measure;
$('random').onclick=()=>{ $('home').value=Math.floor(Math.random()*teams.length); let a; do{a=Math.floor(Math.random()*teams.length);}while(a==$('home').value); $('away').value=a; refresh(); };
document.querySelectorAll('.guess').forEach(b=>b.onclick=()=>{ guess=b.dataset.side; document.querySelectorAll('.guess').forEach(x=>x.style.borderColor=''); b.style.borderColor='var(--color-text-info)'; });
refresh();
</script>"""

BODY = BODY.replace("__N__", f"{DATA['n']:,}").replace("__ACC__", str(round(DATA["acc"]*100, 1)))
fragment = BODY.replace("__DATA__", data_js)

with open("results_widget_fragment.html", "w", encoding="utf-8") as f:
    f.write(fragment)

standalone = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predictor Cuántico de Partidos — WC26</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.17.0/dist/tabler-icons.min.css">
<style>
:root{{--font-sans:system-ui,sans-serif;--color-text-primary:#1a1a1a;--color-text-secondary:#555;--color-text-tertiary:#999;--color-text-info:#185fa5;--color-background-primary:#fff;--color-background-secondary:#f1efe8;--color-border-tertiary:rgba(0,0,0,.15);--border-radius-md:8px;}}
body{{max-width:680px;margin:2rem auto;padding:0 1rem;font-family:var(--font-sans);color:var(--color-text-primary);}}
h1{{font-weight:500;font-size:22px;}} .sr-only{{position:absolute;left:-9999px;}}
select,button{{font:inherit;padding:8px 12px;border:0.5px solid rgba(0,0,0,.25);border-radius:8px;background:#fff;cursor:pointer;}}
button:hover{{background:#f1efe8;}}
</style></head><body>
<h1>🔮⚽ Predictor Cuántico de Partidos</h1>
{fragment}
</body></html>"""

with open("quantum_predictor.html", "w", encoding="utf-8") as f:
    f.write(standalone)

print("OK -> quantum_predictor.html y results_widget_fragment.html")
print(f"grilla {len(gx)}x{len(gy)} | equipos {len(teams)} | zmax {zmax:.3f}")
