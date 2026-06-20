# ============================================================
# Actualización diaria del Predictor Mundial 2026
# ============================================================
# Baja los resultados nuevos del histórico, incorpora wc26_manual.csv
# y reentrena los modelos. Pensado para Windows Task Scheduler.
#
# Programarlo (corre todos los días a las 09:00):
#   schtasks /create /tn "WC26 Update" /tr "powershell -ExecutionPolicy Bypass -File `"%CD%\scripts\daily_update.ps1`"" /sc daily /st 09:00
#
# Ver el log:  Get-Content data\update.log -Tail 30
# ============================================================

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py  = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "data\update.log"

"$(Get-Date -Format o)  ▶ iniciando actualización" | Out-File -Append -Encoding utf8 $log

# 1) Snapshot de predicciones+cuotas de los partidos pendientes, con el modelo
#    ACTUAL (antes de incorporar los resultados de hoy) -> tracking honesto.
& $py -c "from src.evaluation.tracking import snapshot_predictions; snapshot_predictions()" *>> $log

# 2) Pipeline: descarga histórico + manual + reentrena (sin noticias para que sea rápido)
& $py -m src.pipeline --no-news *>> $log
$code = $LASTEXITCODE

"$(Get-Date -Format o)  ✓ finalizado (exit $code)" | Out-File -Append -Encoding utf8 $log
exit $code
