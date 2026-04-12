from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os, io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.environ.get("DATA_FOLDER", BASE_DIR)

TUKEY_FILES = {
    'LapTime':     'tukey_laptime.csv',
    'AvgSpeed':    'tukey_avgspeed.csv',
    'AvgRPM':      'tukey_avgRPM.csv',
    'AvgThrottle': 'tukey_avgthrottle.csv',
    'MaxSpeed':    'tukey_maxspeed.csv',
    'Sector1':     'tukey_sector1.csv',
    'Sector2':     'tukey_sector2.csv',
}

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>F1 Analytics · Silverstone 2024</title>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:ital,wght@0,400;0,600;0,700;0,800;1,700&family=Barlow:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  :root{--red:#e10600;--dark:#080808;--card:#131313;--border:#252525;--text:#f0f0f0;--muted:#555;--accent:#ff4444;--green:#00d2a0;--blue:#4da6ff;--gold:#ffd700;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--dark);color:var(--text);font-family:'Barlow',sans-serif;min-height:100vh;}

  /* ── HEADER ── */
  header{background:#0a0a0a;border-bottom:3px solid var(--red);padding:0 36px;display:flex;align-items:center;justify-content:space-between;height:58px;position:relative;overflow:hidden;}
  header::before{content:'';position:absolute;left:0;top:0;width:100%;height:100%;background:linear-gradient(90deg,rgba(225,6,0,.07) 0%,transparent 60%);pointer-events:none;}
  .logo{font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:28px;letter-spacing:4px;color:var(--red);z-index:1;}
  .logo span{color:#fff;}
  .header-right{display:flex;align-items:center;gap:22px;z-index:1;}
  .race-meta{text-align:right;}
  .race-meta .r1{font-family:'Barlow Condensed',sans-serif;font-size:15px;font-weight:700;letter-spacing:1px;color:#e0e0e0;}
  .race-meta .r2{font-size:11px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;margin-top:1px;}
  .flag{font-size:22px;}

  /* ── HERO BANNER ── */
  .hero{background:#0c0c0c;border-bottom:1px solid var(--border);padding:0;overflow:hidden;position:relative;}
  .hero-track{width:100%;height:130px;position:relative;background:linear-gradient(180deg,#0c0c0c 0%,#0f0f0f 100%);}

  /* dashed track line */
  .track-line{position:absolute;bottom:42px;left:0;width:100%;height:2px;background:repeating-linear-gradient(90deg,#333 0,#333 18px,transparent 18px,transparent 34px);}

  /* animated F1 car (pure CSS/SVG) */
  .car-wrap{position:absolute;bottom:30px;animation:drive 7s linear infinite;}
  @keyframes drive{0%{left:-220px}100%{left:calc(100% + 30px)}}

  /* speed blur streaks behind car */
  .streak{position:absolute;bottom:36px;height:3px;border-radius:2px;animation:streak 7s linear infinite;}
  .streak:nth-child(1){width:120px;background:linear-gradient(90deg,transparent,rgba(225,6,0,.5));animation-delay:-.1s;bottom:40px;}
  .streak:nth-child(2){width:80px;background:linear-gradient(90deg,transparent,rgba(255,100,0,.3));animation-delay:-.05s;bottom:35px;}
  .streak:nth-child(3){width:60px;background:linear-gradient(90deg,transparent,rgba(255,200,0,.2));bottom:43px;}
  @keyframes streak{0%{left:-230px}100%{left:calc(100% + 20px)}}

  /* podium cards inside hero */
  .podium-row{display:flex;gap:0;justify-content:center;align-items:stretch;padding:18px 36px 16px;}
  .podium-card{flex:1;max-width:340px;background:var(--card);border:1px solid var(--border);padding:14px 20px;position:relative;transition:transform .2s;}
  .podium-card:hover{transform:translateY(-2px);}
  .podium-card:first-child{border-radius:8px 0 0 8px;border-right:none;border-top:3px solid var(--gold);}
  .podium-card:nth-child(2){border-left:1px solid #333;border-right:1px solid #333;border-top:3px solid #bbb;}
  .podium-card:last-child{border-radius:0 8px 8px 0;border-left:none;border-top:3px solid #cd7f32;}
  .p-icon{font-size:20px;margin-bottom:4px;}
  .p-pos{font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:2.5px;text-transform:uppercase;color:var(--muted);margin-bottom:3px;}
  .p-name{font-family:'Barlow Condensed',sans-serif;font-size:22px;font-weight:800;letter-spacing:.5px;color:#fff;line-height:1.1;}
  .p-team{font-size:11px;color:var(--muted);margin-top:3px;letter-spacing:.5px;}
  .p-detail{font-size:11px;color:var(--green);margin-top:5px;font-weight:500;}
  .p-pos-num{position:absolute;top:12px;right:16px;font-family:'Barlow Condensed',sans-serif;font-size:38px;font-weight:800;color:rgba(255,255,255,.04);line-height:1;}

  /* ── NAV ── */
  nav{display:flex;border-bottom:2px solid var(--border);padding:0 36px;background:#0a0a0a;}
  .tab{padding:13px 28px;font-family:'Barlow Condensed',sans-serif;font-size:14px;font-weight:700;letter-spacing:2px;text-transform:uppercase;cursor:pointer;color:var(--muted);border-bottom:3px solid transparent;margin-bottom:-2px;transition:all .2s;}
  .tab.active{color:var(--red);border-bottom-color:var(--red);}
  .tab:hover:not(.active){color:#ccc;}

  /* ── MAIN ── */
  main{padding:28px 36px;max-width:1300px;margin:0 auto;}
  .panel{display:none;animation:fadeIn .2s ease;} .panel.active{display:block;}
  @keyframes fadeIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}

  .controls{display:flex;align-items:center;gap:14px;margin-bottom:26px;flex-wrap:wrap;background:var(--card);border:1px solid var(--border);border-radius:6px;padding:14px 18px;}
  .ctrl-label{font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);}
  select{background:#1a1a1a;color:var(--text);border:1px solid var(--border);border-radius:4px;padding:7px 14px;font-family:'Barlow',sans-serif;font-size:13px;cursor:pointer;outline:none;transition:border-color .2s;min-width:160px;}
  select:focus,select:hover{border-color:var(--red);}
  button{background:var(--red);color:#fff;border:none;border-radius:4px;padding:8px 20px;font-family:'Barlow Condensed',sans-serif;font-size:14px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;cursor:pointer;transition:background .2s;}
  button:hover{background:#c00;}

  .stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:22px;}
  .stat-card{background:var(--card);border:1px solid var(--border);border-top:2px solid var(--red);border-radius:6px;padding:16px 18px;}
  .stat-label{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:6px;}
  .stat-value{font-family:'Barlow Condensed',sans-serif;font-size:28px;font-weight:700;}
  .stat-value.red{color:var(--red);}
  .stat-value.green{color:var(--green);}
  .stat-value.blue{color:var(--blue);}

  .metrics-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px;}
  .metric-card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:16px;text-align:center;}
  .metric-val{font-family:'Barlow Condensed',sans-serif;font-size:32px;font-weight:800;color:var(--red);}
  .metric-lbl{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-top:4px;}

  .sec{font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:2.5px;text-transform:uppercase;color:var(--red);margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--border);}
  .card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:20px;}
  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}
  .one-col{display:grid;grid-template-columns:1fr;gap:16px;margin-bottom:16px;}
  .card img{width:100%;border-radius:4px;display:block;}

  .table-wrap{overflow-x:auto;}
  table{width:100%;border-collapse:collapse;font-size:13px;}
  thead tr{background:#1a1a1a;}
  th{padding:9px 13px;text-align:left;font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);white-space:nowrap;}
  td{padding:9px 13px;border-bottom:1px solid #1a1a1a;}
  tr:last-child td{border-bottom:none;}
  tr:hover td{background:#181818;}
  .badge{display:inline-block;padding:2px 8px;border-radius:3px;font-size:11px;font-weight:600;}
  .badge.yes{background:rgba(225,6,0,.18);color:var(--accent);}
  .badge.no{background:#1e1e1e;color:var(--muted);}

  .loader{text-align:center;padding:52px;color:var(--muted);font-size:11px;letter-spacing:3px;text-transform:uppercase;}
  .err{color:var(--accent);padding:20px;font-size:13px;background:rgba(225,6,0,.07);border-radius:6px;border:1px solid rgba(225,6,0,.2);}

  @media(max-width:900px){.podium-row{flex-direction:column;align-items:stretch;}.podium-card{max-width:100%;border-radius:6px!important;border-right:1px solid var(--border)!important;border-left:1px solid var(--border)!important;margin-bottom:8px;}}
  @media(max-width:720px){.stats-row,.metrics-row{grid-template-columns:1fr 1fr;}.two-col{grid-template-columns:1fr;}header,nav,main,.podium-row{padding-left:14px;padding-right:14px;}}
</style>
</head>
<body>

<!-- ── HEADER ── -->
<header>
  <div class="logo">F1<span>·</span>ANALYTICS</div>
  <div class="header-right">
    <div class="race-meta">
      <div class="r1">British Grand Prix 2024</div>
      <div class="r2">Silverstone · 7 July · 52 Laps</div>
    </div>
    <div class="flag">🇬🇧</div>
  </div>
</header>

<!-- ── HERO BANNER ── -->
<div class="hero">
  <!-- animated car track -->
  <div class="hero-track">
    <div class="track-line"></div>
    <!-- speed streaks -->
    <div class="streak"></div>
    <div class="streak"></div>
    <div class="streak"></div>
    <!-- SVG F1 car -->
    <div class="car-wrap">
      <svg width="200" height="52" viewBox="0 0 200 52" fill="none" xmlns="http://www.w3.org/2000/svg">
        <!-- rear wing -->
        <rect x="2" y="10" width="18" height="4" rx="1" fill="#cc0000"/>
        <rect x="4" y="8" width="14" height="3" rx="1" fill="#e10600"/>
        <!-- body -->
        <path d="M18 22 Q30 14 70 13 L130 13 Q160 13 175 20 L182 26 Q175 32 130 33 L70 33 Q30 33 18 28 Z" fill="#e10600"/>
        <!-- cockpit surround / halo -->
        <path d="M80 13 Q100 6 120 13" stroke="#111" stroke-width="3" fill="none" stroke-linecap="round"/>
        <rect x="88" y="7" width="24" height="7" rx="3" fill="#222"/>
        <!-- nose -->
        <path d="M175 20 L195 24 L182 26 Z" fill="#cc0000"/>
        <!-- sidepods -->
        <rect x="60" y="28" width="60" height="7" rx="2" fill="#c00000"/>
        <!-- front wing -->
        <rect x="178" y="26" width="20" height="3" rx="1" fill="#cc0000"/>
        <rect x="176" y="22" width="22" height="2" rx="1" fill="#aaa"/>
        <!-- rear wheel -->
        <circle cx="32" cy="36" r="10" fill="#111" stroke="#444" stroke-width="2"/>
        <circle cx="32" cy="36" r="5" fill="#333"/>
        <circle cx="32" cy="36" r="2" fill="#555"/>
        <!-- front wheel -->
        <circle cx="162" cy="36" r="10" fill="#111" stroke="#444" stroke-width="2"/>
        <circle cx="162" cy="36" r="5" fill="#333"/>
        <circle cx="162" cy="36" r="2" fill="#555"/>
        <!-- silver accents -->
        <rect x="90" y="20" width="30" height="2" rx="1" fill="rgba(255,255,255,0.15)"/>
        <!-- exhaust glow -->
        <ellipse cx="12" cy="24" rx="6" ry="3" fill="rgba(255,120,0,0.4)"/>
        <ellipse cx="8" cy="24" rx="4" ry="2" fill="rgba(255,200,0,0.3)"/>
      </svg>
    </div>
  </div>

  <!-- podium results -->
  <div class="podium-row">
    <div class="podium-card">
      <div class="p-pos-num">1</div>
      <div class="p-icon">🏆</div>
      <div class="p-pos">Race Winner</div>
      <div class="p-name">Lewis Hamilton</div>
      <div class="p-team">Mercedes · #44</div>
      <div class="p-detail">104th career victory · Record 9th British GP win</div>
    </div>
    <div class="podium-card">
      <div class="p-pos-num">2</div>
      <div class="p-icon">🥈</div>
      <div class="p-pos">2nd Place</div>
      <div class="p-name">Max Verstappen</div>
      <div class="p-team">Red Bull Racing · #1</div>
      <div class="p-detail" style="color:#bbb">Championship leader at time of race</div>
    </div>
    <div class="podium-card">
      <div class="p-pos-num">⚡</div>
      <div class="p-icon">⚡</div>
      <div class="p-pos">Fastest Lap</div>
      <div class="p-name">Carlos Sainz</div>
      <div class="p-team">Ferrari · #55</div>
      <div class="p-detail">1:28.293 · Lap 52 · 240.2 km/h</div>
    </div>
  </div>
</div>

<!-- ── NAV TABS ── -->
<nav>
  <div class="tab active" onclick="switchTab('tukey',this)">Tukey / ANOVA</div>
  <div class="tab" onclick="switchTab('regression',this)">Lap Time Prediction</div>
</nav>

<main>

  <!-- TUKEY -->
  <div class="panel active" id="panel-tukey">
    <div class="controls">
      <span class="ctrl-label">Parameter</span>
      <select id="param-select" onchange="loadTukey()">
        <option>LapTime</option><option>AvgSpeed</option><option>AvgRPM</option>
        <option>AvgThrottle</option><option>MaxSpeed</option>
        <option>Sector1</option><option>Sector2</option>
      </select>
    </div>
    <div id="tukey-content"><div class="loader">Loading…</div></div>
  </div>

  <!-- REGRESSION -->
  <div class="panel" id="panel-regression">
    <div class="controls">
      <span class="ctrl-label">Filter by Driver</span>
      <select id="driver-select"><option value="">All Drivers</option></select>
      <button onclick="loadRegression()">&#9654;&nbsp; Predict</button>
    </div>
    <div id="reg-content"><div class="loader">Choose a driver and click Predict…</div></div>
  </div>

</main>

<script>
function switchTab(name,el){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('panel-'+name).classList.add('active');
  if(name==='regression'&&!window._driversLoaded) loadDriverList();
}

async function loadDriverList(){
  window._driversLoaded=true;
  try{
    const d=await(await fetch('/api/drivers')).json();
    const sel=document.getElementById('driver-select');
    d.drivers.forEach(dr=>{const o=document.createElement('option');o.value=o.textContent=dr;sel.appendChild(o);});
  }catch(e){}
}

async function loadTukey(){
  const param=document.getElementById('param-select').value;
  document.getElementById('tukey-content').innerHTML='<div class="loader">Loading…</div>';
  const d=await(await fetch('/api/tukey?param='+param)).json();
  if(d.error){document.getElementById('tukey-content').innerHTML='<div class="err">'+d.error+'</div>';return;}
  const rows=d.top10.map(r=>`<tr>
    <td>${r.group1}</td><td>${r.group2}</td>
    <td>${(+r.meandiff).toFixed(4)}</td>
    <td>${(+r['p-adj']).toFixed(4)}</td>
    <td><span class="badge ${r.reject?'yes':'no'}">${r.reject?'YES':'NO'}</span></td>
  </tr>`).join('');
  document.getElementById('tukey-content').innerHTML=`
    <div class="stats-row">
      <div class="stat-card"><div class="stat-label">Total Comparisons</div><div class="stat-value">${d.total}</div></div>
      <div class="stat-card"><div class="stat-label">Significant</div><div class="stat-value red">${d.significant}</div></div>
      <div class="stat-card"><div class="stat-label">Sig. Rate</div><div class="stat-value blue">${d.sig_pct}<span style="font-size:14px;color:var(--muted)">%</span></div></div>
      <div class="stat-card"><div class="stat-label">Avg |Δ Mean|</div><div class="stat-value green">${d.avg_diff}</div></div>
    </div>
    <div class="card">
      <div class="sec">Top 10 Largest Pairwise Differences</div>
      <div class="table-wrap">
        <table><thead><tr><th>Group 1</th><th>Group 2</th><th>Mean Diff</th><th>p-adj</th><th>Reject H₀</th></tr></thead>
        <tbody>${rows}</tbody></table>
      </div>
    </div>`;
}

async function loadRegression(){
  const driver=document.getElementById('driver-select').value;
  const el=document.getElementById('reg-content');
  el.innerHTML='<div class="loader">Running prediction model…</div>';
  const d=await(await fetch('/api/regression?driver='+encodeURIComponent(driver))).json();
  if(d.error){el.innerHTML='<div class="err">'+d.error+'</div>';return;}
  const predRows=d.preview.map(r=>`<tr>
    <td><strong>${r.driver}</strong></td>
    <td>${r.actual}</td><td>${r.predicted}</td>
    <td style="color:${Math.abs(r.error)>1?'var(--accent)':'var(--green)'}">${r.error}</td>
  </tr>`).join('');
  const histHtml = (driver==='') ? `<div class="card"><div class="sec">Error Distribution</div><img src="data:image/png;base64,${d.hist}"/></div>` : '';
  const wrapCls  = (driver==='') ? 'two-col' : 'one-col';
  el.innerHTML=`
    <div class="metrics-row">
      <div class="metric-card"><div class="metric-val">${d.rmse}</div><div class="metric-lbl">RMSE</div></div>
      <div class="metric-card"><div class="metric-val">${d.mse}</div><div class="metric-lbl">MSE</div></div>
      <div class="metric-card"><div class="metric-val">${d.mae}</div><div class="metric-lbl">MAE</div></div>
      <div class="metric-card"><div class="metric-val">${d.mean_error}</div><div class="metric-lbl">Mean Error</div></div>
    </div>
    <div class="${wrapCls}">
      ${histHtml}
      <div class="card"><div class="sec">Predicted vs Actual Lap Times</div><img src="data:image/png;base64,${d.lineplot}"/></div>
    </div>
    <div class="card">
      <div class="sec">Sample Predictions — ${driver||'All Drivers'} (first 15 rows)</div>
      <div class="table-wrap">
        <table><thead><tr><th>Driver</th><th>Actual (s)</th><th>Predicted (s)</th><th>Error</th></tr></thead>
        <tbody>${predRows}</tbody></table>
      </div>
    </div>`;
}

loadTukey();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML


def get_driver_cols(df):
    """Return list of driver codes extracted from Driver_XXX dummy columns."""
    return [c.replace("Driver_", "") for c in df.columns if c.startswith("Driver_")]

@app.route("/api/drivers")
def api_drivers():
    try:
        df = pd.read_csv(os.path.join(DATA_FOLDER, "f1_deployment_data.csv"))
        drivers = sorted(get_driver_cols(df))
        return jsonify({"drivers": drivers})
    except Exception as e:
        return jsonify({"drivers": [], "error": str(e)})


@app.route("/api/tukey")
def api_tukey():
    param = request.args.get("param", "LapTime")
    if param not in TUKEY_FILES:
        return jsonify({"error": f"Unknown parameter: {param}"})
    try:
        df = pd.read_csv(os.path.join(DATA_FOLDER, TUKEY_FILES[param]))
    except Exception as e:
        return jsonify({"error": str(e)})
    if 'reject' not in df.columns:
        return jsonify({"error": "No 'reject' column found in this CSV."})
    num_sig = int(df['reject'].sum())
    total = len(df)
    sig_pct = round(100 * num_sig / total, 2) if total else 0
    avg_diff = round(df['meandiff'].abs().mean(), 4)
    top10 = (df.sort_values('meandiff', key=abs, ascending=False)
               .head(10)[['group1', 'group2', 'meandiff', 'p-adj', 'reject']]
               .to_dict(orient='records'))
    return jsonify({"total": total, "significant": num_sig,
                    "sig_pct": sig_pct, "avg_diff": avg_diff, "top10": top10})


@app.route("/api/regression")
def api_regression():
    driver_filter = request.args.get("driver", "").strip()
    try:
        df = pd.read_csv(os.path.join(DATA_FOLDER, "f1_deployment_data.csv"))

        driver_cols = [c for c in df.columns if c.startswith("Driver_")]

        # Filter rows where the selected driver's dummy column == True/1
        if driver_filter:
            col = f"Driver_{driver_filter}"
            if col not in df.columns:
                return jsonify({"error": f"Column '{col}' not found in CSV."})
            plot_df = df[df[col].astype(str).str.upper().isin(['TRUE', '1', 'YES'])].copy()
            if plot_df.empty:
                return jsonify({"error": f"No rows found where {col} is True."})
        else:
            plot_df = df.copy()

        # Derive a human-readable driver label per row from dummy columns
        def row_driver(r):
            for c in driver_cols:
                if str(r[c]).upper() in ('TRUE', '1', 'YES'):
                    return c.replace("Driver_", "")
            return "UNK"

        plot_df = plot_df.copy()
        plot_df['_driver_label'] = plot_df.apply(row_driver, axis=1)

        actual    = plot_df['LapTime'].values
        predicted = plot_df['Predicted_LapTime'].values
        error     = predicted - actual

        rmse     = round(float(np.sqrt(mean_squared_error(actual, predicted))), 4)
        mse      = round(float(mean_squared_error(actual, predicted)), 4)
        mae      = round(float(mean_absolute_error(actual, predicted)), 4)
        mean_err = round(float(np.mean(error)), 6)

        # Histogram
        fig, ax = plt.subplots(figsize=(5.5, 3.2), facecolor='#111')
        ax.set_facecolor('#1a1a1a')
        ax.hist(error, bins=14, color='#e10600', edgecolor='#ff6666', linewidth=0.7)
        ax.tick_params(colors='#aaa', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#333')
        ax.set_xlabel("Prediction Error (s)", color='#999', fontsize=8)
        ax.set_ylabel("Frequency", color='#999', fontsize=8)
        ax.set_title("Error Distribution", color='#ccc', fontsize=9, pad=8)
        buf = io.BytesIO(); plt.tight_layout()
        plt.savefig(buf, format='png', dpi=120, facecolor='#111'); plt.close()
        hist_b64 = base64.b64encode(buf.getvalue()).decode()

        # Line plot
        n = min(50, len(plot_df))
        idx = np.arange(n)
        fig2, ax2 = plt.subplots(figsize=(6.5, 3.2), facecolor='#111')
        ax2.set_facecolor('#1a1a1a')
        ax2.plot(idx, actual[:n],    color='#4da6ff', marker='o', markersize=3.5,
                 linewidth=1.4, label='Actual')
        ax2.plot(idx, predicted[:n], color='#e10600', marker='x', markersize=3.5,
                 linewidth=1.4, label='Predicted', linestyle='--')
        ax2.tick_params(colors='#aaa', labelsize=8)
        for sp in ax2.spines.values(): sp.set_edgecolor('#333')
        ax2.set_xlabel("Lap Index", color='#999', fontsize=8)
        ax2.set_ylabel("Lap Time (s)", color='#999', fontsize=8)
        title = f"Predicted vs Actual · {driver_filter}" if driver_filter else "Predicted vs Actual · All Drivers"
        ax2.set_title(title, color='#ccc', fontsize=9, pad=8)
        ax2.legend(fontsize=8, facecolor='#1e1e1e', edgecolor='#444', labelcolor='#ccc')
        ax2.grid(True, linestyle='--', alpha=0.2, color='#555')
        buf2 = io.BytesIO(); plt.tight_layout()
        plt.savefig(buf2, format='png', dpi=120, facecolor='#111'); plt.close()
        line_b64 = base64.b64encode(buf2.getvalue()).decode()

        # Preview table
        preview = []
        for _, row in plot_df.head(15).iterrows():
            preview.append({
                "driver":    row['_driver_label'],
                "actual":    round(row['LapTime'], 3),
                "predicted": round(row['Predicted_LapTime'], 3),
                "error":     round(row['Predicted_LapTime'] - row['LapTime'], 4)
            })

        return jsonify({
            "rmse": rmse, "mse": mse, "mae": mae, "mean_error": mean_err,
            "hist": hist_b64, "lineplot": line_b64, "preview": preview
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
