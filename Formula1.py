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

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>F1 Analytics · Silverstone 2024</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Barlow+Condensed:ital,wght@0,400;0,600;0,700;0,800;1,700&family=Barlow:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  :root{
    --red:#e10600; --dark:#06060a; --card:#0f0f18; --border:#1e1e2e;
    --text:#eef0ff; --muted:#4a4a6a; --accent:#ff4444;
    --cyan:#00e5ff; --purple:#9b59ff; --gold:#ffd700;
    --soft:#ffdd44; --medium:#ffffff; --hard:#444;
    --green:#00ffaa;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  html{scroll-behavior:smooth;}
  body{background:var(--dark);color:var(--text);font-family:'Barlow',sans-serif;min-height:100vh;
    background-image:radial-gradient(ellipse at 20% 0%,rgba(155,89,255,.06) 0%,transparent 60%),
                     radial-gradient(ellipse at 80% 100%,rgba(225,6,0,.05) 0%,transparent 60%);}

  /* ── SCANLINE OVERLAY ── */
  body::after{content:'';position:fixed;inset:0;pointer-events:none;z-index:999;
    background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.03) 2px,rgba(0,0,0,.03) 4px);}

  /* ── HEADER ── */
  header{background:rgba(6,6,10,.95);backdrop-filter:blur(12px);border-bottom:1px solid var(--border);
    padding:0 40px;display:flex;align-items:center;justify-content:space-between;height:62px;
    position:sticky;top:0;z-index:100;box-shadow:0 2px 40px rgba(225,6,0,.1);}
  header::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,var(--red) 30%,var(--purple) 70%,transparent);}
  .logo{font-family:'Orbitron',monospace;font-weight:900;font-size:20px;letter-spacing:4px;
    color:var(--red);text-shadow:0 0 20px rgba(225,6,0,.5);z-index:1;}
  .logo em{color:var(--text);font-style:normal;}
  .header-center{position:absolute;left:50%;transform:translateX(-50%);
    font-family:'Barlow Condensed',sans-serif;font-size:12px;letter-spacing:3px;
    text-transform:uppercase;color:var(--muted);}
  .header-right{display:flex;align-items:center;gap:20px;z-index:1;}
  .race-meta .r1{font-family:'Barlow Condensed',sans-serif;font-size:14px;font-weight:700;
    letter-spacing:1px;color:var(--text);text-align:right;}
  .race-meta .r2{font-size:10px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;
    text-align:right;margin-top:2px;}
  .flag{font-size:24px;filter:drop-shadow(0 0 6px rgba(255,255,255,.2));}

  /* ── HERO ── */
  .hero{background:linear-gradient(180deg,#09090f 0%,#0c0c16 100%);
    border-bottom:1px solid var(--border);overflow:hidden;position:relative;}
  .hero::before{content:'';position:absolute;inset:0;
    background:radial-gradient(ellipse at 50% 0%,rgba(225,6,0,.08),transparent 70%);pointer-events:none;}

  /* circuit dots pattern */
  .hero::after{content:'';position:absolute;inset:0;pointer-events:none;
    background-image:radial-gradient(circle,rgba(255,255,255,.04) 1px,transparent 1px);
    background-size:28px 28px;}

  /* track strip */
  .hero-track{width:100%;height:110px;position:relative;}
  .track-line{position:absolute;bottom:36px;left:0;width:100%;height:1px;
    background:repeating-linear-gradient(90deg,rgba(255,255,255,.12) 0,rgba(255,255,255,.12) 16px,transparent 16px,transparent 32px);}
  .track-shadow{position:absolute;bottom:28px;left:0;width:100%;height:16px;
    background:linear-gradient(180deg,transparent,rgba(0,229,255,.03));}

  /* car */
  .car-wrap{position:absolute;bottom:24px;animation:drive 6s linear infinite;}
  @keyframes drive{0%{left:-220px;opacity:0}5%{opacity:1}95%{opacity:1}100%{left:calc(100% + 40px);opacity:0}}
  .streak{position:absolute;border-radius:2px;animation:streak 6s linear infinite;}
  .streak:nth-child(1){width:140px;height:2px;bottom:36px;
    background:linear-gradient(90deg,transparent,rgba(0,229,255,.4));animation-delay:0s;}
  .streak:nth-child(2){width:90px;height:2px;bottom:31px;
    background:linear-gradient(90deg,transparent,rgba(225,6,0,.35));animation-delay:.04s;}
  .streak:nth-child(3){width:55px;height:1px;bottom:41px;
    background:linear-gradient(90deg,transparent,rgba(155,89,255,.25));animation-delay:.02s;}
  @keyframes streak{0%{left:-250px}100%{left:calc(100% + 30px)}}

  /* podium */
  .podium-row{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;
    background:var(--border);border-top:1px solid var(--border);}
  .podium-card{background:var(--card);padding:18px 22px;position:relative;
    transition:background .25s;overflow:hidden;}
  .podium-card::before{content:'';position:absolute;inset:0;opacity:0;transition:opacity .3s;
    background:radial-gradient(ellipse at 50% 0%,rgba(255,255,255,.03),transparent 70%);}
  .podium-card:hover::before{opacity:1;}
  .podium-card:hover{background:#121220;}
  .podium-card:nth-child(1){border-top:2px solid var(--gold);}
  .podium-card:nth-child(2){border-top:2px solid #8a8a9a;}
  .podium-card:nth-child(3){border-top:2px solid var(--cyan);}
  .p-bg-num{position:absolute;right:14px;top:8px;font-family:'Orbitron',monospace;
    font-size:52px;font-weight:900;color:rgba(255,255,255,.03);line-height:1;pointer-events:none;}
  .p-pill{display:inline-block;padding:2px 9px;border-radius:20px;font-size:10px;
    font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:7px;}
  .p-pill.win{background:rgba(255,215,0,.12);color:var(--gold);border:1px solid rgba(255,215,0,.2);}
  .p-pill.p2{background:rgba(200,200,210,.08);color:#aaa;border:1px solid rgba(200,200,210,.15);}
  .p-pill.fast{background:rgba(0,229,255,.1);color:var(--cyan);border:1px solid rgba(0,229,255,.2);}
  .p-name{font-family:'Barlow Condensed',sans-serif;font-size:24px;font-weight:800;
    letter-spacing:.5px;line-height:1.1;color:#fff;}
  .p-team{font-size:11px;color:var(--muted);margin-top:4px;letter-spacing:.5px;}
  .p-detail{font-size:11px;margin-top:7px;font-weight:500;font-family:'Barlow Condensed',sans-serif;
    letter-spacing:.5px;}

  /* ── NAV ── */
  nav{display:flex;border-bottom:1px solid var(--border);padding:0 40px;
    background:rgba(6,6,10,.9);backdrop-filter:blur(8px);position:sticky;top:62px;z-index:99;}
  .tab{padding:14px 30px;font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:700;
    letter-spacing:2.5px;text-transform:uppercase;cursor:pointer;color:var(--muted);
    border-bottom:2px solid transparent;margin-bottom:-1px;transition:all .2s;position:relative;}
  .tab.active{color:var(--cyan);border-bottom-color:var(--cyan);}
  .tab.active::after{content:'';position:absolute;bottom:-1px;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,var(--cyan),transparent);filter:blur(3px);}
  .tab:hover:not(.active){color:#ccc;}

  /* ── MAIN ── */
  main{padding:32px 40px;max-width:1340px;margin:0 auto;}
  .panel{display:none;animation:fadeIn .25s ease;} .panel.active{display:block;}
  @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}

  /* controls bar */
  .controls{display:flex;align-items:center;gap:16px;margin-bottom:28px;flex-wrap:wrap;
    background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px 20px;
    box-shadow:0 4px 24px rgba(0,0,0,.4);}
  .ctrl-group{display:flex;align-items:center;gap:10px;}
  .ctrl-divider{width:1px;height:28px;background:var(--border);margin:0 4px;}
  .ctrl-label{font-family:'Barlow Condensed',sans-serif;font-size:10px;letter-spacing:2.5px;
    text-transform:uppercase;color:var(--muted);}

  select{background:#0d0d1a;color:var(--text);border:1px solid var(--border);border-radius:5px;
    padding:7px 14px;font-family:'Barlow',sans-serif;font-size:13px;cursor:pointer;outline:none;
    transition:border-color .2s,box-shadow .2s;min-width:160px;
    appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%234a4a6a'/%3E%3C/svg%3E");
    background-repeat:no-repeat;background-position:right 12px center;padding-right:30px;}
  select:focus,select:hover{border-color:var(--cyan);box-shadow:0 0 0 2px rgba(0,229,255,.08);}

  /* tyre compound pills */
  .tyre-group{display:flex;gap:8px;}
  .tyre-btn{padding:6px 16px;border-radius:20px;font-family:'Barlow Condensed',sans-serif;
    font-size:12px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;cursor:pointer;
    border:1px solid transparent;transition:all .2s;}
  .tyre-btn.all{background:#1a1a2e;color:var(--muted);border-color:var(--border);}
  .tyre-btn.all.active,
  .tyre-btn.all:hover{background:#252535;color:#ccc;border-color:#444;}
  .tyre-btn.soft{background:rgba(255,221,68,.08);color:var(--soft);border-color:rgba(255,221,68,.2);}
  .tyre-btn.soft.active{background:rgba(255,221,68,.18);border-color:var(--soft);
    box-shadow:0 0 10px rgba(255,221,68,.2);}
  .tyre-btn.medium{background:rgba(255,255,255,.06);color:#e0e0e0;border-color:rgba(255,255,255,.15);}
  .tyre-btn.medium.active{background:rgba(255,255,255,.14);border-color:#fff;
    box-shadow:0 0 10px rgba(255,255,255,.15);}

  button.run-btn{background:linear-gradient(135deg,var(--red),#b00000);color:#fff;border:none;
    border-radius:5px;padding:8px 22px;font-family:'Orbitron',monospace;font-size:11px;font-weight:700;
    letter-spacing:2px;text-transform:uppercase;cursor:pointer;transition:all .2s;
    box-shadow:0 2px 12px rgba(225,6,0,.3);}
  button.run-btn:hover{background:linear-gradient(135deg,#ff1a1a,var(--red));
    box-shadow:0 4px 20px rgba(225,6,0,.5);transform:translateY(-1px);}

  /* stat cards */
  .stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:22px;}
  .stat-card{background:var(--card);border:1px solid var(--border);border-top:2px solid var(--red);
    border-radius:8px;padding:16px 18px;position:relative;overflow:hidden;transition:border-color .2s;}
  .stat-card::after{content:'';position:absolute;inset:0;
    background:linear-gradient(135deg,rgba(255,255,255,.02),transparent);pointer-events:none;}
  .stat-label{font-size:9px;letter-spacing:2.5px;text-transform:uppercase;color:var(--muted);margin-bottom:7px;
    font-family:'Barlow Condensed',sans-serif;}
  .stat-value{font-family:'Orbitron',monospace;font-size:24px;font-weight:700;}
  .stat-value.red{color:var(--red);text-shadow:0 0 16px rgba(225,6,0,.4);}
  .stat-value.green{color:var(--green);text-shadow:0 0 16px rgba(0,255,170,.3);}
  .stat-value.cyan{color:var(--cyan);text-shadow:0 0 16px rgba(0,229,255,.3);}
  .stat-value.purple{color:var(--purple);text-shadow:0 0 16px rgba(155,89,255,.3);}

  /* metric cards */
  .metrics-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px;}
  .metric-card{background:var(--card);border:1px solid var(--border);border-radius:8px;
    padding:18px 16px;text-align:center;position:relative;overflow:hidden;}
  .metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,var(--purple),transparent);}
  .metric-val{font-family:'Orbitron',monospace;font-size:26px;font-weight:700;color:var(--cyan);
    text-shadow:0 0 20px rgba(0,229,255,.35);}
  .metric-lbl{font-size:9px;letter-spacing:2.5px;text-transform:uppercase;color:var(--muted);
    margin-top:5px;font-family:'Barlow Condensed',sans-serif;}

  /* section title */
  .sec{font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:3px;
    text-transform:uppercase;color:var(--cyan);margin-bottom:12px;padding-bottom:7px;
    border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px;}
  .sec::before{content:'';display:inline-block;width:3px;height:12px;
    background:var(--cyan);border-radius:2px;flex-shrink:0;}

  /* cards */
  .card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:20px;
    box-shadow:0 4px 24px rgba(0,0,0,.3);}
  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}
  .one-col{display:grid;grid-template-columns:1fr;gap:16px;margin-bottom:16px;}
  .card img{width:100%;border-radius:6px;display:block;}

  /* table */
  .table-wrap{overflow-x:auto;}
  table{width:100%;border-collapse:collapse;font-size:13px;}
  thead tr{background:rgba(255,255,255,.03);border-bottom:1px solid var(--border);}
  th{padding:10px 14px;text-align:left;font-family:'Barlow Condensed',sans-serif;font-size:10px;
    letter-spacing:2px;text-transform:uppercase;color:var(--muted);white-space:nowrap;}
  td{padding:10px 14px;border-bottom:1px solid rgba(255,255,255,.03);}
  tr:last-child td{border-bottom:none;}
  tbody tr:hover td{background:rgba(0,229,255,.03);}
  .badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:10px;font-weight:700;letter-spacing:.5px;}
  .badge.yes{background:rgba(225,6,0,.15);color:#ff6666;border:1px solid rgba(225,6,0,.25);}
  .badge.no{background:rgba(255,255,255,.04);color:var(--muted);border:1px solid var(--border);}

  /* tyre indicators in table */
  .tyre-soft{color:var(--soft);font-weight:700;}
  .tyre-medium{color:#ddd;font-weight:700;}

  .loader{text-align:center;padding:60px;color:var(--muted);font-family:'Orbitron',monospace;
    font-size:10px;letter-spacing:3px;text-transform:uppercase;
    animation:pulse 1.5s ease-in-out infinite;}
  @keyframes pulse{0%,100%{opacity:.4}50%{opacity:1}}
  .err{color:#ff6666;padding:20px;font-size:13px;background:rgba(225,6,0,.06);
    border-radius:8px;border:1px solid rgba(225,6,0,.2);font-family:'Barlow Condensed',sans-serif;
    letter-spacing:.5px;}

  /* compound badge in table */
  .cbadge{display:inline-block;padding:1px 8px;border-radius:3px;font-size:10px;font-weight:700;}
  .cbadge.soft{background:rgba(255,221,68,.15);color:var(--soft);border:1px solid rgba(255,221,68,.25);}
  .cbadge.medium{background:rgba(255,255,255,.08);color:#ddd;border:1px solid rgba(255,255,255,.18);}

  @media(max-width:900px){
    .podium-row{grid-template-columns:1fr;}
    .podium-card:nth-child(n){border-top-width:2px;}
  }
  @media(max-width:720px){
    .stats-row,.metrics-row{grid-template-columns:1fr 1fr;}
    .two-col{grid-template-columns:1fr;}
    header,nav,main{padding-left:16px;padding-right:16px;}
    .tyre-group{flex-wrap:wrap;}
    .header-center{display:none;}
  }
</style>
</head>
<body>

<!-- ── HEADER ── -->
<header>
  <div class="logo">F1<em>·</em>ANALYTICS</div>
  <div class="header-center">Silverstone Circuit · Round 12</div>
  <div class="header-right">
    <div class="race-meta">
      <div class="r1">British Grand Prix 2024</div>
      <div class="r2">7 July &nbsp;·&nbsp; 52 Laps &nbsp;·&nbsp; Silverstone</div>
    </div>
    <div class="flag">🇬🇧</div>
  </div>
</header>

<!-- ── HERO ── -->
<div class="hero">
  <div class="hero-track">
    <div class="track-line"></div>
    <div class="track-shadow"></div>
    <div class="streak"></div><div class="streak"></div><div class="streak"></div>
    <div class="car-wrap">
      <svg width="210" height="54" viewBox="0 0 210 54" fill="none" xmlns="http://www.w3.org/2000/svg">
        <!-- exhaust flame -->
        <ellipse cx="11" cy="26" rx="7" ry="3.5" fill="rgba(255,120,0,.5)"/>
        <ellipse cx="7"  cy="26" rx="4" ry="2"   fill="rgba(255,220,0,.35)"/>
        <!-- rear wing -->
        <rect x="2"  y="11" width="20" height="4" rx="1.5" fill="#c00"/>
        <rect x="5"  y="8"  width="14" height="4" rx="1"   fill="#e10600"/>
        <!-- main body -->
        <path d="M20 23 Q32 14 72 13 L135 13 Q164 13 180 21 L188 27 Q180 33 135 34 L72 34 Q32 34 20 29 Z" fill="#e10600"/>
        <!-- body highlight -->
        <path d="M60 13 Q100 11 140 13" stroke="rgba(255,255,255,.12)" stroke-width="1.5" fill="none"/>
        <!-- halo -->
        <path d="M82 13 Q102 5 122 13" stroke="#111" stroke-width="3.5" fill="none" stroke-linecap="round"/>
        <rect x="90" y="6" width="22" height="8" rx="3.5" fill="#1a1a1a"/>
        <!-- cockpit tint -->
        <rect x="92" y="8" width="18" height="5" rx="2" fill="rgba(0,229,255,.15)"/>
        <!-- nose -->
        <path d="M180 21 L200 26 L188 27 Z" fill="#cc0000"/>
        <!-- sidepods -->
        <rect x="62" y="28" width="66" height="8" rx="2" fill="#b80000"/>
        <!-- front wing main -->
        <rect x="184" y="27" width="22" height="3" rx="1" fill="#cc0000"/>
        <!-- front wing flap -->
        <rect x="182" y="22" width="24" height="2" rx="1" fill="#888"/>
        <!-- cyan accent stripe -->
        <rect x="22" y="19" width="155" height="2" rx="1" fill="rgba(0,229,255,.25)"/>
        <!-- rear wheel -->
        <circle cx="34"  cy="38" r="11" fill="#111" stroke="#2a2a2a" stroke-width="2"/>
        <circle cx="34"  cy="38" r="6"  fill="#222"/>
        <circle cx="34"  cy="38" r="2.5" fill="#444"/>
        <!-- front wheel -->
        <circle cx="165" cy="38" r="11" fill="#111" stroke="#2a2a2a" stroke-width="2"/>
        <circle cx="165" cy="38" r="6"  fill="#222"/>
        <circle cx="165" cy="38" r="2.5" fill="#444"/>
        <!-- wheel glow -->
        <circle cx="34"  cy="38" r="12" fill="none" stroke="rgba(0,229,255,.08)" stroke-width="2"/>
        <circle cx="165" cy="38" r="12" fill="none" stroke="rgba(0,229,255,.08)" stroke-width="2"/>
      </svg>
    </div>
  </div>

  <!-- podium -->
  <div class="podium-row">
    <div class="podium-card">
      <div class="p-bg-num">1</div>
      <div class="p-pill win">🏆 Race Winner</div>
      <div class="p-name">Lewis Hamilton</div>
      <div class="p-team">Mercedes &nbsp;·&nbsp; #44</div>
      <div class="p-detail" style="color:var(--gold)">104th win &nbsp;·&nbsp; Record 9th British GP victory</div>
    </div>
    <div class="podium-card">
      <div class="p-bg-num">2</div>
      <div class="p-pill p2">🥈 2nd Place</div>
      <div class="p-name">Max Verstappen</div>
      <div class="p-team">Red Bull Racing &nbsp;·&nbsp; #1</div>
      <div class="p-detail" style="color:#aaa">Championship leader at time of race</div>
    </div>
    <div class="podium-card">
      <div class="p-bg-num">⚡</div>
      <div class="p-pill fast">⚡ Fastest Lap</div>
      <div class="p-name">Carlos Sainz</div>
      <div class="p-team">Ferrari &nbsp;·&nbsp; #55</div>
      <div class="p-detail" style="color:var(--cyan)">1:28.293 &nbsp;·&nbsp; Lap 52 &nbsp;·&nbsp; 240.2 km/h</div>
    </div>
  </div>
</div>

<!-- ── NAV ── -->
<nav>
  <div class="tab active" onclick="switchTab('tukey',this)">Tukey / ANOVA</div>
  <div class="tab" onclick="switchTab('regression',this)">Lap Time Prediction</div>
</nav>

<main>

  <!-- TUKEY PANEL -->
  <div class="panel active" id="panel-tukey">
    <div class="controls">
      <div class="ctrl-group">
        <span class="ctrl-label">Parameter</span>
        <select id="param-select" onchange="loadTukey()">
          <option>LapTime</option><option>AvgSpeed</option><option>AvgRPM</option>
          <option>AvgThrottle</option><option>MaxSpeed</option>
          <option>Sector1</option><option>Sector2</option>
        </select>
      </div>
    </div>
    <div id="tukey-content"><div class="loader">Initialising…</div></div>
  </div>

  <!-- REGRESSION PANEL -->
  <div class="panel" id="panel-regression">
    <div class="controls">
      <div class="ctrl-group">
        <span class="ctrl-label">Driver</span>
        <select id="driver-select"><option value="">All Drivers</option></select>
      </div>
      <div class="ctrl-divider"></div>
      <div class="ctrl-group">
        <span class="ctrl-label">Tyre Compound</span>
        <div class="tyre-group">
          <button class="tyre-btn all active"  onclick="setTyre('',this)">ALL</button>
          <button class="tyre-btn soft"         onclick="setTyre('SOFT',this)">◉ SOFT</button>
          <button class="tyre-btn medium"       onclick="setTyre('MEDIUM',this)">◉ MEDIUM</button>
        </div>
      </div>
      <div class="ctrl-divider"></div>
      <button class="run-btn" onclick="loadRegression()">▶ PREDICT</button>
    </div>
    <div id="reg-content"><div class="loader">Select filters and click PREDICT…</div></div>
  </div>

</main>

<script>
let activeCompound = '';

function setTyre(val, el){
  activeCompound = val;
  document.querySelectorAll('.tyre-btn').forEach(b=>b.classList.remove('active'));
  el.classList.add('active');
}

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
    <td style="font-family:monospace">${(+r.meandiff).toFixed(4)}</td>
    <td style="font-family:monospace">${(+r['p-adj']).toFixed(4)}</td>
    <td><span class="badge ${r.reject?'yes':'no'}">${r.reject?'YES':'NO'}</span></td>
  </tr>`).join('');
  document.getElementById('tukey-content').innerHTML=`
    <div class="stats-row">
      <div class="stat-card"><div class="stat-label">Total Comparisons</div><div class="stat-value purple">${d.total}</div></div>
      <div class="stat-card"><div class="stat-label">Significant</div><div class="stat-value red">${d.significant}</div></div>
      <div class="stat-card"><div class="stat-label">Sig. Rate</div><div class="stat-value cyan">${d.sig_pct}<span style="font-size:13px;color:var(--muted)">%</span></div></div>
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
  const driver   = document.getElementById('driver-select').value;
  const compound = activeCompound;
  const el       = document.getElementById('reg-content');
  el.innerHTML   = '<div class="loader">Running prediction model…</div>';
  const url      = '/api/regression?driver='+encodeURIComponent(driver)+'&compound='+encodeURIComponent(compound);
  const d        = await(await fetch(url)).json();
  if(d.error){el.innerHTML='<div class="err">'+d.error+'</div>';return;}

  const cLabel = compound==='SOFT'
    ? '<span class="cbadge soft">SOFT</span>'
    : compound==='MEDIUM'
      ? '<span class="cbadge medium">MEDIUM</span>'
      : '<span style="color:var(--muted)">ALL</span>';

  const predRows=d.preview.map(r=>`<tr>
    <td><strong>${r.driver}</strong></td>
    <td><span class="cbadge ${r.compound.toLowerCase()}">${r.compound}</span></td>
    <td style="font-family:monospace">${r.actual}</td>
    <td style="font-family:monospace">${r.predicted}</td>
    <td style="font-family:monospace;color:${Math.abs(r.error)>1?'var(--accent)':'var(--green)'}">${r.error}</td>
  </tr>`).join('');

  const isAll    = (driver==='' && compound==='');
  const histHtml = isAll ? `<div class="card"><div class="sec">Error Distribution</div><img src="data:image/png;base64,${d.hist}"/></div>` : '';
  const wrapCls  = isAll ? 'two-col' : 'one-col';

  const filterLabel = [driver||'All Drivers', compound ? cLabel : ''].filter(Boolean).join(' &nbsp;·&nbsp; ');

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
      <div class="sec">Sample Predictions — ${filterLabel} (first 15 rows)</div>
      <div class="table-wrap">
        <table><thead><tr><th>Driver</th><th>Compound</th><th>Actual (s)</th><th>Predicted (s)</th><th>Error</th></tr></thead>
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
    num_sig  = int(df['reject'].sum())
    total    = len(df)
    sig_pct  = round(100 * num_sig / total, 2) if total else 0
    avg_diff = round(df['meandiff'].abs().mean(), 4)
    top10    = (df.sort_values('meandiff', key=abs, ascending=False)
                  .head(10)[['group1', 'group2', 'meandiff', 'p-adj', 'reject']]
                  .to_dict(orient='records'))
    return jsonify({"total": total, "significant": num_sig,
                    "sig_pct": sig_pct, "avg_diff": avg_diff, "top10": top10})


@app.route("/api/regression")
def api_regression():
    driver_filter   = request.args.get("driver",   "").strip()
    compound_filter = request.args.get("compound", "").strip().upper()

    try:
        df = pd.read_csv(os.path.join(DATA_FOLDER, "f1_deployment_data.csv"))
        driver_cols = [c for c in df.columns if c.startswith("Driver_")]
        plot_df = df.copy()

        # ── filter by driver dummy column ──
        if driver_filter:
            col = f"Driver_{driver_filter}"
            if col not in df.columns:
                return jsonify({"error": f"Column '{col}' not found in CSV."})
            plot_df = plot_df[plot_df[col].astype(str).str.upper().isin(['TRUE', '1', 'YES'])]

        # ── filter by compound ──
        if compound_filter in ('SOFT', 'MEDIUM'):
            comp_col = f"Compound_{compound_filter}"
            if comp_col not in plot_df.columns:
                return jsonify({"error": f"Column '{comp_col}' not found in CSV."})
            plot_df = plot_df[plot_df[comp_col].astype(str).str.upper().isin(['TRUE', '1', 'YES'])]

        if plot_df.empty:
            return jsonify({"error": "No rows match the selected filters."})

        plot_df = plot_df.copy()

        # derive driver label per row
        def row_driver(r):
            for c in driver_cols:
                if str(r[c]).upper() in ('TRUE', '1', 'YES'):
                    return c.replace("Driver_", "")
            return "UNK"
        plot_df['_driver_label'] = plot_df.apply(row_driver, axis=1)

        # derive compound label per row
        def row_compound(r):
            for cmp in ['SOFT', 'MEDIUM', 'HARD']:
                col = f"Compound_{cmp}"
                if col in r and str(r[col]).upper() in ('TRUE', '1', 'YES'):
                    return cmp
            return "UNK"
        plot_df['_compound_label'] = plot_df.apply(row_compound, axis=1)

        actual    = plot_df['LapTime'].values
        predicted = plot_df['Predicted_LapTime'].values
        error     = predicted - actual

        rmse     = round(float(np.sqrt(mean_squared_error(actual, predicted))), 4)
        mse      = round(float(mean_squared_error(actual, predicted)), 4)
        mae      = round(float(mean_absolute_error(actual, predicted)), 4)
        mean_err = round(float(np.mean(error)), 6)

        # ── histogram (dark theme matching site) ──
        fig, ax = plt.subplots(figsize=(5.5, 3.2), facecolor='#06060a')
        ax.set_facecolor('#0f0f18')
        ax.hist(error, bins=14, color='#e10600', edgecolor='#ff5555', linewidth=0.6, alpha=0.9)
        ax.tick_params(colors='#4a4a6a', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#1e1e2e')
        ax.set_xlabel("Prediction Error (s)", color='#4a4a6a', fontsize=8)
        ax.set_ylabel("Frequency",            color='#4a4a6a', fontsize=8)
        ax.set_title("Error Distribution",    color='#7070a0', fontsize=9, pad=8)
        ax.grid(True, linestyle='--', alpha=0.12, color='#3a3a5a')
        buf = io.BytesIO(); plt.tight_layout()
        plt.savefig(buf, format='png', dpi=120, facecolor='#06060a'); plt.close()
        hist_b64 = base64.b64encode(buf.getvalue()).decode()

        # ── line plot ──
        n   = min(50, len(plot_df))
        idx = np.arange(n)
        fig2, ax2 = plt.subplots(figsize=(6.5, 3.2), facecolor='#06060a')
        ax2.set_facecolor('#0f0f18')
        ax2.plot(idx, actual[:n],    color='#00e5ff', marker='o', markersize=3.5,
                 linewidth=1.5, label='Actual')
        ax2.plot(idx, predicted[:n], color='#e10600', marker='x', markersize=3.5,
                 linewidth=1.5, label='Predicted', linestyle='--')
        ax2.tick_params(colors='#4a4a6a', labelsize=8)
        for sp in ax2.spines.values(): sp.set_edgecolor('#1e1e2e')
        ax2.set_xlabel("Lap Index",     color='#4a4a6a', fontsize=8)
        ax2.set_ylabel("Lap Time (s)",  color='#4a4a6a', fontsize=8)
        parts = []
        if driver_filter:   parts.append(driver_filter)
        if compound_filter: parts.append(compound_filter)
        title = "Predicted vs Actual · " + (" | ".join(parts) if parts else "All")
        ax2.set_title(title, color='#7070a0', fontsize=9, pad=8)
        ax2.legend(fontsize=8, facecolor='#0f0f18', edgecolor='#2a2a3a', labelcolor='#aaa')
        ax2.grid(True, linestyle='--', alpha=0.12, color='#3a3a5a')
        buf2 = io.BytesIO(); plt.tight_layout()
        plt.savefig(buf2, format='png', dpi=120, facecolor='#06060a'); plt.close()
        line_b64 = base64.b64encode(buf2.getvalue()).decode()

        # ── preview table ──
        preview = []
        for _, row in plot_df.head(15).iterrows():
            preview.append({
                "driver":    row['_driver_label'],
                "compound":  row['_compound_label'],
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
