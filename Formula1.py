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
    'Sector1':     'tukey_sector2.csv',
    'Sector2':     'tukey_sector3.csv',
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
    --text:#eef0ff; --muted:#4a4a6a; --accent:#ff5555;
    --steel:#a8c8e8; --purple:#9b59ff; --gold:#ffd700;
    --soft:#ffdd44; --medium:#e8e8e8; --green:#00ffaa;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  html{scroll-behavior:smooth;}

  body{
    background:var(--dark);color:var(--text);
    font-family:'Barlow',sans-serif;min-height:100vh;
    background-image:
      radial-gradient(ellipse at 15% 0%,rgba(155,89,255,.08) 0%,transparent 55%),
      radial-gradient(ellipse at 85% 100%,rgba(225,6,0,.07) 0%,transparent 55%),
      radial-gradient(ellipse at 50% 50%,rgba(168,200,232,.02) 0%,transparent 70%);
  }
  /* scanlines */
  body::after{content:'';position:fixed;inset:0;pointer-events:none;z-index:998;
    background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,.025) 3px,rgba(0,0,0,.025) 4px);}

  /* ── HEADER ── */
  header{
    background:rgba(6,6,10,.97);backdrop-filter:blur(16px);
    border-bottom:1px solid var(--border);
    padding:0 44px;display:flex;align-items:center;justify-content:space-between;height:64px;
    position:sticky;top:0;z-index:100;
    box-shadow:0 1px 0 var(--border),0 4px 40px rgba(225,6,0,.12);
  }
  header::after{
    content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent 0%,var(--red) 25%,rgba(168,200,232,.6) 50%,var(--purple) 75%,transparent 100%);
    animation:headerShimmer 4s ease-in-out infinite;
  }
  @keyframes headerShimmer{
    0%,100%{opacity:.7;background-position:0% 50%;}
    50%{opacity:1;background-position:100% 50%;}
  }
  .logo{
    font-family:'Orbitron',monospace;font-weight:900;font-size:21px;letter-spacing:5px;
    color:var(--red);z-index:1;
    text-shadow:0 0 30px rgba(225,6,0,.6),0 0 60px rgba(225,6,0,.2);
    animation:logoPulse 3s ease-in-out infinite;
  }
  @keyframes logoPulse{0%,100%{text-shadow:0 0 20px rgba(225,6,0,.5),0 0 40px rgba(225,6,0,.15);}
    50%{text-shadow:0 0 35px rgba(225,6,0,.8),0 0 70px rgba(225,6,0,.3);}}
  .logo em{color:var(--steel);font-style:normal;}
  .header-center{
    position:absolute;left:50%;transform:translateX(-50%);
    font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:4px;
    text-transform:uppercase;color:var(--muted);
    animation:fadeIn .8s ease forwards;
  }
  .header-right{display:flex;align-items:center;gap:22px;z-index:1;}
  .race-meta .r1{font-family:'Barlow Condensed',sans-serif;font-size:14px;font-weight:700;
    letter-spacing:1px;color:var(--text);text-align:right;}
  .race-meta .r2{font-size:10px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;
    text-align:right;margin-top:2px;}
  .flag{font-size:24px;filter:drop-shadow(0 0 8px rgba(255,255,255,.25));
    animation:flagWave .5s ease-in-out infinite alternate;}
  @keyframes flagWave{from{transform:rotate(-3deg);}to{transform:rotate(3deg);}}

  /* ── HERO ── */
  .hero{
    background:linear-gradient(180deg,#08080e 0%,#0b0b15 100%);
    border-bottom:1px solid var(--border);overflow:hidden;position:relative;
  }
  /* animated dot grid */
  .hero::before{
    content:'';position:absolute;inset:0;pointer-events:none;
    background-image:radial-gradient(circle,rgba(255,255,255,.045) 1px,transparent 1px);
    background-size:26px 26px;
    animation:gridDrift 20s linear infinite;
  }
  @keyframes gridDrift{0%{background-position:0 0;}100%{background-position:26px 26px;}}
  /* red glow */
  .hero::after{content:'';position:absolute;inset:0;pointer-events:none;
    background:radial-gradient(ellipse at 50% -10%,rgba(225,6,0,.1),transparent 65%);}

  /* track strip */
  .hero-track{width:100%;height:116px;position:relative;}
  .track-line{
    position:absolute;bottom:38px;left:0;width:100%;height:1px;
    background:repeating-linear-gradient(90deg,
      rgba(255,255,255,.14) 0,rgba(255,255,255,.14) 14px,
      transparent 14px,transparent 28px);
  }
  .track-glow{position:absolute;bottom:34px;left:0;width:100%;height:8px;
    background:linear-gradient(180deg,rgba(168,200,232,.06),transparent);}

  /* car animation */
  .car-wrap{position:absolute;bottom:26px;animation:drive 5.5s linear infinite;}
  @keyframes drive{
    0%  {left:-220px;opacity:0;}
    6%  {opacity:1;}
    94% {opacity:1;}
    100%{left:calc(100% + 40px);opacity:0;}
  }
  /* motion blur streaks */
  .streak{position:absolute;border-radius:2px;animation:streak 5.5s linear infinite;}
  .streak:nth-child(1){width:160px;height:2px;bottom:37px;
    background:linear-gradient(90deg,transparent,rgba(168,200,232,.5));animation-delay:0s;}
  .streak:nth-child(2){width:100px;height:2px;bottom:32px;
    background:linear-gradient(90deg,transparent,rgba(225,6,0,.45));animation-delay:.05s;}
  .streak:nth-child(3){width:65px;height:1px;bottom:43px;
    background:linear-gradient(90deg,transparent,rgba(155,89,255,.3));animation-delay:.025s;}
  .streak:nth-child(4){width:40px;height:1px;bottom:29px;
    background:linear-gradient(90deg,transparent,rgba(255,221,68,.2));animation-delay:.01s;}
  @keyframes streak{0%{left:-270px;}100%{left:calc(100% + 30px);}}

  /* podium */
  .podium-row{
    display:grid;grid-template-columns:1fr 1fr;gap:1px;
    background:var(--border);border-top:1px solid var(--border);
  }
  .podium-card{
    background:var(--card);padding:20px 24px;position:relative;
    transition:background .3s,transform .3s;overflow:hidden;
  }
  /* shimmer sweep on hover */
  .podium-card::after{
    content:'';position:absolute;inset:0;opacity:0;transition:opacity .4s;
    background:linear-gradient(110deg,transparent 30%,rgba(255,255,255,.04) 50%,transparent 70%);
    transform:translateX(-100%);
  }
  .podium-card:hover::after{opacity:1;transform:translateX(100%);transition:transform .7s ease,opacity .1s;}
  .podium-card::before{content:'';position:absolute;inset:0;opacity:0;transition:opacity .3s;
    background:radial-gradient(ellipse at 50% 0%,rgba(255,255,255,.04),transparent 70%);}
  .podium-card:hover::before{opacity:1;}
  .podium-card:hover{background:#121220;transform:translateY(-1px);}
  .podium-card:nth-child(1){border-top:2px solid var(--gold);}
  .podium-card:nth-child(2){border-top:2px solid #7a7a8a;}
  .p-bg-num{
    position:absolute;right:16px;top:6px;font-family:'Orbitron',monospace;
    font-size:58px;font-weight:900;color:rgba(255,255,255,.025);line-height:1;pointer-events:none;
  }
  .p-pill{
    display:inline-flex;align-items:center;gap:5px;padding:3px 10px;
    border-radius:20px;font-size:10px;font-weight:700;letter-spacing:2px;
    text-transform:uppercase;margin-bottom:8px;
  }
  .p-pill.win{background:rgba(255,215,0,.1);color:var(--gold);border:1px solid rgba(255,215,0,.22);}
  .p-pill.p2{background:rgba(180,180,190,.07);color:#999;border:1px solid rgba(180,180,190,.14);}
  .p-name{
    font-family:'Barlow Condensed',sans-serif;font-size:26px;font-weight:800;
    letter-spacing:.5px;line-height:1.1;color:#fff;
  }
  .p-team{font-size:11px;color:var(--muted);margin-top:5px;letter-spacing:.5px;}
  .p-detail{font-size:11px;margin-top:8px;font-weight:600;
    font-family:'Barlow Condensed',sans-serif;letter-spacing:.5px;}

  /* ── NAV ── */
  nav{
    display:flex;border-bottom:1px solid var(--border);padding:0 44px;
    background:rgba(6,6,10,.92);backdrop-filter:blur(10px);
    position:sticky;top:64px;z-index:99;
  }
  .tab{
    padding:15px 32px;font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:700;
    letter-spacing:2.5px;text-transform:uppercase;cursor:pointer;color:var(--muted);
    border-bottom:2px solid transparent;margin-bottom:-1px;
    transition:color .25s,border-color .25s;position:relative;
  }
  .tab::after{
    content:'';position:absolute;bottom:-1px;left:50%;right:50%;height:2px;
    background:linear-gradient(90deg,transparent,var(--steel),transparent);
    filter:blur(4px);transition:left .3s ease,right .3s ease;
  }
  .tab.active{color:var(--steel);border-bottom-color:var(--steel);}
  .tab.active::after{left:0;right:0;}
  .tab:hover:not(.active){color:#c0c8d8;}

  /* ── MAIN ── */
  main{padding:34px 44px;max-width:1360px;margin:0 auto;}

  /* panel transitions */
  .panel{display:none;}
  .panel.active{
    display:block;
    animation:panelIn .35s cubic-bezier(.22,.68,0,1.2) forwards;
  }
  @keyframes panelIn{
    from{opacity:0;transform:translateY(10px) scale(.995);}
    to{opacity:1;transform:translateY(0) scale(1);}
  }
  @keyframes fadeIn{from{opacity:0;}to{opacity:1;}}

  /* controls bar */
  .controls{
    display:flex;align-items:center;gap:16px;margin-bottom:30px;flex-wrap:wrap;
    background:var(--card);border:1px solid var(--border);border-radius:10px;
    padding:14px 22px;
    box-shadow:0 4px 32px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.03);
  }
  .ctrl-group{display:flex;align-items:center;gap:10px;}
  .ctrl-divider{width:1px;height:30px;background:var(--border);margin:0 6px;}
  .ctrl-label{
    font-family:'Barlow Condensed',sans-serif;font-size:10px;letter-spacing:2.5px;
    text-transform:uppercase;color:var(--muted);
  }

  select{
    background:#0a0a16;color:var(--text);border:1px solid var(--border);border-radius:6px;
    padding:8px 36px 8px 14px;font-family:'Barlow',sans-serif;font-size:13px;
    cursor:pointer;outline:none;min-width:165px;appearance:none;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%234a4a6a'/%3E%3C/svg%3E");
    background-repeat:no-repeat;background-position:right 13px center;
    transition:border-color .2s,box-shadow .2s,background .2s;
  }
  select:focus,select:hover{
    border-color:var(--steel);
    box-shadow:0 0 0 3px rgba(168,200,232,.1);
    background-color:#0d0d1e;
  }

  button.run-btn{
    background:linear-gradient(135deg,#c00000,var(--red),#ff1a00);
    background-size:200% 200%;background-position:100% 0;
    color:#fff;border:none;border-radius:6px;
    padding:9px 24px;font-family:'Orbitron',monospace;font-size:11px;font-weight:700;
    letter-spacing:2px;text-transform:uppercase;cursor:pointer;
    transition:background-position .4s ease,box-shadow .3s,transform .2s;
    box-shadow:0 3px 16px rgba(225,6,0,.35);
  }
  button.run-btn:hover{
    background-position:0 100%;
    box-shadow:0 5px 24px rgba(225,6,0,.6);
    transform:translateY(-2px);
  }
  button.run-btn:active{transform:translateY(0);box-shadow:0 2px 8px rgba(225,6,0,.4);}

  /* stat cards */
  .stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px;}
  .stat-card{
    background:var(--card);border:1px solid var(--border);
    border-top:2px solid var(--red);border-radius:10px;
    padding:17px 20px;position:relative;overflow:hidden;
    transition:transform .25s,border-top-color .3s,box-shadow .3s;
    animation:cardIn .4s ease backwards;
  }
  .stat-card:hover{transform:translateY(-3px);box-shadow:0 8px 32px rgba(0,0,0,.5);}
  .stat-card:nth-child(1){animation-delay:.05s;}
  .stat-card:nth-child(2){animation-delay:.1s;}
  .stat-card:nth-child(3){animation-delay:.15s;}
  .stat-card:nth-child(4){animation-delay:.2s;}
  @keyframes cardIn{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:translateY(0);}}
  .stat-card::after{content:'';position:absolute;inset:0;
    background:linear-gradient(135deg,rgba(255,255,255,.025),transparent);pointer-events:none;}
  .stat-label{font-size:9px;letter-spacing:2.5px;text-transform:uppercase;
    color:var(--muted);margin-bottom:8px;font-family:'Barlow Condensed',sans-serif;}
  .stat-value{font-family:'Orbitron',monospace;font-size:26px;font-weight:700;
    transition:text-shadow .3s;}
  .stat-value.red{color:var(--red);text-shadow:0 0 18px rgba(225,6,0,.45);}
  .stat-value.green{color:var(--green);text-shadow:0 0 18px rgba(0,255,170,.35);}
  .stat-value.steel{color:var(--steel);text-shadow:0 0 18px rgba(168,200,232,.35);}
  .stat-value.purple{color:var(--purple);text-shadow:0 0 18px rgba(155,89,255,.35);}

  /* metric cards */
  .metrics-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:22px;}
  .metric-card{
    background:var(--card);border:1px solid var(--border);border-radius:10px;
    padding:19px 16px;text-align:center;position:relative;overflow:hidden;
    transition:transform .25s,box-shadow .25s;
    animation:cardIn .4s ease backwards;
  }
  .metric-card:hover{transform:translateY(-3px);box-shadow:0 8px 32px rgba(0,0,0,.5);}
  .metric-card:nth-child(1){animation-delay:.05s;}
  .metric-card:nth-child(2){animation-delay:.1s;}
  .metric-card:nth-child(3){animation-delay:.15s;}
  .metric-card:nth-child(4){animation-delay:.2s;}
  .metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,var(--steel),transparent);}
  .metric-card::after{content:'';position:absolute;bottom:0;left:0;right:0;height:60px;
    background:radial-gradient(ellipse at 50% 100%,rgba(168,200,232,.04),transparent 70%);
    pointer-events:none;}
  .metric-val{
    font-family:'Orbitron',monospace;font-size:28px;font-weight:700;color:var(--steel);
    text-shadow:0 0 22px rgba(168,200,232,.4);
  }
  .metric-lbl{font-size:9px;letter-spacing:2.5px;text-transform:uppercase;
    color:var(--muted);margin-top:6px;font-family:'Barlow Condensed',sans-serif;}

  /* section title */
  .sec{
    font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:3px;
    text-transform:uppercase;color:var(--steel);margin-bottom:13px;padding-bottom:8px;
    border-bottom:1px solid var(--border);display:flex;align-items:center;gap:9px;
  }
  .sec::before{
    content:'';display:inline-block;width:3px;height:13px;
    background:linear-gradient(180deg,var(--steel),rgba(168,200,232,.3));
    border-radius:2px;flex-shrink:0;
  }

  /* cards */
  .card{
    background:var(--card);border:1px solid var(--border);border-radius:10px;padding:22px;
    box-shadow:0 4px 28px rgba(0,0,0,.35);
    transition:box-shadow .3s;
  }
  .card:hover{box-shadow:0 6px 36px rgba(0,0,0,.5);}
  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}
  .one-col{display:grid;grid-template-columns:1fr;gap:16px;margin-bottom:16px;}
  .card img{width:100%;border-radius:8px;display:block;transition:opacity .3s;opacity:.92;}
  .card img:hover{opacity:1;}

  /* table */
  .table-wrap{overflow-x:auto;}
  table{width:100%;border-collapse:collapse;font-size:13px;}
  thead tr{background:rgba(255,255,255,.025);border-bottom:1px solid var(--border);}
  th{padding:11px 14px;text-align:left;font-family:'Barlow Condensed',sans-serif;
    font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);white-space:nowrap;}
  td{padding:11px 14px;border-bottom:1px solid rgba(255,255,255,.028);}
  tr:last-child td{border-bottom:none;}
  tbody tr{transition:background .15s;}
  tbody tr:hover td{background:rgba(168,200,232,.04);}

  .badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:10px;font-weight:700;letter-spacing:.5px;}
  .badge.yes{background:rgba(225,6,0,.14);color:#ff7070;border:1px solid rgba(225,6,0,.24);}
  .badge.no{background:rgba(255,255,255,.04);color:var(--muted);border:1px solid var(--border);}

  .cbadge{display:inline-block;padding:2px 9px;border-radius:4px;font-size:10px;font-weight:700;letter-spacing:.5px;}
  .cbadge.soft{background:rgba(255,221,68,.13);color:var(--soft);border:1px solid rgba(255,221,68,.24);}
  .cbadge.medium{background:rgba(232,232,232,.07);color:#d0d0d0;border:1px solid rgba(232,232,232,.18);}
  .cbadge.unk,.cbadge.hard{background:rgba(120,120,140,.1);color:#888;border:1px solid rgba(120,120,140,.2);}

  /* loader */
  .loader{
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    padding:70px 20px;gap:18px;color:var(--muted);
  }
  .loader-spinner{
    width:36px;height:36px;border:2px solid var(--border);
    border-top-color:var(--steel);border-radius:50%;
    animation:spin .8s linear infinite;
  }
  @keyframes spin{to{transform:rotate(360deg);}}
  .loader-text{
    font-family:'Orbitron',monospace;font-size:10px;letter-spacing:3px;
    text-transform:uppercase;animation:pulse 1.6s ease-in-out infinite;
  }
  @keyframes pulse{0%,100%{opacity:.35;}50%{opacity:.9;}}

  .err{
    color:#ff7070;padding:20px 22px;font-size:13px;
    background:rgba(225,6,0,.06);border-radius:10px;
    border:1px solid rgba(225,6,0,.2);
    font-family:'Barlow Condensed',sans-serif;letter-spacing:.5px;
    animation:fadeIn .3s ease;
  }

  @media(max-width:900px){.podium-row{grid-template-columns:1fr;}}
  @media(max-width:720px){
    .stats-row,.metrics-row{grid-template-columns:1fr 1fr;}
    .two-col{grid-template-columns:1fr;}
    header,nav,main{padding-left:16px;padding-right:16px;}
    .header-center{display:none;}
    .logo{font-size:17px;}
  }
</style>
</head>
<body>

<!-- HEADER -->
<header>
  <div class="logo">F1<em>·</em>ANALYTICS</div>
  <div class="header-center">Silverstone Circuit &nbsp;·&nbsp; Round 12 &nbsp;·&nbsp; 2024</div>
  <div class="header-right">
    <div class="race-meta">
      <div class="r1">British Grand Prix 2024</div>
      <div class="r2">7 July &nbsp;·&nbsp; 52 Laps &nbsp;·&nbsp; Silverstone</div>
    </div>
    <div class="flag">🇬🇧</div>
  </div>
</header>

<!-- HERO -->
<div class="hero">
  <div class="hero-track">
    <div class="track-line"></div>
    <div class="track-glow"></div>
    <div class="streak"></div>
    <div class="streak"></div>
    <div class="streak"></div>
    <div class="streak"></div>
    <div class="car-wrap">
      <svg width="210" height="54" viewBox="0 0 210 54" fill="none" xmlns="http://www.w3.org/2000/svg">
        <ellipse cx="11" cy="26" rx="8" ry="4" fill="rgba(255,120,0,.55)"/>
        <ellipse cx="7"  cy="26" rx="5" ry="2.5" fill="rgba(255,220,0,.4)"/>
        <rect x="2"  y="11" width="20" height="4" rx="1.5" fill="#c00"/>
        <rect x="5"  y="8"  width="14" height="4" rx="1"   fill="#e10600"/>
        <path d="M20 23 Q32 14 72 13 L135 13 Q164 13 180 21 L188 27 Q180 33 135 34 L72 34 Q32 34 20 29 Z" fill="#e10600"/>
        <path d="M60 13 Q100 11 140 13" stroke="rgba(255,255,255,.14)" stroke-width="1.5" fill="none"/>
        <path d="M82 13 Q102 5 122 13" stroke="#0d0d0d" stroke-width="4" fill="none" stroke-linecap="round"/>
        <rect x="90" y="6" width="22" height="8" rx="3.5" fill="#181828"/>
        <rect x="92" y="8" width="18" height="5" rx="2" fill="rgba(168,200,232,.18)"/>
        <path d="M180 21 L200 26 L188 27 Z" fill="#cc0000"/>
        <rect x="62" y="28" width="66" height="8" rx="2" fill="#b50000"/>
        <rect x="184" y="27" width="22" height="3" rx="1" fill="#cc0000"/>
        <rect x="182" y="22" width="24" height="2" rx="1" fill="#777"/>
        <rect x="22"  y="19" width="155" height="2" rx="1" fill="rgba(168,200,232,.22)"/>
        <circle cx="34"  cy="38" r="11" fill="#0e0e0e" stroke="#282838" stroke-width="2"/>
        <circle cx="34"  cy="38" r="6"  fill="#1e1e2e"/>
        <circle cx="34"  cy="38" r="2.5" fill="#3a3a4a"/>
        <circle cx="165" cy="38" r="11" fill="#0e0e0e" stroke="#282838" stroke-width="2"/>
        <circle cx="165" cy="38" r="6"  fill="#1e1e2e"/>
        <circle cx="165" cy="38" r="2.5" fill="#3a3a4a"/>
        <circle cx="34"  cy="38" r="12.5" fill="none" stroke="rgba(168,200,232,.08)" stroke-width="1.5"/>
        <circle cx="165" cy="38" r="12.5" fill="none" stroke="rgba(168,200,232,.08)" stroke-width="1.5"/>
      </svg>
    </div>
  </div>

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
      <div class="p-detail" style="color:#888">Championship leader at time of race</div>
    </div>
  </div>
</div>

<!-- NAV -->
<nav>
  <div class="tab active" onclick="switchTab('regression',this)">Lap Time Prediction</div>
  <div class="tab" onclick="switchTab('tukey',this)">Tukey / ANOVA</div>
</nav>

<main>

  <!-- REGRESSION PANEL -->
  <div class="panel active" id="panel-regression">
    <div class="controls">
      <div class="ctrl-group">
        <span class="ctrl-label">Driver</span>
        <select id="driver-select">
          <option value="">All Drivers</option>
        </select>
      </div>
      <div class="ctrl-divider"></div>
      <button class="run-btn" onclick="loadRegression()">&#9654;&nbsp; PREDICT</button>
    </div>
    <div id="reg-content"><div class="loader"><div class="loader-spinner"></div><div class="loader-text">Loading…</div></div></div>
  </div>

  <!-- TUKEY PANEL -->
  <div class="panel" id="panel-tukey">
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
    <div id="tukey-content"><div class="loader"><div class="loader-spinner"></div><div class="loader-text">Loading…</div></div></div>
  </div>

</main>

<script>
function loader(){
  return '<div class="loader"><div class="loader-spinner"></div><div class="loader-text">Loading…</div></div>';
}

function switchTab(name, el){
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('panel-' + name).classList.add('active');
  // auto-load Tukey immediately on first visit
  if(name === 'tukey' && !window._tukeyLoaded){
    window._tukeyLoaded = true;
    loadTukey();
  }
}

async function loadDriverList(){
  try{
    const d = await (await fetch('/api/drivers')).json();
    const sel = document.getElementById('driver-select');
    d.drivers.forEach(dr => {
      const o = document.createElement('option');
      o.value = o.textContent = dr;
      sel.appendChild(o);
    });
  } catch(e){}
}

async function loadTukey(){
  const param = document.getElementById('param-select').value;
  document.getElementById('tukey-content').innerHTML = loader();
  const d = await (await fetch('/api/tukey?param=' + param)).json();
  if(d.error){
    document.getElementById('tukey-content').innerHTML = '<div class="err">' + d.error + '</div>';
    return;
  }
  const rows = d.top10.map((r,i) => `<tr style="animation:cardIn .3s ease ${i*.04}s backwards">
    <td>${r.group1}</td>
    <td>${r.group2}</td>
    <td style="font-family:monospace">${(+r.meandiff).toFixed(4)}</td>
    <td style="font-family:monospace">${(+r['p-adj']).toFixed(4)}</td>
    <td><span class="badge ${r.reject ? 'yes' : 'no'}">${r.reject ? 'YES' : 'NO'}</span></td>
  </tr>`).join('');
  document.getElementById('tukey-content').innerHTML = `
    <div class="stats-row">
      <div class="stat-card"><div class="stat-label">Total Comparisons</div><div class="stat-value purple">${d.total}</div></div>
      <div class="stat-card"><div class="stat-label">Significant</div><div class="stat-value red">${d.significant}</div></div>
      <div class="stat-card"><div class="stat-label">Sig. Rate</div><div class="stat-value steel">${d.sig_pct}<span style="font-size:13px;color:var(--muted)">%</span></div></div>
      <div class="stat-card"><div class="stat-label">Avg |Δ Mean|</div><div class="stat-value green">${d.avg_diff}</div></div>
    </div>
    <div class="card">
      <div class="sec">Top 10 Largest Pairwise Differences</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Group 1</th><th>Group 2</th><th>Mean Diff</th><th>p-adj</th><th>Reject H₀</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>`;
}

async function loadRegression(){
  const driver = document.getElementById('driver-select').value;
  const el     = document.getElementById('reg-content');
  el.innerHTML = loader();
  const d = await (await fetch('/api/regression?driver=' + encodeURIComponent(driver))).json();
  if(d.error){ el.innerHTML = '<div class="err">' + d.error + '</div>'; return; }

  const predRows = d.preview.map((r,i) => {
    const cls = r.compound ? r.compound.toLowerCase() : 'unk';
    return `<tr style="animation:cardIn .3s ease ${i*.03}s backwards">
      <td><strong>${r.driver}</strong></td>
      <td><span class="cbadge ${cls}">${r.compound || 'UNK'}</span></td>
      <td style="font-family:monospace">${r.actual}</td>
      <td style="font-family:monospace">${r.predicted}</td>
      <td style="font-family:monospace;color:${Math.abs(r.error) > 1 ? 'var(--accent)' : 'var(--green)'}">${r.error}</td>
    </tr>`;
  }).join('');

  const isAll   = (driver === '');
  const histHtml = isAll
    ? `<div class="card"><div class="sec">Error Distribution</div><img src="data:image/png;base64,${d.hist}" loading="lazy"/></div>`
    : '';
  const wrapCls = isAll ? 'two-col' : 'one-col';
  const filterLabel = driver || 'All Drivers';

  el.innerHTML = `
    <div class="metrics-row">
      <div class="metric-card"><div class="metric-val">${d.rmse}</div><div class="metric-lbl">RMSE</div></div>
      <div class="metric-card"><div class="metric-val">${d.mse}</div><div class="metric-lbl">MSE</div></div>
      <div class="metric-card"><div class="metric-val">${d.mae}</div><div class="metric-lbl">MAE</div></div>
      <div class="metric-card"><div class="metric-val">${d.mean_error}</div><div class="metric-lbl">Mean Error</div></div>
    </div>
    <div class="${wrapCls}">
      ${histHtml}
      <div class="card"><div class="sec">Predicted vs Actual Lap Times</div><img src="data:image/png;base64,${d.lineplot}" loading="lazy"/></div>
    </div>
    <div class="card">
      <div class="sec">Sample Predictions — ${filterLabel} (first 15 rows)</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Driver</th><th>Compound</th><th>Actual (s)</th><th>Predicted (s)</th><th>Error</th></tr></thead>
          <tbody>${predRows}</tbody>
        </table>
      </div>
    </div>`;
}

// init
loadDriverList();
loadRegression();
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
    driver_filter = request.args.get("driver", "").strip()

    try:
        df = pd.read_csv(os.path.join(DATA_FOLDER, "f1_deployment_data.csv"))
        driver_cols  = [c for c in df.columns if c.startswith("Driver_")]
        compound_cols = [c for c in df.columns if c.startswith("Compound_")]
        plot_df = df.copy()

        if driver_filter:
            col = f"Driver_{driver_filter}"
            if col not in df.columns:
                return jsonify({"error": f"Column '{col}' not found in CSV."})
            plot_df = plot_df[plot_df[col].astype(str).str.upper().isin(['TRUE', '1', 'YES'])]

        if plot_df.empty:
            return jsonify({"error": "No rows match the selected filters."})

        plot_df = plot_df.copy()

        # ── derive driver label: find which Driver_XXX col is TRUE per row ──
        def row_driver(r):
            for c in driver_cols:
                val = r.get(c, '') if hasattr(r, 'get') else r[c]
                if str(val).upper() in ('TRUE', '1', 'YES'):
                    return c.replace("Driver_", "")
            return "UNK"

        # ── derive compound label: find which Compound_XXX col is TRUE per row ──
        # Uses vectorised approach to avoid UNK from index mismatch
        def build_compound_labels(frame):
            labels = pd.Series("UNK", index=frame.index)
            for cname in compound_cols:
                mask = frame[cname].astype(str).str.upper().isin(['TRUE', '1', 'YES'])
                compound_name = cname.replace("Compound_", "")
                labels[mask] = compound_name
            return labels

        plot_df['_driver_label']   = plot_df.apply(row_driver, axis=1)
        plot_df['_compound_label'] = build_compound_labels(plot_df)

        actual    = plot_df['LapTime'].values
        predicted = plot_df['Predicted_LapTime'].values
        error     = predicted - actual

        rmse     = round(float(np.sqrt(mean_squared_error(actual, predicted))), 4)
        mse      = round(float(mean_squared_error(actual, predicted)), 4)
        mae      = round(float(mean_absolute_error(actual, predicted)), 4)
        mean_err = round(float(np.mean(error)), 6)

        # histogram
        fig, ax = plt.subplots(figsize=(5.5, 3.2), facecolor='#06060a')
        ax.set_facecolor('#0f0f18')
        ax.hist(error, bins=14, color='#e10600', edgecolor='#ff5555', linewidth=0.6, alpha=0.9)
        ax.tick_params(colors='#4a4a6a', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#1e1e2e')
        ax.set_xlabel("Prediction Error (s)", color='#4a4a6a', fontsize=8)
        ax.set_ylabel("Frequency",            color='#4a4a6a', fontsize=8)
        ax.set_title("Error Distribution",    color='#7a8aaa', fontsize=9, pad=8)
        ax.grid(True, linestyle='--', alpha=0.12, color='#2a2a4a')
        buf = io.BytesIO(); plt.tight_layout()
        plt.savefig(buf, format='png', dpi=120, facecolor='#06060a'); plt.close()
        hist_b64 = base64.b64encode(buf.getvalue()).decode()

        # line plot
        n   = min(50, len(plot_df))
        idx = np.arange(n)
        fig2, ax2 = plt.subplots(figsize=(6.5, 3.2), facecolor='#06060a')
        ax2.set_facecolor('#0f0f18')
        ax2.plot(idx, actual[:n],    color='#a8c8e8', marker='o', markersize=3.5,
                 linewidth=1.5, label='Actual')
        ax2.plot(idx, predicted[:n], color='#e10600', marker='x', markersize=3.5,
                 linewidth=1.5, label='Predicted', linestyle='--')
        ax2.tick_params(colors='#4a4a6a', labelsize=8)
        for sp in ax2.spines.values(): sp.set_edgecolor('#1e1e2e')
        ax2.set_xlabel("Lap Index",    color='#4a4a6a', fontsize=8)
        ax2.set_ylabel("Lap Time (s)", color='#4a4a6a', fontsize=8)
        title = f"Predicted vs Actual · {driver_filter}" if driver_filter else "Predicted vs Actual · All Drivers"
        ax2.set_title(title, color='#7a8aaa', fontsize=9, pad=8)
        ax2.legend(fontsize=8, facecolor='#0f0f18', edgecolor='#2a2a3a', labelcolor='#aaa')
        ax2.grid(True, linestyle='--', alpha=0.12, color='#2a2a4a')
        buf2 = io.BytesIO(); plt.tight_layout()
        plt.savefig(buf2, format='png', dpi=120, facecolor='#06060a'); plt.close()
        line_b64 = base64.b64encode(buf2.getvalue()).decode()

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
