"""Render the benchmark results as one self-contained HTML file.

No external CSS, fonts or scripts, so the file opens anywhere (email
attachment, Slack, a laptop with no internet). Called by cost_report.py.
"""
from __future__ import annotations

import html
import time

CSS = """
:root{--bg:#fbfaf8;--fg:#1f2328;--muted:#5f6670;--line:#e3e0da;--card:#fff;--accent:#2f6fb0;
--api:#c9821f;--good:#2e7d4f;--bad:#b3261e;--chip:#f1efea}
@media (prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e6e6e6;--muted:#9aa1ab;--line:#2c3036;
--card:#1d2025;--accent:#6aa5e0;--api:#e0a54a;--good:#5fbf86;--bad:#ef7a72;--chip:#262a30}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1240px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:40px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:22px 0 8px}.muted{color:var(--muted)}.small{font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:20px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .v{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}.kpi .l{color:var(--muted);font-size:13px}
.over{color:var(--bad)}.under{color:var(--good)}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{white-space:nowrap}td.n{white-space:nowrap}
th{font-weight:600;color:var(--muted);background:var(--chip)}tr:last-child td{border-bottom:0}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr.op td{background:color-mix(in srgb,var(--accent) 12%,transparent);font-weight:600}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600}
.pill.ok{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}
.pill.no{background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad)}
.pill.na{background:var(--chip);color:var(--muted)}
ul{padding-left:20px}li{margin:3px 0}code{background:var(--chip);padding:1px 5px;border-radius:4px;font-size:12.5px}
.legend{display:flex;gap:16px;font-size:13px;color:var(--muted);margin:6px 0 0}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}
svg text{fill:var(--fg)}svg .grid{stroke:var(--line)}
.ctl{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;
background:var(--bg);padding:10px 0;margin:8px 0 0;border-bottom:1px solid var(--line)}
.ctl label{font-weight:600}.ctl input,.ctl button,.ctl select{font:inherit;padding:4px 8px;border:1px solid var(--line);
border-radius:6px;background:var(--card);color:var(--fg)}.ctl input{width:96px}.ctl button{cursor:pointer}
.chg{color:var(--api);font-weight:600}
"""


class H(str):
    """Text that is already HTML (not escaped by e())."""


def e(x) -> str:
    if isinstance(x, H):
        return str(x)
    return html.escape("" if x is None else str(x))


def num(x, d=2, pre="", suf=""):
    if x is None:
        return "–"
    return f"{pre}{x:,.{d}f}{suf}"


def usd(x, d=2):
    return num(x, d, "$")


def pill(ok):
    if ok is None:
        return '<span class="pill na">not run</span>'
    return '<span class="pill ok">met</span>' if ok else '<span class="pill no">missed</span>'


def table(headers, rows, numeric=(), highlight=None):
    th = "".join(f'<th class="{"n" if i in numeric else ""}">{e(h)}</th>' for i, h in enumerate(headers))
    body = []
    for ri, r in enumerate(rows):
        cls = ' class="op"' if highlight is not None and ri == highlight else ""
        tds = "".join(f'<td class="{"n" if i in numeric else ""}">{c}</td>' for i, c in enumerate(r))
        body.append(f"<tr{cls}>{tds}</tr>")
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def bar_chart(items, target):
    """Horizontal bars: self-hosted vs hosted-API $/student-month per spec line."""
    items = [it for it in items if it["self"] is not None or it["api"] is not None]
    if not items:
        return ""
    raw_max = max([target] + [v for it in items for v in (it["self"], it["api"]) if v is not None])
    step = nice_step(raw_max / 5)
    vmax = step * (int(raw_max / step) + 1)
    W, left, right, rowh, bh = 760, 250, 70, 44, 14
    H = 24 + rowh * len(items) + 24
    sx = lambda v: left + (W - left - right) * v / vmax  # noqa: E731
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Cost per student-month by spec line">']
    for t in range(0, int(round(vmax / step)) + 1):
        v = step * t
        x = sx(v)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="16" x2="{x:.1f}" y2="{H - 22}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{H - 6}" font-size="11" text-anchor="middle" opacity=".7">${v:,.2f}</text>')
    for i, it in enumerate(items):
        y = 24 + i * rowh
        weight = ' font-weight="650"' if it.get("total") else ""
        parts.append(f'<text x="{left - 10}" y="{y + 16}" font-size="13" text-anchor="end"{weight}>{e(it["label"])}</text>')
        for j, (key, color) in enumerate((("self", "var(--accent)"), ("api", "var(--api)"))):
            v = it[key]
            yy = y + j * (bh + 2)
            if v is None:
                continue
            w = max(1.5, sx(v) - left)
            parts.append(f'<rect x="{left}" y="{yy}" width="{w:.1f}" height="{bh}" rx="2" fill="{color}"/>')
            parts.append(f'<text x="{left + w + 6:.1f}" y="{yy + 11}" font-size="11.5">${v:,.2f}</text>')
    tx = sx(target)
    parts.append(f'<line x1="{tx:.1f}" y1="14" x2="{tx:.1f}" y2="{H - 22}" stroke="var(--bad)" stroke-width="1.5" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{tx + 4:.1f}" y="12" font-size="11" fill="var(--bad)">${target:.2f} budget</text>')
    parts.append("</svg>")
    legend = ('<div class="legend"><span><span class="sw" style="background:var(--accent)"></span>Self-hosted (measured)</span>'
              '<span><span class="sw" style="background:var(--api)"></span>Hosted API (list price)</span></div>')
    return "".join(parts) + legend


def nice_step(x):
    """Round a raw tick step up to 1, 2 or 5 x 10^n."""
    import math
    if x <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 5, 10):
        if x <= m * mag:
            return m * mag
    return 10 * mag


def session_section(res, op_conc, label=None):
    rows, hi = [], None
    for i, lv in enumerate(res["levels"]):
        if lv["concurrency"] == op_conc:
            hi = i
        g = (lv.get("gpu") or {}).get("per_gpu") or {}
        util = sum(v["util_mean"] for v in g.values()) / len(g) if g else None
        power = sum((v["power_mean_w"] or 0) for v in g.values()) if g else None
        mem = max(v["mem_peak_mib"] for v in g.values()) / 1024 if g else None
        cache = lv.get("cache_hit_rate_usage") or lv.get("cache_hit_rate_metrics")
        rows.append([
            e(lv["concurrency"]), num(lv["output_tok_per_s"], 0), num(lv["turns_per_s"], 2),
            num(lv["ttft_s"].get("p50"), 2), num(lv["ttft_s"].get("p95"), 2),
            num(lv["tpot_ms"].get("p50"), 1), num(lv["tpot_ms"].get("p95"), 1),
            num(lv["per_stream_output_tok_per_s"].get("p50"), 0),
            num(cache * 100 if cache is not None else None, 1, suf="%"),
            num(lv.get("preemptions"), 0), num((lv.get("kv_cache_usage_max") or 0) * 100 if lv.get("kv_cache_usage_max") is not None else None, 0, suf="%"),
            num(util, 0, suf="%"), num(power, 0, suf=" W"), num(mem, 1, suf=" GB"), e(lv["errors"]),
        ])
    p = res["profile"]
    t = res["target"]
    head = (f'<h3>{e(label or res["target_name"])} · Socratic sessions <span class="muted small">· {e(t.get("model"))} · {e(t.get("gpus"))} GPU(s)</span></h3>'
            f'<p class="small muted">Session shape: {p["system_prompt_tokens"]:,}-token shared system prompt, '
            f'{p["turn_new_input_tokens"]:,} new + {p["turn_output_tokens"]:,} output tokens per turn, '
            f'history summarised at {p["compact_at_tokens"]:,} tokens. Highlighted row = operating point used for cost.</p>')
    return head + table(
        ["Students", "Out tok/s", "Turns/s", "TTFT p50 s", "TTFT p95 s", "TPOT p50 ms", "TPOT p95 ms",
         "Tok/s per student", "Cache hit", "Preempt", "KV max", "GPU util", "Power", "Peak mem", "Errors"],
        rows, numeric=set(range(15)), highlight=hi)


def check_cell(c):
    """Measured value and how far it is from the target."""
    if c["measured"] is None:
        return "not measured"
    m, t, u = c["measured"], c["target"], c["unit"]
    if c["kind"] == "max":
        if m > t:
            return f'<b class="over">{m:.2f} {e(u)}</b> <span class="small">({m / t:.1f}× the target)</span>'
        return f'{m:.2f} {e(u)} <span class="small muted">({m / t:.0%} of the target)</span>'
    if m < t:
        return f'<b class="over">{m:.2f}{e(u)}</b> <span class="small">({1 - m / t:.0%} below the target)</span>'
    return f'{m:.2f}{e(u)} <span class="small muted">(above the target)</span>'


def money(x):
    """Dollar amount with enough decimals to show small unit prices."""
    if x is None:
        return "–"
    if x == 0:
        return "$0"
    d = 2 if abs(x) >= 1 else 3 if abs(x) >= 0.1 else 4 if abs(x) >= 0.001 else 6
    return f"${x:,.{d}f}"


def rng(r):
    if not r:
        return "–"
    lo, hi = r
    return money(lo) if lo == hi else f"{money(lo)}–{money(hi)}"




JS = r"""
(function(){
var P0 = __P0__, C = __CHART__;
var input = document.getElementById("price"), reset = document.getElementById("reset"), note = document.getElementById("pnote"),
    pick = document.getElementById("machine");
function money(x){ if (x == null || !isFinite(x)) return "–"; if (x === 0) return "$0";
  var a = Math.abs(x), d = a >= 1 ? 2 : a >= 0.1 ? 3 : a >= 0.001 ? 4 : 6;
  return "$" + x.toLocaleString("en-US", {minimumFractionDigits: d, maximumFractionDigits: d}); }
function money2(x){ return "$" + x.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2}); }
function esc(t){ return String(t).replace(/[&<>"]/g, function(c){ return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]; }); }
function niceStep(x){ if (x <= 0) return 1; var m = Math.pow(10, Math.floor(Math.log10(x)));
  var ks = [1, 2, 5, 10]; for (var i = 0; i < ks.length; i++) if (x <= ks[i] * m * (1 + 1e-12)) return ks[i] * m; return 10 * m; }
function chart(p){
  var el = document.getElementById("chart"); if (!el || !C) return;
  var items = C.items.map(function(it){ return {label: it.label, self: it.g == null ? null : it.g * p / 3600, api: it.api, t: it.t}; });
  var st = 0, bm = 0;
  items.forEach(function(it){ if (!it.t) return; st += it.self || 0;
    var v = [it.self, it.api].filter(function(x){ return x != null; }); bm += v.length ? Math.min.apply(null, v) : 0; });
  items.push({label: C.totalLabel, self: st, api: bm, total: true});
  items = items.filter(function(it){ return it.self != null || it.api != null; });
  var target = C.target, rawMax = target;
  items.forEach(function(it){ [it.self, it.api].forEach(function(v){ if (v != null && v > rawMax) rawMax = v; }); });
  var step = niceStep(rawMax / 5), vmax = step * (Math.floor(rawMax / step) + 1);
  var W = 760, left = 250, right = 70, rowh = 44, bh = 14, H = 24 + rowh * items.length + 24;
  var sx = function(v){ return left + (W - left - right) * v / vmax; };
  var o = ['<svg viewBox="0 0 ' + W + ' ' + H + '" width="100%" role="img" aria-label="Cost per student-month by spec line">'];
  for (var t = 0; t <= Math.round(vmax / step); t++) { var v = step * t, x = sx(v).toFixed(1);
    o.push('<line class="grid" x1="' + x + '" y1="16" x2="' + x + '" y2="' + (H - 22) + '" stroke-width="1"/>');
    o.push('<text x="' + x + '" y="' + (H - 6) + '" font-size="11" text-anchor="middle" opacity=".7">' + money2(v) + '</text>'); }
  items.forEach(function(it, i){ var y = 24 + i * rowh;
    o.push('<text x="' + (left - 10) + '" y="' + (y + 16) + '" font-size="13" text-anchor="end"' + (it.total ? ' font-weight="650"' : '') + '>' + esc(it.label) + '</text>');
    [["self", "var(--accent)"], ["api", "var(--api)"]].forEach(function(kc, j){ var v = it[kc[0]]; if (v == null) return;
      var yy = y + j * (bh + 2), w = Math.max(1.5, sx(v) - left);
      o.push('<rect x="' + left + '" y="' + yy + '" width="' + w.toFixed(1) + '" height="' + bh + '" rx="2" fill="' + kc[1] + '"/>');
      o.push('<text x="' + (left + w + 6).toFixed(1) + '" y="' + (yy + 11) + '" font-size="11.5">' + money2(v) + '</text>'); }); });
  var tx = sx(target).toFixed(1);
  o.push('<line x1="' + tx + '" y1="14" x2="' + tx + '" y2="' + (H - 22) + '" stroke="var(--bad)" stroke-width="1.5" stroke-dasharray="4 3"/>');
  o.push('<text x="' + (sx(target) + 4).toFixed(1) + '" y="12" font-size="11" fill="var(--bad)">' + money2(target) + ' budget</text></svg>');
  el.innerHTML = o.join("");
}
function apply(p){
  document.querySelectorAll(".usd").forEach(function(el){
    var v = ((+el.dataset.g) * p / 3600 / (+el.dataset.d) + (+el.dataset.f)) * (+el.dataset.m);
    el.textContent = money(v);
    if (el.dataset.t !== undefined) { el.classList.toggle("over", v > +el.dataset.t); el.classList.toggle("under", v <= +el.dataset.t); } });
  document.querySelectorAll(".ratio").forEach(function(el){
    var v = (+el.dataset.g) * p / 3600 / (+el.dataset.api);
    el.textContent = v.toFixed(1) + "×"; el.classList.toggle("over", v > 1); el.classList.toggle("under", v <= 1); });
  document.querySelectorAll(".price").forEach(function(el){ el.textContent = "$" + p.toFixed(2); });
  document.querySelectorAll(".price-s").forEach(function(el){ el.textContent = "$" + (p / 3600).toFixed(6); });
  chart(p);
  if (note) note.hidden = Math.abs(p - P0) < 1e-9;
}
input.addEventListener("input", function(){ var p = parseFloat(input.value); if (isFinite(p) && p >= 0) { apply(p);
  if (pick) { var m = Array.prototype.find.call(pick.options, function(o){ return o.value && Math.abs(+o.value - p) < 1e-9; }); pick.value = m ? m.value : ""; } } });
if (pick) pick.addEventListener("change", function(){ if (pick.value) { input.value = (+pick.value).toFixed(2); apply(+pick.value); } });
reset.addEventListener("click", function(){ input.value = P0.toFixed(2); apply(P0); if (pick) pick.selectedIndex = pick.dataset.i0; });
})();
"""


def render(ctx: dict) -> str:
    import json as _json

    spec, rows, notes = ctx["spec"], ctx["rows"], ctx["notes"]
    gcfg, bill, day, slo = spec["gpu"], spec["billing"], spec["student_day"], spec["slo"]
    target = bill["target_usd_per_student_month"]
    missing = ctx["missing"]
    units, chosen, llm_pick = ctx.get("units", []), ctx.get("chosen", []), ctx.get("llm_pick")
    pricing = ctx["pricing"]
    price, dpm = pricing["usd_per_gpu_hour"], bill["days_per_month"]
    util = gcfg.get("utilization", 0.3)
    levels = (spec.get("report") or {}).get("utilization_levels") or [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    m = ctx.get("machine") or {}
    gpus = ", ".join(m.get("gpus", [])) or "not recorded"
    llm_rows = [r for r in rows if r["kind"] == "llm"]
    speech_rows = [r for r in rows if r["kind"] == "speech"]
    total_label = f"Total (with {llm_pick['model']})" if llm_pick else "Total"

    def D(gs, fixed=0.0, div=1.0, mult=1.0, tgt=None):
        """A $ figure that the page recomputes when the $/GPU-hour box changes:
        (GPU-seconds x price / 3600 / div + fixed) x mult."""
        if gs is None and not fixed:
            return "–"
        g = float(gs or 0.0)
        v = (g * price / 3600 / div + fixed) * mult
        t = cls = ""
        if tgt is not None:
            t, cls = f' data-t="{tgt!r}"', (" over" if v > tgt else " under")
        return (f'<span class="usd{cls}" data-g="{g!r}" data-f="{float(fixed)!r}" data-d="{float(div)!r}" '
                f'data-m="{float(mult)!r}"{t}>{money(v)}</span>')

    def ratio(gs, api_usd):
        if gs is None or not api_usd:
            return "–"
        v = gs * price / 3600 / api_usd
        return f'<b class="ratio {"over" if v > 1 else "under"}" data-g="{float(gs)!r}" data-api="{float(api_usd)!r}">{v:.1f}×</b>'

    P = H(f'<span class="price">${price:.2f}</span>')
    pure_gs = sum(r["gpu_s_per_student_day"] for r in chosen if r.get("gpu_s_per_student_day") is not None)

    def kpi(v_html, label):
        return f'<div class="kpi"><div class="v">{v_html}</div><div class="l">{label}</div></div>'

    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Token Factory Baseline</title><style>{CSS}</style></head><body><main>"]
    out.append(f"<h1>Token Factory baseline: cost per student</h1>")
    out.append(f'<p class="muted">{e(pricing["label"])} · generated {e(time.strftime("%d %b %Y, %H:%M"))} on '
               f'{e(m.get("host", "?"))} · GPUs: {e(gpus)}</p>')
    opts, i0 = [], 0
    for k, (mk, mv) in enumerate(pricing.get("all", {}).items()):
        sel = mk == pricing["machine"] and pricing["source"].startswith("machines.")
        if sel:
            i0 = k
        opts.append(f'<option value="{mv["usd_per_gpu_hour"]!r}"{" selected" if sel else ""}>'
                    f'{e(mv["label"])} (${mv["usd_per_gpu_hour"]:.2f})</option>')
    opts.append(f'<option value=""{"" if pricing["source"].startswith("machines.") else " selected"}>Custom price</option>')
    if not pricing["source"].startswith("machines."):
        i0 = len(opts) - 1
    out.append(f'<div class="ctl"><label for="machine">Machine</label><select id="machine" data-i0="{i0}">{"".join(opts)}</select>'
               f'<label for="price">$ per GPU-hour</label>'
               f'<input id="price" type="number" min="0" step="0.05" value="{price:.2f}">'
               f'<button id="reset" type="button">Reset to ${price:.2f}</button>'
               f'<span class="small muted">Results priced at ${price:.2f}/GPU-hour from {e(pricing["source"])} '
               f'(machine chosen by {e(pricing["why_machine"])}). Change it and every $ figure below updates.</span>'
               f'<span id="pnote" class="small chg" hidden>Showing a price you picked; report.md and report.json keep ${price:.2f}.</span></div>')
    miss = f' Not yet measured, so left out of the totals: {e(", ".join(missing))}.' if missing else ""
    out.append('<div class="kpis">'
               + kpi(D(pure_gs), f"per student-day, pure compute (GPU busy 100%) at {P}/GPU-hour")
               + kpi(D(pure_gs, mult=dpm, tgt=target), f"per student-month, pure compute ({dpm} days)")
               + kpi(D(pure_gs, div=util, mult=dpm, tgt=target), f"per student-month at {util:.0%} utilization")
               + kpi(money(target), "target per student-month") + "</div>")
    with_llm = f" ASR + TTS + OCR + {e(llm_pick['model'])}." if llm_pick else ""
    out.append(f'<p class="small muted">Pure compute charges only the GPU seconds the work actually used; real cost is that '
               f'divided by utilization (see the utilization table).{with_llm}{miss}</p>')

    # ---- LLM: Socratic sessions vs public benchmark
    glance = ctx.get("glance") or []
    if glance:
        out.append("<h2>LLM: Socratic sessions vs public benchmark</h2>")
        out.append('<p class="small muted">Each model at its best load that met the latency targets (or the closest load, marked). '
                   "“Hosted API, same tokens” prices the same input and output tokens at that model’s cheapest listed API price "
                   "(<code>api_price</code> in llm_targets.yaml). Above 1× means self-hosting costs more.</p>")
        grow = []
        for g in glance:
            op = e(g["op"]) + ("" if g["slo_ok"] else ' <span class="pill no">missed target</span>')
            grow.append([e(g["model"]), e(g["workload"]), op, num(g["out_tps"], 0),
                         f'{g["mean_in"]:,.0f} / {g["mean_out"]:,.0f}', D(g["gs_per_m_out"]), money(g["api_per_m_out"]),
                         ratio(g["gs_per_m_out"], g["api_per_m_out"])])
        out.append(table(["Model", "Workload", "Operating point", "Output tok/s", "Tokens in / out per request",
                          "Self-hosted $ per 1M output", "Hosted API, same tokens", "vs hosted"],
                         grow, numeric={3, 4, 5, 6, 7}))
        out.append('<p class="small muted">The GPU time for each 1M output tokens includes reading the prompts that came with them. '
                   "Socratic turns carry ~20K tokens of prompt each; public-benchmark (ShareGPT) requests carry a few hundred.</p>")

    # ---- usage
    lv = day["llm"]
    out.append("<h2>Usage per student-day</h2>")
    out.append('<p class="small muted">Fixed for this report; edit <code>student_day</code> in <code>config/spec.yaml</code> '
               "and re-run <code>bench/cost_report.py</code> (no new benchmark needed).</p>")
    urows = [["1. ASR", f"{day['asr_minutes']} min of audio"], ["2. TTS", f"{day['tts_minutes']} min of audio"],
             ["3. OCR", f"{day['ocr_pages']} pages"]]
    for r in llm_rows:
        urows.append([e(r["line"]), f"{lv['cached_input_tokens']:,} cached + {lv['uncached_input_tokens']:,} new input + "
                                    f"{lv['output_tokens']:,} output tokens"])
    out.append(table(["Line", "Daily volume"], urows))

    # ---- raw unit costs
    out.append("<h2>Unit costs (raw)</h2>")
    out.append(f'<p class="small muted">GPU-seconds per unit is the raw measurement: multiply by your own $/GPU-second to price it. '
               f'The $ column uses {P}/GPU-hour with the GPU busy 100% of the time. '
               f'Target range is from the planning sheet (<code>market_reference</code> in spec.yaml).</p>')
    trs = []
    for u in units:
        model = e(u["model"]) + (f'<br><span class="muted small">{e(u["note"])}</span>' if u.get("note") else "")
        gs = u["gpu_s_per_unit"]
        trs.append([e(u["line"]), model, e(u["unit"]), e(u["throughput"]),
                    num(gs, 1) if gs else ("0" if gs == 0 else "–"), D(gs) if gs is not None else "–", rng(u.get("market"))])
    out.append(table(["Line", "Model", "Per", "Measured", "GPU-seconds per unit",
                      H(f"$ per unit at {P}/GPU-hr, 100% busy"), "Target range"], trs, numeric={4, 5, 6}))

    # ---- how it's calculated
    out.append("<h2>How the numbers are calculated</h2><ol>")
    out.append(f"<li><b>Price of one GPU-second</b> = {P} per GPU-hour ÷ 3,600 = "
               f'<span class="price-s">${price / 3600:.6f}</span>. (From {e(pricing["source"])}; try other prices at the top.)</li>')
    out.append("<li><b>GPU-seconds per unit</b> = units ÷ measured throughput, × the number of GPUs the model occupies. "
               "Throughput is measured at the operating point: the highest load that still met the latency targets.</li>")
    out.append("<li><b>$ per unit</b> = GPU-seconds per unit × price of one GPU-second. This assumes the GPU is never idle.</li>")
    out.append("<li><b>$ per student-day</b> = Σ (daily volume × $ per unit) ÷ utilization. Utilization is the share of the "
               "day the GPU does useful work.</li>")
    out.append("<li><b>LLM input vs output split</b> is approximate. The GPU reads prompts and writes replies in the same "
               "batches, so the benchmark measures them together (the all-in row). The split assigns each turn's time before "
               "its first token to reading the new input and the rest to writing; queue waiting counts as reading. Cached tokens "
               "need almost no compute, but they occupy GPU memory, which caps how many students fit at once.</li></ol>")
    ex = next((u for u in units if u["unit"].endswith("all-in") and u["gpu_s_per_unit"]), None)
    if ex:
        gs = ex["gpu_s_per_unit"]
        per_day_gs = gs * lv["output_tokens"] / 1e6
        out.append(f'<p class="small"><b>Worked example</b> ({e(ex["model"])}, {e(ex["throughput"])}): '
                   f'1,000,000 output tokens take {gs:,.0f} GPU-seconds ({gs / 3600:.2f} GPU-hours) → '
                   f'{gs / 3600:.2f} × {P} = {D(gs)} per 1M output tokens, including the input that came with them. '
                   f'A student-day of {lv["output_tokens"]:,} output tokens is {per_day_gs:,.0f} GPU-seconds = '
                   f'{D(per_day_gs)} at 100% busy, or {D(per_day_gs, div=util)} at {util:.0%} utilization.</p>')

    # ---- utilization table: one column per line, total = speech + the chosen LLM
    out.append("<h2>Cost per student by utilization</h2>")
    out.append(f'<p class="small muted">Daily volumes as above, {P}/GPU-hour, {dpm} days per month. '
               f'Rows: <code>report.utilization_levels</code> in spec.yaml. The planning assumption ({util:.0%}) is highlighted.</p>')
    cols = speech_rows + llm_rows
    tot_gs = sum(r["gpu_s_per_student_day"] for r in chosen if r.get("gpu_s_per_student_day") is not None)
    trs, hi = [], None
    for i, u in enumerate(levels):
        cells = [f"{u:.0%}"] + [D(r["gpu_s_per_student_day"], div=u) if r["gpu_s_per_student_day"] is not None else "–"
                                for r in cols]
        cells += [D(tot_gs, div=u), D(tot_gs, div=u, mult=dpm)]
        if abs(u - util) < 1e-9:
            hi = i
        trs.append(cells)
    api_cells = ["Hosted APIs"] + [money(r.get("api_usd_day")) for r in cols]
    api_tot = sum(r["api_usd_day"] for r in chosen if r.get("api_usd_day") is not None)
    trs.append(api_cells + [money(api_tot), money(api_tot * dpm)])
    heads = ["Utilization"] + [r["line"].split(". ", 1)[-1].split(" (")[0] + " $/day" for r in cols] + \
            [f"{total_label} $/day", f"{total_label} $/month"]
    out.append(table(heads, trs, numeric=set(range(1, len(heads))), highlight=hi))
    out.append('<p class="small muted">"Hosted APIs" uses Bodhan\'s list prices for ASR/TTS/OCR and the cheapest listed host '
               "for each LLM (<code>api_reference</code> in spec.yaml). Lines not yet measured show –.</p>")

    # ---- chart (pure compute); redrawn by the script when the price changes
    citems, items = [], []
    for r in cols:
        g = None if r["gpu_s_per_student_day"] is None else r["gpu_s_per_student_day"] * dpm
        in_total = r in chosen
        citems.append({"label": r["line"].split(" (")[0], "g": g, "api": r.get("api_usd_month"), "t": in_total})
        items.append({"label": r["line"].split(" (")[0], "self": None if g is None else g * price / 3600,
                      "api": r.get("api_usd_month"), "t": in_total})
    best_mix = sum(min([v for v in (it["self"], it["api"]) if v is not None] or [0]) for it in items if it["t"])
    items.append({"label": total_label, "self": pure_gs * price / 3600 * dpm, "api": best_mix, "total": True})
    out.append("<h2>Cost per student-month by line, pure compute</h2>")
    out.append('<p class="small muted">Blue: self-hosted at 100% busy. Orange: hosted API list price. '
               "For the total, orange is the cheaper of the two per line.</p>")
    out.append(f'<div id="chart">{bar_chart(items, target)}</div>')

    out.append("<h2>Operating points</h2>")
    out.append('<p class="small muted">The load level each cost is based on: the highest that met every latency target, '
               "or, if none did, the one closest to them. Targets are in <code>slo</code> in spec.yaml.</p>")
    trs = []
    for r in rows:
        checks = r.get("checks") or []
        tgt = "<br>".join(e(f"{c['name']}: {'≤' if c['kind'] == 'max' else '≥'} {c['target']:g} {c['unit']}") for c in checks) or "–"
        meas = "<br>".join(check_cell(c) for c in checks) or "–"
        trs.append([e(r["line"]), e(r["model"]), e(r["measured"]), e(r["throughput"]),
                    num(r["gpu_s_per_student_day"], 1), D(r["gpu_s_per_student_day"]),
                    money(r.get("api_usd_day")), tgt, meas, pill(r["slo_met"])])
    out.append(table(["Line", "Model", "Operating point", "Throughput", "GPU-s / student-day",
                      "$/day pure compute", "Hosted API $/day", "Latency target", "Measured", "Met"],
                     trs, numeric={4, 5, 6}))

    # details
    R = ctx["raw"]
    out.append("<h2>Details</h2>")
    for res, op, label in R.get("sessions", []):
        out.append(session_section(res, op, label))
    for st in ctx.get("public") or []:
        res = st["res"]
        t = res["target"]
        out.append(f'<h3>{e(st["label"])} · public benchmark ({e(res["dataset"])}) '
                   f'<span class="muted small">· {e(res.get("tool", "vllm bench serve"))} · {e(t.get("gpus"))} GPU(s)</span></h3>')
        out.append(f'<p class="small muted">{e(res.get("dataset_desc", ""))}. Independent requests, no shared prompt or history. '
                   f"Highlighted row = best load that met the latency targets"
                   f'{"" if st["ok"] else " (none did; this is the lowest-latency load)"}. '
                   f'Hosted price: {e(st.get("api_name") or "none set")}.</p>')
        srows, hi = [], None
        for i, l in enumerate(res["levels"]):
            if l["concurrency"] == st["op"]:
                hi = i
            n = max(1, l.get("completed") or 1)
            g = (l.get("gpu") or {}).get("per_gpu") or {}
            gutil = sum(v["util_mean"] for v in g.values()) / len(g) if g else None
            out_tok = l.get("total_output_tokens") or 0
            gs_m = l["gpu_s"] / out_tok * 1e6 if out_tok else None
            api_m = l["api_usd"] / out_tok * 1e6 if out_tok and l.get("api_usd") else None
            srows.append([e(l["concurrency"]), num(l.get("request_throughput"), 2), num(l.get("output_throughput"), 0),
                          num(l.get("total_token_throughput"), 0),
                          f'{(l.get("total_input_tokens") or 0) / n:,.0f} / {out_tok / n:,.0f}',
                          num((l["ttft_ms"].get("p50") or 0) / 1000, 2), num((l["ttft_ms"].get("p95") or 0) / 1000, 2),
                          num(l["tpot_ms"].get("p95"), 1), num(gutil, 0, suf="%"),
                          D(gs_m), ratio(gs_m, api_m), e(l.get("failed", 0))])
        out.append(table(["Requests at once", "Req/s", "Out tok/s", "Total tok/s", "Mean in / out tokens", "TTFT p50 s",
                          "TTFT p95 s", "TPOT p95 ms", "GPU util", "$ per 1M output", "vs hosted", "Failed"],
                         srows, numeric=set(range(12)), highlight=hi))
    tv = R.get("tts_vllm")
    if tv and tv.get("levels"):
        out.append('<h3>TTS · Indic-Speak speech LM on vLLM <span class="muted small">· Vocos decoder not included</span></h3>')
        out.append(table(["Streams", "Audio s per GPU-s", "Stream speed p5 (× real time)", "Stream speed p50", "First audio p95 s",
                          "Ran to max tokens", "Errors"],
                         [[e(l["concurrency"]), num(l["audio_s_per_s"], 1), num(l["stream_speed_x_realtime"].get("p5"), 2),
                           num(l["stream_speed_x_realtime"].get("p50"), 2), num(l["first_4_frames_s"].get("p95"), 2),
                           num((l.get("share_hit_max_tokens") or 0) * 100, 0, suf="%"), e(l["errors"])] for l in tv["levels"]],
                         numeric=set(range(7))))
    th = R.get("tts_hf")
    if th:
        out.append("<h3>TTS · end to end, one stream</h3>")
        out.append(table(["Clips", "Audio total s", "RTFx", "RTF p50", "Latency p95 s", "Chars per audio s", "Peak mem GB"],
                         [[e(th["items"]), num(th["audio_s_total"], 0), num(th["rtfx_single_stream"], 1), num(th["rtf"].get("p50"), 2),
                           num(th["latency_s"].get("p95"), 2), num(th["chars_per_audio_s"], 1), num(th.get("peak_mem_gib"), 1)]],
                         numeric=set(range(7))))
    asr = R.get("asr")
    if asr:
        lat = asr["latency_batch1_s"]
        out.append(f'<h3>ASR · Indic-Transcribe-core <span class="muted small">· {e(asr["backend"])} backend, '
                   f'~{asr.get("utterance_s", 20):.0f} s utterances</span></h3>')
        out.append(f'<p class="small">Batch-1 latency p50 {num(lat.get("p50"), 2)} s, p95 {num(lat.get("p95"), 2)} s · '
                   f'WER {num((asr["wer"] or 0) * 100 if asr.get("wer") is not None else None, 1, suf="%")}, '
                   f'CER {num((asr["cer"] or 0) * 100 if asr.get("cer") is not None else None, 1, suf="%")} · '
                   f'peak memory {num(asr.get("peak_mem_gib"), 1)} GB</p>')
        out.append(table(["Batch size", "Audio min", "Wall s", "RTFx"],
                         [[e(l["batch_size"]), num(l["audio_s"] / 60, 1), num(l["wall_s"], 1), num(l["rtfx"], 0)] for l in asr["throughput"]],
                         numeric={0, 1, 2, 3}))
    ocr = R.get("ocr")
    if ocr and ocr.get("levels"):
        out.append(f'<h3>OCR · IndicOCR <span class="muted small">· {e(ocr.get("n_inputs", "?"))} input pages</span></h3>')
        out.append(table(["Model copies", "Pages/s", "Latency p50 s", "Latency p95 s", "Blocks/page", "Errors"],
                         [[e(l["workers"]), num(l["pages_per_s"], 2), num(l["latency_s"].get("p50"), 2), num(l["latency_s"].get("p95"), 2),
                           num(l.get("blocks_per_page"), 1), e(l["errors"])] for l in ocr["levels"]],
                         numeric=set(range(6))))

    out.append("<h2>Assumptions</h2><ul>")
    out.append(f"<li>Per student-day: {day['asr_minutes']} min ASR, {day['tts_minutes']} min TTS, {day['ocr_pages']} OCR pages; "
               f"LLM {lv['cached_input_tokens'] / 1e6:.1f}M cached + {lv['uncached_input_tokens'] / 1e3:.0f}K input + "
               f"{lv['output_tokens'] / 1e3:.0f}K output tokens.</li>")
    others = ", ".join("{} ${:.2f}".format(v["label"], v["usd_per_gpu_hour"]) for v in pricing.get("all", {}).values())
    out.append(f"<li>GPU cost {P} per GPU-hour on {e(pricing['label'])} ({e(pricing['source'])}). "
               f"Machine prices in spec.yaml: {e(others)}.</li>")
    out.append(f"<li>Latency targets: LLM TTFT p95 ≤ {slo['llm_ttft_p95_s']} s and TPOT p95 ≤ {slo['llm_tpot_p95_ms']} ms; "
               f"TTS streams ≥ {slo['tts_realtime_margin']}× real time; ASR p95 ≤ {slo['asr_latency_p95_s']} s; "
               f"OCR page p95 ≤ {slo['ocr_page_latency_p95_s']} s.</li>")
    out.append(f"<li>{bill['days_per_month']} days/month; ₹{bill['inr_per_usd']} per US$ for Bodhan's hosted prices.</li></ul>")
    out.append("<h2>Notes</h2><ul>" + "".join(f"<li>{e(n)}</li>" for n in notes)
               + "".join(f"<li>Not costed yet: {e(n)}</li>" for n in spec.get("not_costed_yet", [])) + "</ul>")
    out.append(f'<p class="small muted">Measured on {e(pricing["label"])}: {e(gpus)}.</p>')
    chart_data = {"items": citems, "target": target, "totalLabel": total_label}
    out.append("<script>" + JS.replace("__P0__", repr(float(price))).replace("__CHART__", _json.dumps(chart_data)) + "</script>")
    out.append("</main></body></html>")
    return "\n".join(out)
