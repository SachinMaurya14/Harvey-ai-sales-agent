"""Local multi-product AI studio routes and self-contained web UI."""

from __future__ import annotations

import html
import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from sellbird.hardware import detect_hardware
from sellbird.multi_product import MultiProductSalesEngine

router = APIRouter()


def _engine() -> MultiProductSalesEngine:
    return MultiProductSalesEngine()


async def _body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"success": False, "error": message}, status_code=status_code)


def _required(body: dict[str, Any], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _id(body: dict[str, Any], key: str) -> int:
    value = body.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


@router.get("/studio", response_class=HTMLResponse)
async def studio() -> str:
    return STUDIO_HTML


@router.get("/api/studio/hardware")
async def studio_hardware():
    return detect_hardware().to_dict()


@router.get("/api/studio/health")
async def studio_health():
    return await _engine().check_ollama_health()


@router.get("/api/studio/products")
async def studio_products():
    return {"products": _engine().list_products()}


@router.post("/api/studio/products")
async def studio_add_product(request: Request):
    body = await _body(request)
    try:
        product_id = _engine().create_product(
            _required(body, "name"),
            _required(body, "company_name"),
            _required(body, "raw_company_info"),
            _required(body, "product_description"),
            str(body.get("pricing") or "").strip(),
        )
    except ValueError as exc:
        return _error(str(exc))
    return {"success": True, "product": _engine().get_product(product_id)}


@router.get("/api/studio/prospects")
async def studio_prospects(product_id: int | None = None):
    return {"prospects": _engine().list_prospects(product_id)}


@router.post("/api/studio/prospects")
async def studio_add_prospect(request: Request):
    body = await _body(request)
    try:
        engine = _engine()
        prospect_id = engine.add_prospect(
            _id(body, "product_id"),
            _required(body, "company_name"),
            _required(body, "contact_name"),
            _required(body, "contact_title"),
            _required(body, "contact_email"),
            _required(body, "industry"),
            _required(body, "local_data"),
            str(body.get("sample_reply") or "").strip(),
        )
        prospect = engine.get_prospect(prospect_id)
    except ValueError as exc:
        return _error(str(exc))
    return {"success": True, "prospect": prospect}


@router.post("/api/studio/run")
async def studio_run(request: Request):
    body = await _body(request)
    try:
        product_id = _id(body, "product_id")
        engine = _engine()
        engine.get_product(product_id)
        result = await engine.run_full_product_pipeline(product_id)
    except ValueError as exc:
        return _error(str(exc), 404)
    except RuntimeError as exc:
        return _error(str(exc), 503)
    return {"success": True, "result": result}


@router.get("/api/studio/results")
async def studio_results(prospect_id: int):
    try:
        return {"prospect": _engine().get_prospect(prospect_id)}
    except ValueError as exc:
        return _error(str(exc), 404)


@router.get("/api/studio/logs")
async def studio_logs(limit: int = 50):
    return {"logs": _engine().list_logs(max(1, min(limit, 200)))}


@router.get("/api/studio/queue")
async def studio_queue():
    return {"tasks": _engine().list_queue()}


STUDIO_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sellbird Local AI Studio</title>
<script src="/static/auth.js"></script>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#121a2d;--line:#263451;--text:#e9eefc;--muted:#9ba9c7;--accent:#68e0b0;--warn:#ffcc66;--bad:#ff8b8b}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#0b1020,#111b31);color:var(--text);font:14px system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1400px;margin:auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{font-size:18px;margin:0 0 14px}.sub,.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin:20px 0}
.panel{background:rgba(18,26,45,.92);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 12px 30px #0002}.wide{grid-column:1/-1}
.metric{font-size:22px;font-weight:700;margin:5px 0}.ok{color:var(--accent)}.warn{color:var(--warn)}.bad{color:var(--bad)}.badge{border:1px solid currentColor;border-radius:20px;padding:4px 9px;display:inline-block}
form{display:grid;gap:9px}label{color:var(--muted);font-size:12px}input,textarea,select,button{font:inherit;border-radius:8px;border:1px solid var(--line);padding:9px;background:#0d1527;color:var(--text)}textarea{min-height:70px;resize:vertical}button{background:#1b8065;border-color:#2ab58d;cursor:pointer;font-weight:650}button:hover{filter:brightness(1.15)}.secondary{background:#1a2945;border-color:var(--line)}
.row{display:flex;gap:10px;align-items:end;flex-wrap:wrap}.row>*{flex:1}.row button{flex:0 0 auto}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted)}pre{white-space:pre-wrap;word-break:break-word;margin:0;max-height:300px;overflow:auto}.workflow{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}.result{border:1px solid var(--line);border-radius:9px;padding:11px}.result h3{margin:0 0 7px;font-size:13px;color:var(--accent)}#notice{min-height:20px;color:var(--warn)}
</style>
</head>
<body><main>
<h1>Local AI Studio</h1><div class="sub">Multi-product prospecting with Ollama only. No cloud fallback. Inference cost: <b>$0.00</b>.</div>
<div class="grid">
<section class="panel"><h2>Hardware</h2><div id="hardware">Loading...</div></section>
<section class="panel"><h2>Ollama</h2><div id="health">Checking local server...</div><p class="muted">Cloud fallback: disabled</p></section>
<section class="panel"><h2>Products</h2><select id="product"></select><div id="productInfo" class="muted"></div></section>
</div>
<div class="grid">
<section class="panel"><h2>Add Product</h2><form id="productForm">
<input name="name" placeholder="Product name" required><input name="company_name" placeholder="Company name" required>
<textarea name="raw_company_info" placeholder="Company information stored locally" required></textarea>
<textarea name="product_description" placeholder="Product description" required></textarea><input name="pricing" placeholder="Pricing (optional)">
<button>Add Product</button></form></section>
<section class="panel"><h2>Load Local Prospect</h2><form id="prospectForm">
<input name="company_name" placeholder="Prospect company" required><input name="contact_name" placeholder="Contact name" required>
<input name="contact_title" placeholder="Contact title" required><input name="contact_email" type="email" placeholder="Contact email" required>
<input name="industry" placeholder="Industry" required><textarea name="local_data" placeholder="Downloaded local dossier / notes" required></textarea>
<textarea name="sample_reply" placeholder="Sample reply (optional)"></textarea><button>Add Prospect</button></form></section>
</div>
<section class="panel"><div class="row"><div><h2>Run the 8 local AI workflows</h2><div class="muted">Business profile → ICP → research → signals → problem → hybrid fit → email → reply.</div></div><button id="run">Run All 8 Local AI Workflows</button></div><div id="notice"></div><div id="results" class="workflow"></div></section>
<div class="grid"><section class="panel wide"><h2>Traceable local inference logs</h2><div id="logs"></div></section><section class="panel wide"><h2>Queued tasks (Ollama offline)</h2><div id="queue"></div></section></div>
</main>
<script>
const $=s=>document.querySelector(s), esc=s=>{const d=document.createElement('div');d.textContent=String(s??'');return d.innerHTML};
const json=v=>typeof v==='string'?v:JSON.stringify(v,null,2);
async function api(url,opt){const r=await authFetch(url,opt);const d=await r.json();if(!r.ok)throw Error(d.error||'Request failed');return d}
function formData(form){return Object.fromEntries(new FormData(form).entries())}
async function refreshProducts(){const d=await api('/api/studio/products');const p=$('#product');p.innerHTML=d.products.map(x=>`<option value="${x.id}">${esc(x.name)} — ${esc(x.company_name)}</option>`).join('')||'<option value="">Add a product first</option>';p.onchange=()=>{$('#productInfo').textContent=d.products.find(x=>x.id==p.value)?.product_description||''};p.onchange()}
async function refresh(){try{const [h,o,l,q]=await Promise.all([api('/api/studio/hardware'),api('/api/studio/health'),api('/api/studio/logs'),api('/api/studio/queue')]);$('#hardware').innerHTML=`<div class="metric">${esc(h.recommended_model)}</div><div>${esc(h.cpu_model)} · ${h.physical_cores} cores / ${h.logical_threads} threads</div><div>${h.total_ram_gb} GB RAM (${h.available_ram_gb} GB free)</div><div>${h.gpu_detected?'GPU: '+esc(h.gpu_name)+' ('+h.gpu_vram_gb+' GB)':'CPU / integrated graphics'}</div>`;$('#health').innerHTML=`<span class="badge ${o.available?'ok':'bad'}">${o.available?'Online · '+esc(o.configured_model):'Offline'}</span><div class="muted">${esc(o.error||o.endpoint)}</div>`;$('#logs').innerHTML=table(l.logs,['timestamp','workflow','provider','model','duration_ms','cost_usd','status','output_preview']);$('#queue').innerHTML=table(q.tasks,['id','created_at','workflow','status','last_error'])}catch(e){$('#notice').textContent=e.message}}
function table(rows,cols){if(!rows.length)return '<p class="muted">None yet.</p>';return '<table><thead><tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>`<td>${esc(r[c])}</td>`).join('')+'</tr>').join('')+'</tbody></table>'}
function showResults(result){let items=[['Business Profile',result.business_profile],['Product ICP',result.icp]];for(const p of result.prospects||[]){const prefix=p.company?`${p.company} · `:'';items.push([prefix+'Prospect Research',p.research],[prefix+'Buying Signals',p.signal],[prefix+'Problem Statement',p.problem],[prefix+'Hybrid Semantic + LLM Fit Score',p.score],[prefix+'Personalised Email Draft',p.email],[prefix+'Reply Classification',p.reply_classification])}$('#results').innerHTML=items.map(x=>`<article class="result"><h3>${esc(x[0])}</h3><pre>${esc(json(x[1]||{}))}</pre></article>`).join('')}
$('#productForm').onsubmit=async e=>{e.preventDefault();try{await api('/api/studio/products',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(formData(e.target))});e.target.reset();await refreshProducts();$('#notice').textContent='Product saved.'}catch(x){$('#notice').textContent=x.message}}
$('#prospectForm').onsubmit=async e=>{e.preventDefault();try{const b=formData(e.target);b.product_id=Number($('#product').value);await api('/api/studio/prospects',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});e.target.reset();$('#notice').textContent='Prospect loaded locally.'}catch(x){$('#notice').textContent=x.message}}
$('#run').onclick=async()=>{if(!$('#product').value)return;$('#notice').textContent='Running locally...';try{const d=await api('/api/studio/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({product_id:Number($('#product').value)})});showResults(d.result);$('#notice').textContent='Completed with local inference only.'}catch(x){$('#notice').textContent=x.message}finally{refresh()}}
refreshProducts().then(refresh);setInterval(refresh,5000);
</script></body></html>"""
