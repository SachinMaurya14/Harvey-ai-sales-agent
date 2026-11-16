"""Multi-Product Local AI Sales Engine (100% Offline-Capable via Ollama + SentenceTransformers)."""

import asyncio
import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from sellbird.hardware import detect_hardware, semantic_similarity
from sellbird.paths import PROJECT_ROOT

logger = logging.getLogger("sellbird.multi_product")

DB_PATH = PROJECT_ROOT / "data" / "multi_product.db"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_json(text: str) -> dict | list | None:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


class MultiProductSalesEngine:
    """SQLite-backed multi-product sales agent orchestrated on local Ollama + embeddings."""

    def __init__(
        self,
        db_path: Path = DB_PATH,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    company_name TEXT NOT NULL,
                    raw_company_info TEXT NOT NULL,
                    product_description TEXT NOT NULL,
                    pricing TEXT DEFAULT '',
                    business_profile_json TEXT DEFAULT '{}',
                    icp_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prospects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id INTEGER NOT NULL,
                    company_name TEXT NOT NULL,
                    contact_name TEXT NOT NULL,
                    contact_title TEXT NOT NULL,
                    contact_email TEXT NOT NULL,
                    industry TEXT NOT NULL,
                    local_data TEXT NOT NULL,
                    research_json TEXT DEFAULT '{}',
                    signal_json TEXT DEFAULT '{}',
                    problem_statement TEXT DEFAULT '',
                    semantic_score REAL DEFAULT 0.0,
                    fit_score INTEGER DEFAULT 0,
                    priority_tier TEXT DEFAULT 'Unscored',
                    scoring_reason TEXT DEFAULT '',
                    email_subject TEXT DEFAULT '',
                    email_body TEXT DEFAULT '',
                    sample_reply TEXT DEFAULT '',
                    reply_classification_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(product_id) REFERENCES products(id)
                );

                CREATE TABLE IF NOT EXISTS inference_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    workflow TEXT NOT NULL,
                    product_id INTEGER,
                    prospect_id INTEGER,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0.0,
                    status TEXT NOT NULL,
                    output_preview TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS queued_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    workflow TEXT NOT NULL,
                    product_id INTEGER,
                    prospect_id INTEGER,
                    payload_json TEXT NOT NULL,
                    status TEXT DEFAULT 'queued',
                    last_error TEXT DEFAULT ''
                );
                """
            )

    # ── Health, Hardware & Failure Handling ────────────────────────────────

    async def check_ollama_health(self) -> dict[str, Any]:
        """Verify local Ollama server status without any cloud fallback."""
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.get(f"{self.ollama_url}/api/tags")
                resp.raise_for_status()
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                model_ready = any(self.model in m for m in models)
                return {
                    "available": model_ready,
                    "endpoint": self.ollama_url,
                    "configured_model": self.model,
                    "installed_models": models,
                    "cloud_fallback": False,
                    "error": "" if model_ready else f"Model '{self.model}' not pulled locally.",
                }
        except Exception as e:
            return {
                "available": False,
                "endpoint": self.ollama_url,
                "configured_model": self.model,
                "installed_models": [],
                "cloud_fallback": False,
                "error": f"Ollama unreachable at {self.ollama_url}: {e}. Cloud API fallback is strictly disabled.",
            }

    def _enqueue_task(
        self,
        workflow: str,
        product_id: int | None,
        prospect_id: int | None,
        payload: dict,
        error_msg: str,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO queued_tasks (created_at, workflow, product_id, prospect_id, payload_json, status, last_error)
                VALUES (?, ?, ?, ?, ?, 'queued', ?)
                """,
                (_now_iso(), workflow, product_id, prospect_id, json.dumps(payload), error_msg[:500]),
            )
            return int(cur.lastrowid)

    async def _local_llm_json(
        self,
        workflow: str,
        prompt: str,
        *,
        product_id: int | None = None,
        prospect_id: int | None = None,
        queue_payload: dict | None = None,
        timeout: float = 180.0,
    ) -> dict:
        """Execute 100% local Ollama inference. Queues task on outage instead of cloud fallback."""
        url = f"{self.ollama_url}/api/chat"
        req_body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise B2B AI sales analyst. Output ONLY valid JSON with no markdown fences.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }

        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=req_body)
                resp.raise_for_status()
                data = resp.json()

            duration_ms = int((time.monotonic() - start) * 1000)
            raw_text = (data.get("message") or {}).get("content", "").strip()
            parsed = _extract_json(raw_text)
            if not isinstance(parsed, dict):
                parsed = {"raw_output": raw_text}

            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO inference_logs
                    (timestamp, workflow, product_id, prospect_id, provider, model, endpoint,
                     duration_ms, prompt_tokens, completion_tokens, cost_usd, status, output_preview)
                    VALUES (?, ?, ?, ?, 'ollama_local', ?, ?, ?, ?, ?, 0.0, 'success', ?)
                    """,
                    (
                        _now_iso(),
                        workflow,
                        product_id,
                        prospect_id,
                        self.model,
                        url,
                        duration_ms,
                        int(data.get("prompt_eval_count") or 0),
                        int(data.get("eval_count") or 0),
                        raw_text[:280],
                    ),
                )
            return parsed

        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            err_msg = f"Local Ollama unavailable ({type(exc).__name__}: {exc}). No cloud API fallback permitted."
            task_id = self._enqueue_task(
                workflow, product_id, prospect_id, queue_payload or {"workflow": workflow}, err_msg
            )
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO inference_logs
                    (timestamp, workflow, product_id, prospect_id, provider, model, endpoint,
                     duration_ms, cost_usd, status, output_preview)
                    VALUES (?, ?, ?, ?, 'ollama_local', ?, ?, ?, 0.0, 'queued_on_failure', ?)
                    """,
                    (_now_iso(), workflow, product_id, prospect_id, self.model, url, duration_ms, err_msg[:280]),
                )
            raise RuntimeError(f"{err_msg} Task queued as ID #{task_id} for later processing.") from exc

    # ── Product & Prospect CRUD ────────────────────────────────────────────

    def create_product(
        self,
        name: str,
        company_name: str,
        raw_company_info: str,
        product_description: str,
        pricing: str = "",
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO products (name, company_name, raw_company_info, product_description, pricing, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    company_name=excluded.company_name,
                    raw_company_info=excluded.raw_company_info,
                    product_description=excluded.product_description,
                    pricing=excluded.pricing
                """,
                (name.strip(), company_name.strip(), raw_company_info.strip(), product_description.strip(), pricing.strip(), _now_iso()),
            )
            row = conn.execute("SELECT id FROM products WHERE name = ?", (name.strip(),)).fetchone()
            return int(row["id"])

    def add_prospect(
        self,
        product_id: int,
        company_name: str,
        contact_name: str,
        contact_title: str,
        contact_email: str,
        industry: str,
        local_data: str,
        sample_reply: str = "",
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO prospects
                (product_id, company_name, contact_name, contact_title, contact_email, industry, local_data, sample_reply, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product_id,
                    company_name.strip(),
                    contact_name.strip(),
                    contact_title.strip(),
                    contact_email.strip(),
                    industry.strip(),
                    local_data.strip(),
                    sample_reply.strip(),
                    _now_iso(),
                ),
            )
            return int(cur.lastrowid)

    def get_product(self, product_id: int) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
            if not row:
                raise ValueError(f"Product {product_id} not found")
            return dict(row)

    def get_prospect(self, prospect_id: int) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM prospects WHERE id = ?", (prospect_id,)).fetchone()
            if not row:
                raise ValueError(f"Prospect {prospect_id} not found")
            return dict(row)

    def list_products(self) -> list[dict]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM products ORDER BY id DESC").fetchall()]

    def list_prospects(self, product_id: int | None = None) -> list[dict]:
        with self._conn() as conn:
            if product_id is not None:
                rows = conn.execute(
                    "SELECT * FROM prospects WHERE product_id = ? ORDER BY fit_score DESC, id ASC",
                    (product_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM prospects ORDER BY fit_score DESC, id ASC").fetchall()
            return [dict(r) for r in rows]

    def list_logs(self, limit: int = 30) -> list[dict]:
        with self._conn() as conn:
            return [
                dict(r)
                for r in conn.execute("SELECT * FROM inference_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            ]

    def list_queue(self) -> list[dict]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM queued_tasks ORDER BY id DESC").fetchall()]

    # ── The 8 Required Offline-Capable Local AI Workflows ──────────────────

    async def wf1_generate_business_profile(self, product_id: int) -> dict:
        """1. Business profile generation from locally stored company information."""
        prod = self.get_product(product_id)
        prompt = (
            "Analyze this locally stored company information and generate a structured B2B business profile.\n"
            f"Company Name: {prod['company_name']}\n"
            f"Raw Company Notes: {prod['raw_company_info']}\n\n"
            'Return JSON with keys: "company_name", "core_mission", "value_propositions" (list of strings), '
            '"target_segments" (list of strings), "key_differentiators" (list of strings).'
        )
        res = await self._local_llm_json(
            "1_business_profile_generation",
            prompt,
            product_id=product_id,
            queue_payload={"workflow": "wf1_generate_business_profile", "product_id": product_id},
        )
        with self._conn() as conn:
            conn.execute("UPDATE products SET business_profile_json = ? WHERE id = ?", (json.dumps(res), product_id))
        return res

    async def wf2_generate_product_icp(self, product_id: int) -> dict:
        """2. Product-specific ICP generation."""
        prod = self.get_product(product_id)
        prompt = (
            "Generate a product-specific Ideal Customer Profile (ICP) for outbound B2B sales.\n"
            f"Product Name: {prod['name']}\n"
            f"Product Description: {prod['product_description']}\n"
            f"Pricing: {prod['pricing']}\n\n"
            'Return JSON with keys: "target_industries" (list), "company_size", "buyer_personas" (list of job titles), '
            '"core_pain_points" (list), "buying_triggers" (list).'
        )
        res = await self._local_llm_json(
            "2_product_icp_generation",
            prompt,
            product_id=product_id,
            queue_payload={"workflow": "wf2_generate_product_icp", "product_id": product_id},
        )
        with self._conn() as conn:
            conn.execute("UPDATE products SET icp_json = ? WHERE id = ?", (json.dumps(res), product_id))
        return res

    async def wf3_research_prospect(self, prospect_id: int) -> dict:
        """3. Prospect research from previously downloaded local data."""
        p = self.get_prospect(prospect_id)
        prompt = (
            "Extract structured sales intelligence from this locally stored prospect dossier.\n"
            f"Prospect Company: {p['company_name']}\n"
            f"Industry: {p['industry']}\n"
            f"Contact: {p['contact_name']} ({p['contact_title']})\n"
            f"Locally Downloaded Data: {p['local_data']}\n\n"
            'Return JSON with keys: "company_summary", "tech_or_ops_context", "observed_challenges" (list), '
            '"personalization_hooks" (list).'
        )
        res = await self._local_llm_json(
            "3_prospect_research",
            prompt,
            product_id=p["product_id"],
            prospect_id=prospect_id,
            queue_payload={"workflow": "wf3_research_prospect", "prospect_id": prospect_id},
        )
        with self._conn() as conn:
            conn.execute("UPDATE prospects SET research_json = ? WHERE id = ?", (json.dumps(res), prospect_id))
        return res

    async def wf4_classify_buying_signal(self, prospect_id: int) -> dict:
        """4. Buying-signal classification."""
        p = self.get_prospect(prospect_id)
        prod = self.get_product(p["product_id"])
        prompt = (
            "Classify the buying intent signals in the prospect's local data relative to our product.\n"
            f"Product: {prod['name']} — {prod['product_description']}\n"
            f"Prospect: {p['company_name']} | Data: {p['local_data']}\n\n"
            'Return JSON with keys: "signal_detected" (boolean), "signal_category" (e.g., hiring_surge, tech_migration, '
            'compliance_pressure, expansion, none), "urgency" ("high", "medium", or "low"), "evidence_quote".'
        )
        res = await self._local_llm_json(
            "4_buying_signal_classification",
            prompt,
            product_id=p["product_id"],
            prospect_id=prospect_id,
            queue_payload={"workflow": "wf4_classify_buying_signal", "prospect_id": prospect_id},
        )
        with self._conn() as conn:
            conn.execute("UPDATE prospects SET signal_json = ? WHERE id = ?", (json.dumps(res), prospect_id))
        return res

    async def wf5_generate_problem_statement(self, prospect_id: int) -> dict:
        """5. Problem statement generation."""
        p = self.get_prospect(prospect_id)
        prod = self.get_product(p["product_id"])
        prompt = (
            "Write a sharp, prospect-specific business problem statement connecting their situation to our product.\n"
            f"Product: {prod['name']} ({prod['product_description']})\n"
            f"Prospect Company: {p['company_name']} | Contact Role: {p['contact_title']}\n"
            f"Prospect Local Data: {p['local_data']}\n\n"
            'Return JSON with keys: "problem_statement" (2-3 concise sentences), "business_impact", "recommended_angle".'
        )
        res = await self._local_llm_json(
            "5_problem_statement_generation",
            prompt,
            product_id=p["product_id"],
            prospect_id=prospect_id,
            queue_payload={"workflow": "wf5_generate_problem_statement", "prospect_id": prospect_id},
        )
        stmt = str(res.get("problem_statement") or json.dumps(res))
        with self._conn() as conn:
            conn.execute("UPDATE prospects SET problem_statement = ? WHERE id = ?", (stmt, prospect_id))
        return res

    async def wf6_score_prospect(self, prospect_id: int) -> dict:
        """6. Prospect scoring using local SentenceTransformers + Qwen3 reasoning."""
        p = self.get_prospect(prospect_id)
        prod = self.get_product(p["product_id"])
        sem_score = semantic_similarity(
            f"{prod['name']}: {prod['product_description']} {prod['icp_json']}",
            f"{p['company_name']} ({p['industry']}, {p['contact_title']}): {p['local_data']}",
        )
        prompt = (
            "Score this prospect's ICP fit from 0 to 100 for our product.\n"
            f"Product: {prod['name']} | ICP: {prod['icp_json']}\n"
            f"Prospect: {p['company_name']} | Title: {p['contact_title']} | Industry: {p['industry']}\n"
            f"Local Evidence: {p['local_data']}\n"
            f"Local Embedding Semantic Similarity: {sem_score}/100\n\n"
            'Return JSON with keys: "fit_score" (integer 0-100), "priority_tier" ("High", "Medium", or "Low"), '
            '"reasoning" (1-2 sentences).'
        )
        res = await self._local_llm_json(
            "6_prospect_scoring",
            prompt,
            product_id=p["product_id"],
            prospect_id=prospect_id,
            queue_payload={"workflow": "wf6_score_prospect", "prospect_id": prospect_id},
        )
        llm_score = int(res.get("fit_score") or int(sem_score))
        blended_score = max(0, min(100, int(round(0.4 * sem_score + 0.6 * llm_score))))
        tier = str(res.get("priority_tier") or ("High" if blended_score >= 75 else "Medium" if blended_score >= 50 else "Low"))
        reason = str(res.get("reasoning") or "")
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE prospects
                SET semantic_score = ?, fit_score = ?, priority_tier = ?, scoring_reason = ?
                WHERE id = ?
                """,
                (sem_score, blended_score, tier, reason, prospect_id),
            )
        return {
            "semantic_similarity_score": sem_score,
            "llm_fit_score": llm_score,
            "final_fit_score": blended_score,
            "priority_tier": tier,
            "reasoning": reason,
        }

    async def wf7_generate_personalized_email(self, prospect_id: int) -> dict:
        """7. Personalised sales email generation."""
        p = self.get_prospect(prospect_id)
        prod = self.get_product(p["product_id"])
        prompt = (
            "Draft a concise, high-converting cold outbound email (under 95 words) tailored to this prospect.\n"
            f"Product: {prod['name']} ({prod['product_description']})\n"
            f"Prospect: {p['contact_name']}, {p['contact_title']} at {p['company_name']}\n"
            f"Tailored Problem Statement: {p['problem_statement'] or p['local_data']}\n\n"
            'Return JSON with keys: "subject" (short, non-spammy), "body" (plain text email with clear soft CTA).'
        )
        res = await self._local_llm_json(
            "7_personalized_email_generation",
            prompt,
            product_id=p["product_id"],
            prospect_id=prospect_id,
            queue_payload={"workflow": "wf7_generate_personalized_email", "prospect_id": prospect_id},
        )
        subj = str(res.get("subject") or f"Quick question for {p['company_name']}")
        body = str(res.get("body") or "")
        with self._conn() as conn:
            conn.execute(
                "UPDATE prospects SET email_subject = ?, email_body = ? WHERE id = ?",
                (subj, body, prospect_id),
            )
        return {"subject": subj, "body": body}

    async def wf8_classify_reply(self, prospect_id: int, reply_text: str | None = None) -> dict:
        """8. Reply classification using sample conversations."""
        p = self.get_prospect(prospect_id)
        text = (reply_text or p["sample_reply"] or "").strip()
        if not text:
            text = "Thanks for reaching out. Can you share pricing and how this integrates with our existing workflow?"
        prompt = (
            "Classify this prospect's email reply and recommend the next sales action.\n"
            f"Prospect: {p['contact_name']} ({p['contact_title']} at {p['company_name']})\n"
            f"Reply Message: \"{text}\"\n\n"
            'Return JSON with keys: "intent" ("interested", "objection", "not_interested", "request_info", or "out_of_office"), '
            '"sentiment" ("positive", "neutral", or "negative"), "next_action", "suggested_response".'
        )
        res = await self._local_llm_json(
            "8_reply_classification",
            prompt,
            product_id=p["product_id"],
            prospect_id=prospect_id,
            queue_payload={"workflow": "wf8_classify_reply", "prospect_id": prospect_id, "reply_text": text},
        )
        with self._conn() as conn:
            conn.execute(
                "UPDATE prospects SET sample_reply = ?, reply_classification_json = ? WHERE id = ?",
                (text, json.dumps(res), prospect_id),
            )
        return res

    async def run_full_product_pipeline(self, product_id: int) -> dict:
        """Run all 8 local AI workflows end-to-end for a product and its prospects."""
        bp = await self.wf1_generate_business_profile(product_id)
        icp = await self.wf2_generate_product_icp(product_id)
        prospect_results = []
        for p in self.list_prospects(product_id):
            pid = p["id"]
            research = await self.wf3_research_prospect(pid)
            signal = await self.wf4_classify_buying_signal(pid)
            problem = await self.wf5_generate_problem_statement(pid)
            score = await self.wf6_score_prospect(pid)
            email = await self.wf7_generate_personalized_email(pid)
            reply_cls = await self.wf8_classify_reply(pid)
            prospect_results.append(
                {
                    "prospect_id": pid,
                    "company": p["company_name"],
                    "research": research,
                    "signal": signal,
                    "problem": problem,
                    "score": score,
                    "email": email,
                    "reply_classification": reply_cls,
                }
            )
        return {
            "product_id": product_id,
            "business_profile": bp,
            "icp": icp,
            "prospects": prospect_results,
        }

    def seed_demo_data(self) -> list[int]:
        """Seed 2 distinct B2B products with local prospect dossiers for multi-product testing."""
        p1_id = self.create_product(
            name="SAAHAJ Clinical AI",
            company_name="Harvey Local HealthTech",
            raw_company_info=(
                "We build on-premise clinical BioNLP and medical imaging triage software for mid-sized hospitals "
                "and diagnostic chains. All patient records stay inside hospital servers with zero cloud data transfer."
            ),
            product_description=(
                "Offline-first clinical record structuring and radiology triage assistant that reduces doctor "
                "documentation time by 45% without sending PHI to external cloud APIs."
            ),
            pricing="$499/month self-hosted hospital license",
        )
        if not self.list_prospects(p1_id):
            self.add_prospect(
                product_id=p1_id,
                company_name="MetroCare Multi-Specialty Hospital",
                contact_name="Dr. Ananya Sharma",
                contact_title="Chief Medical Information Officer",
                contact_email="ananya.sharma@metrocare-example.org",
                industry="Healthcare / Hospital",
                local_data=(
                    "Downloaded hospital press note: Expanding radiology wing from 2 to 5 MRI units; hiring 4 clinical "
                    "data coordinators; recently mandated strict zero-cloud patient privacy policy."
                ),
                sample_reply="Interesting — we strictly prohibit patient data leaving our local network. Does this run 100% on-premise?",
            )
            self.add_prospect(
                product_id=p1_id,
                company_name="Sunrise Retail Pharmacy Group",
                contact_name="Vikram Mehta",
                contact_title="Operations Head",
                contact_email="vikram@sunrisepharma-example.com",
                industry="Retail Pharmacy",
                local_data="Operates 18 retail pharmacy outlets focused on OTC medicine inventory and billing.",
                sample_reply="We only run retail pharmacy stores, not diagnostic labs or hospitals, so this isn't a fit.",
            )

        p2_id = self.create_product(
            name="Chintan EvalOps",
            company_name="Harvey EdTech Systems",
            raw_company_info=(
                "We provide automated coding assessment and technical interview evaluation software for engineering "
                "colleges and IT staffing firms, running locally on campus/office servers."
            ),
            product_description=(
                "Local AI code-review and DSA evaluation platform that grades 500+ student submissions per hour "
                "with zero per-token API fees."
            ),
            pricing="$299/month unlimited local seats",
        )
        if not self.list_prospects(p2_id):
            self.add_prospect(
                product_id=p2_id,
                company_name="NexGen Engineering Academy",
                contact_name="Prof. Rajesh Verma",
                contact_title="Dean of Placements & Training",
                contact_email="r.verma@nexgenedu-example.edu",
                industry="Higher Education / EdTech",
                local_data=(
                    "Campus placement report: 1,400 final-year CS students preparing for software placements; "
                    "currently struggling with manual code grading bottlenecks and high cloud API bills."
                ),
                sample_reply="High API bills were the exact reason we paused automated grading. Can we test a 15-minute demo next week?",
            )
        return [p1_id, p2_id]


async def _cli_smoke_test():
    hw = detect_hardware()
    print("=== HARDWARE PROFILE ===")
    print(f"CPU: {hw.cpu_model} ({hw.physical_cores}C/{hw.logical_threads}T) | RAM: {hw.total_ram_gb} GB | GPU: {hw.gpu_name} ({hw.gpu_vram_gb} GB)")
    print(f"Tier: {hw.hardware_tier} -> Recommended Model: {hw.recommended_model}\n")

    engine = MultiProductSalesEngine()
    health = await engine.check_ollama_health()
    print("=== OLLAMA LOCAL HEALTH ===")
    print(json.dumps(health, indent=2), "\n")

    product_ids = engine.seed_demo_data()
    print(f"Seeded {len(product_ids)} products. Running all 8 local AI workflows on Product #{product_ids[0]}...\n")
    result = await engine.run_full_product_pipeline(product_ids[0])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(_cli_smoke_test())