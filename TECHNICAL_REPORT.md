# Local Multi-Product AI Sales Agent — Technical Report

## 1. Architecture and zero-cost compliance

### Scope

This report covers the **Local Multi-Product AI Studio** and the
`brain.provider: ollama` configuration. Sellbird also contains explicit,
opt-in Claude and Codex providers for the broader outreach product; those
providers are not used by the local studio and are not part of this
zero-cloud acceptance claim.

### Inference path

| Layer | Implementation | Local contract |
|---|---|---|
| Brain provider | `sellbird/brain.py` | `httpx.AsyncClient` POSTs to `http://localhost:11434/api/chat` |
| Brain defaults | `sellbird/config.py` | Provider `ollama`, model `qwen3:8b`, base URL `http://localhost:11434` |
| Multi-product engine | `sellbird/multi_product.py` | `_local_llm_json()` POSTs to `<ollama_url>/api/chat` |
| Studio routes | `sellbird/studio_routes.py` | Calls `MultiProductSalesEngine`; never calls a cloud model API |
| Embedding score | `sellbird/hardware.py` | `all-MiniLM-L6-v2` through local Sentence Transformers |
| Persistence | SQLite | Products, prospects, logs, and queued tasks stay in `data/multi_product.db` |

The eight studio workflows are:

1. Business profile generation
2. Product-specific ICP generation
3. Prospect research
4. Buying-signal classification
5. Problem-statement generation
6. Hybrid semantic plus LLM fit scoring
7. Personalised email generation
8. Reply classification

Every LLM workflow sends its prompt to the local Ollama `/api/chat` endpoint.
Workflow 6 computes the semantic component locally with
`all-MiniLM-L6-v2`, then sends the local evidence and score to Ollama for the
reasoning component. There is no cloud fallback branch: an Ollama exception is
reported and the multi-product engine queues the task instead.

### Cost statement

- Token/API cost for local inference: **$0.00**
- Cloud fallback: **disabled**
- Ollama endpoint: `http://localhost:11434/api/chat`
- Default model: **`qwen3:8b`**
- Internet requirement for inference: **none after model and embedding assets are installed**

Local model downloads and package installation may use the network once. That
is installation traffic, not hosted inference or per-token billing.

## 2. Exact open-source licence table

The table names the exact licence identified by the upstream project or model
card. The source links are the verification references; downstream users
should retain the corresponding notices when redistributing software or model
artifacts.

| Component | Use here | Exact licence | Verification source |
|---|---|---|---|
| Qwen3 (`qwen3:8b`, Qwen3-8B weights) | Local Ollama LLM | Apache License 2.0 | [Qwen3-8B LICENSE](https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE) |
| `sentence-transformers` package and `all-MiniLM-L6-v2` model | Local semantic similarity | Apache License 2.0 for both the package and the model | [sentence-transformers LICENSE](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE); [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/README.md) |
| FastAPI | HTTP API and router framework | MIT License | [FastAPI LICENSE](https://github.com/fastapi/fastapi/blob/master/LICENSE) |
| SQLite | Local persistence | Public domain | [SQLite copyright page](https://www.sqlite.org/copyright.html) |
| psutil | CPU, RAM, and process hardware telemetry | BSD 3-Clause “New” or “Revised” License | [psutil LICENSE](https://github.com/giampaolo/psutil/blob/master/LICENSE) |
| httpx | Async local HTTP transport | BSD 3-Clause “New” or “Revised” License | [HTTPX LICENSE](https://github.com/encode/httpx/blob/master/LICENSE.md) |
| Scrapy | External web-research ecosystem reference; not required by the local studio | BSD 3-Clause “New” or “Revised” License | [Scrapy LICENSE](https://github.com/scrapy/scrapy/blob/master/LICENSE) |
| Beautiful Soup (`beautifulsoup4`) | HTML parsing for web research | MIT License | [Beautiful Soup licence documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/#license) |
| Playwright | Browser automation for web research | Apache License 2.0 | [Playwright LICENSE](https://github.com/microsoft/playwright/blob/main/LICENSE) |

This table distinguishes the Qwen model licence from the Python client
licences. An Ollama distribution or a model bundle may also include notices
for its own packaging and dependencies; those notices remain applicable.

## 3. Hardware detection and model selection

`sellbird/hardware.py` uses `psutil` for memory and CPU counts and probes
`nvidia-smi` locally for NVIDIA GPU name and VRAM. It does not call a remote
hardware service.

| Profile | Detection rule | Recommended Ollama model | Fallback | Context |
|---|---|---|---|---|
| Lightweight laptop | Less than 12 GB RAM | `qwen3:4b` | `qwen3:1.7b` | Standard embedding model; CPU-friendly |
| Standard CPU | At least 12 GB RAM, no detected GPU | `qwen3:8b` | `qwen3:4b` | 4,096-token context |
| Standard GPU | At least 12 GB RAM with detected GPU | `qwen3:8b` | `qwen3:4b` | 4,096-token context |
| High-performance workstation | At least 28 GB RAM and either no GPU requirement or at least 8 GB GPU VRAM | `qwen3:14b` | `qwen3:8b` | 8,192-token context |

The exported hardware profile also reports the standard laptop and
high-performance workstation reference configurations. The studio defaults
to `qwen3:8b` for reproducibility; hardware telemetry is a recommendation,
not an automatic cloud or model-provider switch.

## 4. Failure handling, task queueing, and system boundaries

### Ollama offline

`MultiProductSalesEngine._local_llm_json()` catches connection, timeout, HTTP,
and malformed-response failures. It then:

1. Inserts the requested workflow into `queued_tasks` with its product,
   prospect, payload, and error message.
2. Inserts a `queued_on_failure` row into `inference_logs`.
3. Raises an explicit error identifying the queued task ID.
4. Never retries through Claude, Codex, OpenAI, or another hosted API.

The Brain Ollama provider similarly calls `_mark_unavailable()` with the
endpoint, model, original failure, and explicit “No cloud API fallback”
wording. Its local budget gate closes while Ollama is unavailable. A later
successful local response clears the unavailable state.

### Offline AI inference versus external web research

These are separate paths:

- **Offline AI inference:** Ollama `/api/chat`, Sentence Transformers,
  SQLite, hardware detection, workflow orchestration, scoring, email drafting,
  and reply classification. These do not require external network access after
  local assets are installed.
- **External web research:** discovery providers, company website retrieval,
  browser automation, email verification, and other prospect-enrichment
  integrations. These may access the internet and may have provider limits or
  costs. They are not invoked by the local studio’s workflow prompts.

“Offline” therefore means the AI analysis path, not that every Sellbird
feature—including discovery and enrichment—can work without a network.

### Known limitations and simulated email disclosure

- Ollama must be installed, running, and have the selected model pulled.
- The default `qwen3:8b` model may be slow or memory-intensive on CPU-only
  laptops; use the detected fallback model when appropriate.
- The embedding model is downloaded and cached locally on first use.
- Queued tasks are persisted for visibility and later processing; this report
  does not claim that a background queue worker automatically drains every
  queued task.
- Studio email drafts are generated text. The local assignment workflow does
  not send email or prove deliverability. Any sample replies and demo
  contacts are fictional simulations, not live outreach.
- The broader Sellbird sender can send through configured Gmail, Microsoft 365,
  SMTP, or legacy integrations only when the user explicitly configures and
  approves that separate workflow.

## 5. Reproducible setup

### Windows PowerShell

```powershell
git clone https://github.com/ethanplusai/sellbird.git
Set-Location sellbird
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m playwright install chromium
ollama pull qwen3:8b
ollama serve
```

Keep `ollama serve` running in one terminal. In a second activated terminal:

```powershell
sellbird dashboard
```

Open the private URL printed by Sellbird and navigate to `/studio`.

### macOS/Linux

```bash
git clone https://github.com/ethanplusai/sellbird.git
cd sellbird
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m playwright install chromium
ollama pull qwen3:8b
ollama serve
```

In a second activated terminal, run `sellbird dashboard` and open `/studio`.

## 6. Offline acceptance test

The test below validates the assignment path without calling a hosted model:

1. Start Ollama locally and confirm the model is installed:

   ```bash
   ollama list
   ```

   Confirm that `qwen3:8b` appears.

2. Start Sellbird:

   ```bash
   sellbird dashboard
   ```

3. Open the printed private dashboard URL, then open `/studio`.

4. Confirm the page reports:
   - Hardware CPU, RAM, GPU, and recommended model.
   - Ollama online with `qwen3:8b`.
   - `$0.00` inference cost and cloud fallback disabled.

5. Add a fictional product and local prospect dossier through the forms.
   Use reserved example domains such as `example.com` if an email field is
   required.

6. Click **Run All 8 Local AI Workflows**. Confirm that the page displays:
   business profile, product ICP, prospect research, buying signals, problem
   statement, hybrid semantic plus LLM fit score, personalised email draft,
   and reply classification.

7. Confirm the inference log table contains `ollama_local`-equivalent local
   activity, model names, durations, zero cost, and success statuses.

8. Stop Ollama, click the workflow button again, and confirm:
   - The UI reports an explicit local Ollama error.
   - No Claude, Codex, OpenAI, or other cloud request is attempted.
   - The queued-task table contains the failed workflow and error.

9. Restart Ollama and repeat the workflow after the local service is ready.
   A successful run demonstrates recovery without changing providers.

For a code-level smoke check of the Brain Ollama path:

```bash
python -m py_compile sellbird/brain.py
```

The acceptance test is complete only when both the online success path and
the offline queue path have been observed.
