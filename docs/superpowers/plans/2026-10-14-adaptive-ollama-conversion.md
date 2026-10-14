# Adaptive Ollama Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert mixed-quality PDFs to Markdown with complete HTML tables, visible LaTeX and lossless figures using bounded Ollama requests on macOS, Ubuntu 25 and Windows 10/11.

**Architecture:** Keep the current Marker CLI and add an adaptive cloud converter that does not launch local model servers. A shared source layer preserves original page coordinates; bounded requests reconstruct table content and complete mathematical expressions. A portable Python launcher handles repository installation, credentials and result publication.

**Tech Stack:** Python 3.10+, pypdfium2, pdfplumber, Pillow, Pydantic, OpenAI-compatible Ollama Cloud API, unittest and hosted CI.

**Spec:** `docs/superpowers/specs/2026-10-14-adaptive-ollama-conversion-design.md`

## Global Constraints

- Markdown with HTML tables; no separate table JSON and no pg-vector/RAG batching work.
- Ollama Cloud; do not install local Gemma or require a local VLM server for the adaptive path.
- Keep `PDF` and `Output` beside the launcher; repository cache installs once.
- Process one logical table across all its pages and split requests within its context budget.
- Original page coordinates stay immutable; virtual layout coordinates are separate.
- API keys remain in process memory and are never printed or written to files.
- Windows 10/11, macOS, and Ubuntu 25 require concrete portability checks.

## Shared Defaults and Contracts

- Cloud endpoint `https://ollama.com/v1`, model `gemma4:31b`. Keys are supplied only through `OLLAMA_API_KEY` in the child environment.
- Request budget `cloud_context_tokens=8192`, response limit `cloud_max_output_tokens=2048`, maximum images `cloud_max_images=3`; these are conservative client budgets, not claims about the server tokenizer or context setting.
- Full vector crops use `cloud_crop_dpi=600`; native raster resources retain their original pixels. Request copies initially have a maximum long edge of 1536 pixels and may be reduced further to fit the budget. Retained resources are never replaced with these reduced copies.
- `cloud_max_attempts=3` includes the original request plus at most two repairs/network retries. Every attempt is charged against `cloud_max_requests`; `0` means no document-wide cap. Manual live verification uses `cloud_max_requests=10` and never runs inside automated tests.
- `CloudClient` owns request counting, budget fitting, response parsing and schema validation. All cloud stages share the same instance, so retries and ambiguous continuation checks cannot bypass the cap.
- The portable launcher invokes `python -m marker.scripts.convert_single` with `--converter_cls marker.converters.ollama.OllamaPdfConverter`, `--output_format markdown`, `--use_llm`, cloud settings above and cloud OpenAI service settings. It must not pass a key as an argument. The adaptive converter reads the key from the environment and does not require local Ollama.
- Preserve the existing `DISABLE_OCR=1`, `FORCE=1`, and `UPDATE_MARKER=1` controls. DISABLE_OCR disables general text OCR, but required formula transcription still occurs; missing required text becomes incomplete. The existing 30-page processing setting is compatible configuration, not new RAG batching.
- Tests use unittest discovery scoped to `tests/cloud` and `tests/html_tables`, without loading the upstream GPU/model-download conftest.

## Fixture Ownership and Verification Gates

- Task 1 creates a committed synthetic 125-row PDF fixture and deterministic generation source under `tests/cloud/fixtures/rows_125.*`; the regression uses extracted PDF cells, then a deterministic transport to reproduce the existing row collision before implementation changes.
- Task 2 creates committed synthetic PDF fixtures and generation source for a three-page continuation, split raster/text formula glyphs, a native raster figure, a scan, damaged text and unrelated tables. Use actual PDF extraction/crops in tests; model responses remain deterministic and key-free.
- Task 5 explicitly asserts that a distinctive dummy API key never enters stdout, stderr, logs, written files, or subprocess arguments. Its presence in the child environment is required and allowed.
- Each task receives a fresh independent review of spec compliance and code quality. Tasks 1 and 5 may run concurrently because their owned files are disjoint; serialize commits. Tasks 2, 3 and 4 run in order. Task 6 stays in the main session.
- macOS checks run locally. Windows and Ubuntu 25 are verified only by actual CI jobs; a mocked platform name does not meet this gate. Record pending/failed CI honestly and keep platform verification outstanding until the jobs succeed.
- A live Ollama check is a separate manual step after integration with a hidden key prompt and at most ten actual requests. Automated implementers and reviewers must not use a live key or API.

## Review Focus

- A formula split into operators, glyph images and ordinary text must become one coherent LaTeX expression (Task 3).
- A three-page table must retain original crop coordinates after successive joins (Tasks 2/3).
- An empty, truncated or malformed LLM response must not produce a success message (Tasks 1/4).
- Mixed scanned/digital pages and damaged encodings must use per-region OCR fallback without local VLM (Task 4).
- Spaces, Unicode filenames, credentials and unavailable GUI input must behave consistently across OSes (Task 5).

### Task 1: Response reliability and existing bugs

**Files:** `marker/services/openai.py`, `marker/cloud/__init__.py`, `marker/cloud/client.py`, `marker/cloud/context.py`, `marker/processors/llm/llm_table.py`, `marker/processors/llm/llm_mathblock.py`, `marker/processors/table.py`, `tests/cloud/test_client.py`, `tests/html_tables/test_html_tables.py`, `tests/cloud/fixtures/rows_125.*`.

**Interfaces:** `CloudClient.request(prompt, images, schema) -> dict`; `RecognitionFailure` for failed required requests; `fit_request(prompt, images, budget, max_output_tokens) -> list[Image]` with conservative bounded sizing.

- [ ] Write failing tests for schema repair, JSON backslashes, code fences, truncated responses, budget exhaustion, and the reproduced 125-row collision (expected unique row IDs `0..124`, two cells each).
- [ ] First run the existing-bug regressions against unchanged implementation code, confirming they fail for row collisions, duplicate math candidates and cumulative counters. Record the exact red commands/results before edits. New-module tests may separately fail on missing imports.
- [ ] Add bounded repair with precise schema errors and clear failure results. Fix cumulative row shifts and duplicate inline-math candidates; reset per-call table statistics.
- [ ] Run focused tests and commit this independently usable fix.

### Task 2: Source images, quality and logical tables

**Files:** `marker/cloud/source.py`, `marker/cloud/regions.py`, `marker/cloud/tables.py`, `marker/processors/table_geometry.py`, `tests/cloud/test_source.py`, `tests/cloud/test_tables.py`, `tests/cloud/fixtures/*` except `rows_125.*`.

**Interfaces:** `PdfSource` context manager provides page dimensions, words, native image metadata and `crop(page, bbox, dpi) -> Image`. `TablePart` preserves page/bbox/grid; `LogicalTable.parts` retains its source parts. `group_tables(parts, client) -> list[LogicalTable]` links supported continuations, using bounded visual decisions when ambiguous.

- [ ] Write failures for native-resolution figures, bad text quality, split headers, three-page continuations and unrelated neighbouring tables.
- [ ] Run tests and observe failures.
- [ ] Implement direct/cropped source extraction and quality gates; preserve source coordinates while constructing virtual layouts. Group by physical compatibility, captions and visual evidence.
- [ ] Verify resource dimensions and continuation tests; commit.

### Task 3: Visible LaTeX and table content within context

**Files:** `marker/cloud/content.py`, `marker/cloud/tables.py`, `marker/processors/table_geometry.py`, `marker/renderers/markdown.py`, `marker/output.py`, `pyproject.toml`, `uv.lock`, `tests/cloud/test_content.py`.

**Interfaces:** `recognize_region(source, region, client) -> recognized text/math/figure result`; `recognize_table(source, logical_table, client) -> HTML/resources/status`. Requests use whole-expression crops with nearby context and table windows with repeated headers.

- [ ] Add failures for a whole inequality assembled from multiple glyphs; text-layer formulas; math outside tables; a diagram in a cell; an oversized table split across windows; invalid cell spans and unresolved math.
- [ ] Run tests and confirm failures.
- [ ] Classify and recognise complete expressions rather than isolated glyphs. Produce real math content with retained audit PNGs. Preserve figures as resource links and retain PNG alpha. Build/validate HTML and stitch bounded windows into one table.
- [ ] Render table math as native MathML with exact LaTeX inside `semantics/annotation encoding="application/x-tex"`; the user confirmed this after actual Obsidian tests. Outside tables retain Markdown LaTeX delimiters. Add the pure-Python latex2mathml runtime dependency and preserve native MathML in MarkdownRenderer.
- [ ] Verify inline/block math rendering, phi/varphi distinction, intact LaTeX annotations, PNG alpha and visual examples; commit.

### Task 4: Adaptive converter, scans and honest completion

**Files:** `marker/converters/ollama.py`, `marker/scripts/convert_single.py`, `marker/cloud/regions.py`, `tests/cloud/test_converter.py`.

**Interfaces:** `OllamaPdfConverter(config, artifact_dict, processor_list, renderer, llm_service)` has `requires_model_dict=False` and returns standard `MarkdownOutput` with recognition metadata. Existing converters keep their model path.

- [ ] Test digital, scanned and mixed pages, empty pages, corrupted text, scan tables without vector boundaries, retries and partial results. Verify no local model factory is called for this converter.
- [ ] Run tests to see failures.
- [ ] Integrate source analysis, bounded cloud layout/OCR and table/math recognition in document order. Resolve overlaps and avoid duplicated content. Save partial results, but expose incomplete recognition through metadata and nonzero exit status.
- [ ] Run fixtures without weights/API keys; commit. Live Ollama verification belongs to Task 6, with the global ten-request cap.

### Task 5: Portable launcher and migration

**Files:** `ScriptOllama.py`, `ScriptMacOS-Ollama.sh`, `ScriptOllama.sh`, `ScriptOllama.ps1`, `ScriptOllama.cmd`, `tests/cloud/test_launcher.py`, `README.md`.

**Interfaces:** `main(argv=None) -> int` uses platform cache paths, the existing folder convention and isolated editable installation. Arguments/environment configure model, context/image limits, DPI, FORCE and UPDATE_MARKER. The macOS name remains a compatible entry point.

- [ ] Add tests for first installation, ordinary second run with unavailable remote, one-time migration, Unicode/spaced paths, Windows executable paths, cancelled credentials, partial completion and the explicit secret-leakage gate above.
- [ ] Run failures.
- [ ] Implement one shared standard-library launcher and minimal OS wrappers. Use hidden credential input with GUI/terminal fallbacks; pass keys only via child environment. Publish Markdown/resources after conversion and report incomplete results.
- [ ] Verify launcher scripts and write concrete macOS, Ubuntu and Windows instructions; commit.

### Task 6: Portability, review and deployment

**Files:** `.github/workflows/cloud-conversion.yml`, portability fixtures/tests, approved plan/spec and README.

- [ ] Add hosted macOS/Windows/Linux checks and an Ubuntu 25 container check. Run the cloud-path tests without model downloads; avoid relying on the existing unavailable `t4_gpu` runner.
- [ ] Run all focused regressions, source-PDF checks and a separate manual real Ollama exercise capped at ten requests. Inspect final table/math/figure previews and review the whole change for dropped content, path bugs, accidental secret logging and misleading success messages.
- [ ] Publish the branch, inspect CI results, and update the installed Mac cache and launcher. Report which other OSes actually ran tests and any remaining blocked checks.
