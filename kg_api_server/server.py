#!/usr/bin/env python3
"""
Lightweight JSON API server for teammate-style Step 7+ validation.

Dependency-free MVP using Python stdlib only.
"""
from __future__ import annotations

import argparse
import json
import ssl
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent / "data" / "frc_seed_catalog.json"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def normalize_name(name: str) -> str:
    return str(name).lower().replace(" ", "").replace("-", "").replace("_", "")


def load_catalog() -> dict:
    if not DATA_PATH.exists():
        raise SystemExit(
            f"Seed catalog not found: {DATA_PATH}\n"
            "Run build_seed_catalog.py first."
        )
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


CATALOG = load_catalog()
RECORDS = CATALOG["records"]
NAME_INDEX = {}
for rec in RECORDS:
    for alias in rec.get("aliases", []):
        NAME_INDEX[normalize_name(alias)] = rec


def json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def lookup_drug(name: str):
    return NAME_INDEX.get(normalize_name(name))


def build_basic_payload(rec: dict) -> dict:
    return {
        "drug_id": rec["drug_id"],
        "drug_name": rec["drug_name"],
        "target": rec["target"],
        "pathway": rec["pathway"],
        "predicted_ic50": rec["predicted_ic50"],
        "sensitivity_rate": rec["sensitivity_rate"],
        "validation_score": rec["validation_score"],
        "safety_score": rec["safety_score"],
        "combined_score": rec["combined_score"],
        "final_rank": rec["final_rank"],
        "category": rec["category"],
        "clinical_bucket": rec["clinical_bucket"],
        "known_brca": rec["known_brca"],
        "target_expressed": rec["target_expressed"],
        "brca_pathway": rec["brca_pathway"],
        "survival_sig": rec["survival_sig"],
        "recommendation_note": rec["recommendation_note"],
    }


def cache_path(prefix: str, key: str) -> Path:
    safe = normalize_name(key)
    return CACHE_DIR / f"{prefix}_{safe}.json"


def load_cache(prefix: str, key: str):
    path = cache_path(prefix, key)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_cache(prefix: str, key: str, payload: dict) -> None:
    path = cache_path(prefix, key)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def http_get_json(url: str, timeout: int = 20):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "say2-preproject-kg-api/1.0",
            "Accept": "application/json",
        },
    )
    context = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
        return json.loads(resp.read().decode())


def fetch_pubmed(query: str, max_results: int = 10) -> list[dict]:
    cached = load_cache("pubmed", f"{query}_{max_results}")
    if cached is not None:
        return cached.get("data", [])

    esearch_params = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "retmode": "json",
            "sort": "relevance",
            "retmax": str(max_results),
            "term": query,
        }
    )
    esearch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{esearch_params}"
    search_payload = http_get_json(esearch_url)
    idlist = search_payload.get("esearchresult", {}).get("idlist", [])
    if not idlist:
        payload = {"data": []}
        save_cache("pubmed", f"{query}_{max_results}", payload)
        return []

    esummary_params = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "retmode": "json",
            "id": ",".join(idlist),
        }
    )
    esummary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?{esummary_params}"
    summary_payload = http_get_json(esummary_url)
    result = summary_payload.get("result", {})

    rows = []
    for pmid in idlist:
        item = result.get(pmid, {})
        if not item:
            continue
        rows.append(
            {
                "pmid": pmid,
                "title": item.get("title"),
                "journal": item.get("fulljournalname"),
                "year": str(item.get("pubdate", "")).split(" ")[0],
                "authors": [a.get("name") for a in item.get("authors", []) if isinstance(a, dict)],
                "source": "ncbi_pubmed",
            }
        )

    payload = {"data": rows}
    save_cache("pubmed", f"{query}_{max_results}", payload)
    return rows


def fetch_trials(drug_name: str, max_results: int = 10) -> list[dict]:
    cache_key = f"{drug_name}_{max_results}"
    cached = load_cache("trials", cache_key)
    if cached is not None:
        return cached.get("data", [])

    # ClinicalTrials.gov official modern API (query.term is the default text search area).
    query = f"{drug_name} breast cancer"
    params = urllib.parse.urlencode(
        {
            "query.term": query,
            "pageSize": str(max_results),
            "format": "json",
        }
    )
    url = f"https://clinicaltrials.gov/api/v2/studies?{params}"
    payload = http_get_json(url)

    studies = payload.get("studies", [])
    rows = []
    for study in studies:
        ps = study.get("protocolSection", {})
        ident = ps.get("identificationModule", {})
        status = ps.get("statusModule", {})
        design = ps.get("designModule", {})
        cond = ps.get("conditionsModule", {})
        sponsor = ps.get("sponsorCollaboratorsModule", {})

        rows.append(
            {
                "nct_id": ident.get("nctId"),
                "title": ident.get("briefTitle"),
                "phase": ", ".join(design.get("phases", []) or []),
                "status": status.get("overallStatus"),
                "conditions": cond.get("conditions", []) or [],
                "lead_sponsor": sponsor.get("leadSponsor", {}).get("name"),
                "source": "clinicaltrials_gov_v2",
            }
        )

    save_cache("trials", cache_key, {"data": rows})
    return rows


def fetch_side_effects(drug_name: str, max_results: int = 10) -> list[dict]:
    cache_key = f"{drug_name}_{max_results}"
    cached = load_cache("faers", cache_key)
    if cached is not None:
        return cached.get("data", [])

    # openFDA drug adverse event aggregated by MedDRA preferred term.
    search = f'patient.drug.medicinalproduct:"{drug_name}"'
    params = urllib.parse.urlencode(
        {
            "search": search,
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": str(max_results),
        }
    )
    url = f"https://api.fda.gov/drug/event.json?{params}"
    payload = http_get_json(url)
    rows = []
    for item in payload.get("results", []):
        rows.append(
            {
                "event": item.get("term"),
                "count": item.get("count"),
                "source": "openfda_faers",
            }
        )

    save_cache("faers", cache_key, {"data": rows})
    return rows


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json_bytes(payload)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A003
        return

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/health":
            self._send(
                200,
                {
                    "ok": True,
                    "status": "ok",
                    "records": len(RECORDS),
                    "ensemble_name": CATALOG["ensemble_name"],
                    "input_bundle": CATALOG["input_bundle"],
                },
            )
            return

        if path.startswith("/api/pubmed"):
            q = query.get("query", [""])[0]
            max_results = int(query.get("max_results", ["10"])[0])
            try:
                matches = fetch_pubmed(q, max_results=max_results)
                self._send(200, {"ok": True, "query": q, "data": matches, "source": "pubmed_live"})
            except Exception:
                matches = []
                q_norm = normalize_name(q.replace("breastcancer", "").strip())
                for rec in RECORDS:
                    if q_norm and q_norm in normalize_name(rec["drug_name"]):
                        matches.append(
                            {
                                "title": f"{rec['drug_name']} and breast cancer relevance summary",
                                "journal": "Local seed catalog",
                                "year": 2026,
                                "pmid": None,
                                "summary": rec["recommendation_note"],
                                "source": "local_fallback",
                            }
                        )
                self._send(200, {"ok": True, "query": q, "data": matches, "source": "local_fallback"})
            return

        if not path.startswith("/api/drug/"):
            self._send(404, {"ok": False, "error": "not_found"})
            return

        remainder = path[len("/api/drug/"):]
        parts = [urllib.parse.unquote(x) for x in remainder.split("/") if x]
        if not parts:
            self._send(404, {"ok": False, "error": "missing_drug_name"})
            return

        drug_name = parts[0]
        rec = lookup_drug(drug_name)
        if not rec:
            self._send(404, {"ok": False, "error": "drug_not_found", "drug_name": drug_name})
            return

        if len(parts) == 1:
            self._send(200, {"ok": True, "data": build_basic_payload(rec)})
            return

        sub = parts[1]
        if sub == "targets":
            target_tokens = [x.strip() for x in str(rec["target"]).split(",") if x.strip()]
            data = [{"target": t, "source": "local_frc_step6_7"} for t in target_tokens]
            self._send(200, {"ok": True, "data": data})
            return

        if sub == "pathways":
            data = [{"pathway": rec["pathway"], "source": "local_frc_step6_7"}]
            self._send(200, {"ok": True, "data": data})
            return

        if sub == "side_effects":
            try:
                data = fetch_side_effects(rec["drug_name"], max_results=10)
                self._send(200, {"ok": True, "data": data, "source": "openfda_live"})
            except Exception:
                data = []
                flags = str(rec.get("flags", "[]"))
                if flags and flags != "[]":
                    for flag in [x.strip(" []'") for x in flags.split(",") if x.strip()]:
                        data.append({"event": flag, "severity": "flagged", "source": "local_admet"})
                self._send(200, {"ok": True, "data": data, "source": "local_fallback"})
            return

        if sub == "trials":
            try:
                data = fetch_trials(rec["drug_name"], max_results=10)
                self._send(200, {"ok": True, "data": data, "source": "clinicaltrials_live"})
            except Exception:
                data = []
                if rec["known_brca"]:
                    data.append(
                        {
                            "title": f"{rec['drug_name']} breast cancer related usage signal",
                            "phase": "Known / related evidence",
                            "status": "Review needed",
                            "source": "local_brca_relevance",
                        }
                    )
                self._send(200, {"ok": True, "data": data, "source": "local_fallback"})
            return

        self._send(404, {"ok": False, "error": "unknown_endpoint"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"KG/API MVP server listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
