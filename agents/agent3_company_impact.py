"""Agent 3: assess potential exposure of supported portfolio companies to events."""

import json
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "outputs" / "agent2_results.json"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "agent3_results.json"
COMPANIES_FILE = PROJECT_ROOT / "data" / "companies.json"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
REQUEST_TIMEOUT_SECONDS = 60

# Ollama constrains its response to this JSON structure.
IMPACT_SCHEMA = {
	"type": "object",
	"properties": {
		"event_id": {"type": "string"},
		"company": {"type": "string"},
		"industry": {"type": "string"},
		"potentially_impacted": {"type": "boolean"},
		"impact_direction": {
			"type": "string",
			"enum": [
				"potentially negative",
				"potentially positive",
				"mixed",
				"uncertain",
				"no material exposure identified",
			],
		},
		"impact_mechanism": {"type": "string"},
		"business_area": {"type": "string"},
		"potential_financial_impact": {
			"type": "array",
			"items": {"type": "string"},
		},
		"analyst_investigation_points": {
			"type": "array",
			"items": {"type": "string"},
		},
		"confidence": {"type": "number", "minimum": 0, "maximum": 1},
		"reason": {"type": "string"},
	},
	"required": [
		"event_id",
		"company",
		"industry",
		"potentially_impacted",
		"impact_direction",
		"impact_mechanism",
		"business_area",
		"potential_financial_impact",
		"analyst_investigation_points",
		"confidence",
		"reason",
	],
	"additionalProperties": False,
}


def _error_result(event: dict[str, Any], company: dict[str, str], message: str) -> dict[str, Any]:
	"""Represent a failed assessment without treating it as no exposure."""
	return {
		"event_id": str(event.get("article_id", event.get("event_id", ""))),
		"company": company["company"],
		"industry": company["industry"],
		"potentially_impacted": False,
		"impact_direction": "uncertain",
		"impact_mechanism": "",
		"business_area": "",
		"potential_financial_impact": [],
		"analyst_investigation_points": [],
		"confidence": 0.0,
		"reason": f"ERROR: {message}",
	}


def load_client_portfolio(path: Path = COMPANIES_FILE) -> list[dict[str, str]]:
	"""Load supported company profiles from companies.json."""
	if not path.exists():
		return []
	try:
		with path.open(encoding="utf-8") as company_file:
			data = json.load(company_file)
	except (OSError, json.JSONDecodeError):
		return []
	if isinstance(data, dict):
		data = data.get("companies", [])
	if not isinstance(data, list):
		return []
	return [
		{
			"company": str(company.get("company", "")).strip(),
			"industry": str(company.get("industry", "")).strip(),
			"business": str(company.get("business", "")).strip(),
		}
		for company in data
		if isinstance(company, dict) and str(company.get("company", "")).strip()
	]


def assess_company_impact(
	event: dict[str, Any],
	company: dict[str, str],
) -> dict[str, Any]:
	"""Ask Ollama whether one event could expose one portfolio company."""
	event_id = str(event.get("article_id", event.get("event_id", "")))
	event_json = json.dumps(event, ensure_ascii=False, default=str)
	company_json = json.dumps(company, ensure_ascii=False)

	# Require evidence-based exposure reasoning and avoid assumptions about the client.
	prompt = f"""
You are Agent 3, assessing potential company exposure to an event for human analyst
investigation. Analyze the event information and the supplied company profile.

Determine whether the company is potentially exposed, what business area and
impact mechanism could be involved, the potential direction of impact, which
financial areas could be affected, and what an analyst should investigate.
Potential financial areas include revenue, operating costs, EBITDA or operating
margin, cash flow, working capital, capital expenditure, liquidity, leverage,
and debt servicing capacity.

Do not assume exposure merely because the company and event share an industry.
Explain why exposure may or may not exist. Use only the event and company facts
provided. Do not invent company-specific facts, financial values, contracts,
hedges, or asset locations. State uncertainty clearly. This is an AI-generated
hypothesis for human investigation, not a verified fact. Do not make or recommend
any credit-rating action.

If no material exposure is supported by the provided information, set
potentially_impacted to false, impact_direction to "no material exposure
identified", and explain the evidence gap in reason. Otherwise use one of the
other allowed impact_direction values. Return ONLY valid JSON matching the
supplied schema, with no markdown or extra text.

Event information:
{event_json}

Company profile:
{company_json}
""".strip()

	# Send a separate request for each event-company pair.
	try:
		response = requests.post(
			OLLAMA_URL,
			json={
				"model": OLLAMA_MODEL,
				"prompt": prompt,
				"format": IMPACT_SCHEMA,
				"stream": False,
				"options": {"temperature": 0},
			},
			timeout=REQUEST_TIMEOUT_SECONDS,
		)
		response.raise_for_status()
	except requests.Timeout:
		return _error_result(event, company, "Ollama request timed out.")
	except requests.RequestException as error:
		return _error_result(event, company, f"Ollama request failed: {error}")

	# Parse the response envelope and the model's JSON object.
	try:
		result = json.loads(response.json()["response"])
	except (KeyError, TypeError, ValueError):
		return _error_result(event, company, "Ollama returned invalid JSON.")

	# Reject malformed fields so invalid model output is counted as an error.
	if not isinstance(result, dict) or set(result) != set(IMPACT_SCHEMA["required"]):
		return _error_result(event, company, "Ollama response did not match the impact schema.")
	if not isinstance(result["potentially_impacted"], bool):
		return _error_result(event, company, "Ollama returned an invalid exposure flag.")
	if result["impact_direction"] not in {
		"potentially negative",
		"potentially positive",
		"mixed",
		"uncertain",
		"no material exposure identified",
	}:
		return _error_result(event, company, "Ollama returned an invalid impact direction.")
	if not isinstance(result["confidence"], (int, float)) or isinstance(result["confidence"], bool):
		return _error_result(event, company, "Ollama returned an invalid confidence value.")
	if not 0 <= result["confidence"] <= 1:
		return _error_result(event, company, "Ollama confidence must be between 0 and 1.")
	if not all(
		isinstance(result[field], list)
		and all(isinstance(item, str) for item in result[field])
		for field in ("potential_financial_impact", "analyst_investigation_points")
	):
		return _error_result(event, company, "Ollama returned invalid list values.")
	if not all(
		isinstance(result[field], str)
		for field in ("event_id", "company", "industry", "impact_mechanism", "business_area", "reason")
	):
		return _error_result(event, company, "Ollama returned invalid text values.")

	# Keep event and client identity tied to the inputs, not model-generated values.
	result["event_id"] = event_id
	result["company"] = company["company"]
	result["industry"] = company["industry"]
	return result


def process_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Assess every HIGH/MEDIUM event against each supported company profile."""
	priority_events = [
		event
		for event in events
		if isinstance(event, dict) and event.get("priority") in {"HIGH", "MEDIUM"}
	]
	companies_to_analyze = load_client_portfolio()
	results = []
	error_count = 0
	impacted_count = 0
	no_exposure_count = 0

	print("=" * 60, flush=True)
	print("AGENT 3 - COMPANY IMPACT ANALYSIS", flush=True)
	print("=" * 60, flush=True)
	if not companies_to_analyze:
		print(f"No supported entities found in {COMPANIES_FILE}.", flush=True)

	for event_number, event in enumerate(priority_events, start=1):
		event_id = str(event.get("article_id", event.get("event_id", "")))
		print(f"\nEvent {event_number}/{len(priority_events)}: {event.get('event_type', '')}", flush=True)
		for company in companies_to_analyze:
			print(f"\nCompany: {company['company']}", flush=True)
			result = assess_company_impact(event, company)
			results.append(result)

			if result["reason"].startswith("ERROR:"):
				error_count += 1
				print(f"Error: {result['reason']}", flush=True)
			else:
				if result["potentially_impacted"]:
					impacted_count += 1
				else:
					no_exposure_count += 1
				print(f"Potentially Impacted: {'YES' if result['potentially_impacted'] else 'NO'}", flush=True)
				print(f"Impact Direction: {result['impact_direction']}", flush=True)
				print(f"Impact Mechanism: {result['impact_mechanism']}", flush=True)
				print(f"Business Area: {result['business_area']}", flush=True)
				print(
					"Potential Financial Impact: "
					+ (", ".join(result["potential_financial_impact"]) or "None identified"),
					flush=True,
				)
				print("Analyst Investigation:", flush=True)
				for point in result["analyst_investigation_points"]:
					print(f"- {point}", flush=True)
				print(f"Confidence: {result['confidence']:.2f}", flush=True)
				print(f"Reason: {result['reason']}", flush=True)
			print("Complete result:", flush=True)
			print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
			print("-" * 60, flush=True)

	print("\n" + "=" * 60, flush=True)
	print("AGENT 3 SUMMARY", flush=True)
	print("=" * 60, flush=True)
	print(f"\nEvents analyzed: {len(priority_events)}", flush=True)
	print(f"Companies analyzed: {len(results)}", flush=True)
	print(f"Potentially impacted: {impacted_count}", flush=True)
	print(f"No material exposure: {no_exposure_count}", flush=True)
	print(f"Errors: {error_count}", flush=True)
	return results


def main() -> None:
	"""Load Agent 2 output, run the portfolio assessment, and save JSON results."""
	if not INPUT_FILE.exists():
		print(f"Agent 2 results file not found: {INPUT_FILE}", flush=True)
		events = []
	else:
		try:
			with INPUT_FILE.open(encoding="utf-8") as result_file:
				events = json.load(result_file)
			if isinstance(events, dict):
				events = events.get("results", [])
			if not isinstance(events, list):
				raise ValueError("Agent 2 results must be a JSON list.")
		except (OSError, json.JSONDecodeError, ValueError) as error:
			print(f"Could not load Agent 2 results: {error}", flush=True)
			events = []

	results = process_events(events)
	OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
	OUTPUT_FILE.write_text(
		json.dumps(results, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)
	print(f"\nStructured results saved to: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
	main()
