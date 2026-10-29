"""Agent 4: create human-in-the-loop analyst investigation briefs."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT1_FILE = PROJECT_ROOT / "outputs" / "agent1_results.json"
AGENT2_FILE = PROJECT_ROOT / "outputs" / "agent2_results.json"
AGENT3_FILE = PROJECT_ROOT / "outputs" / "agent3_results.json"
NEWS_FILE = PROJECT_ROOT / "data" / "news.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "agent4_analyst_brief.json"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
REQUEST_TIMEOUT_SECONDS = 60

# Keep the brief format consistent and machine-readable.
BRIEF_SCHEMA = {
	"type": "object",
	"properties": {
		"event": {"type": "string"},
		"company": {"type": "string"},
		"priority": {"type": "string"},
		"executive_summary": {"type": "string"},
		"market_event": {"type": "string"},
		"company_exposure": {"type": "string"},
		"potential_impact_mechanism": {"type": "string"},
		"potential_financial_areas": {"type": "array", "items": {"type": "string"}},
		"known_evidence": {"type": "array", "items": {"type": "string"}},
		"uncertainties": {"type": "array", "items": {"type": "string"}},
		"analyst_questions": {"type": "array", "items": {"type": "string"}},
		"next_investigation_steps": {"type": "array", "items": {"type": "string"}},
	},
	"required": [
		"event",
		"company",
		"priority",
		"executive_summary",
		"market_event",
		"company_exposure",
		"potential_impact_mechanism",
		"potential_financial_areas",
		"known_evidence",
		"uncertainties",
		"analyst_questions",
		"next_investigation_steps",
	],
	"additionalProperties": False,
}

BRIEF_TEXT_FIELDS = (
	"event",
	"company",
	"priority",
	"executive_summary",
	"market_event",
	"company_exposure",
	"potential_impact_mechanism",
)
BRIEF_LIST_FIELDS = (
	"potential_financial_areas",
	"known_evidence",
	"uncertainties",
	"analyst_questions",
	"next_investigation_steps",
)


def _error_brief(
	agent2_result: dict[str, Any],
	agent3_result: dict[str, Any],
	message: str,
) -> dict[str, Any]:
	"""Return a complete brief-shaped error instead of losing a company/event."""
	return {
		"event": str(agent2_result.get("event_type", "")),
		"company": str(agent3_result.get("company", "")),
		"priority": str(agent2_result.get("priority", "")),
		"executive_summary": f"ERROR: {message}",
		"market_event": "",
		"company_exposure": "",
		"potential_impact_mechanism": "",
		"potential_financial_areas": [],
		"known_evidence": [],
		"uncertainties": [message],
		"analyst_questions": [],
		"next_investigation_steps": [],
	}


def _load_json_list(path: Path) -> list[dict[str, Any]]:
	"""Read a JSON list, also accepting a wrapper object with a results list."""
	with path.open(encoding="utf-8") as json_file:
		data = json.load(json_file)
	if isinstance(data, dict):
		data = data.get("results", [])
	if not isinstance(data, list):
		raise ValueError(f"Expected a JSON list in {path}.")
	return [item for item in data if isinstance(item, dict)]


def _load_news_by_id(path: Path) -> dict[str, dict[str, str]]:
	"""Load optional article headlines and source metadata for evidence context."""
	if not path.exists():
		return {}
	frame = pd.read_csv(path, dtype=str).fillna("")
	return {
		str(row.get("id", "")): row
		for row in frame.to_dict(orient="records")
		if row.get("id", "")
	}


def _is_error(record: dict[str, Any]) -> bool:
	"""Check upstream records for the error markers used by Agents 1-3."""
	return any(
		str(record.get(field, "")).startswith("ERROR:")
		for field in ("reason", "reason_for_priority", "executive_summary")
	)


def create_analyst_brief(
	agent1_result: dict[str, Any],
	agent2_result: dict[str, Any],
	agent3_result: dict[str, Any],
	news_article: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Generate one evidence-aware, non-rating-action analyst brief."""
	if any(_is_error(record) for record in (agent1_result, agent2_result, agent3_result)):
		return _error_brief(agent2_result, agent3_result, "An upstream agent result contains an error.")

	# Keep the event and company identity anchored to upstream records.
	event_name = str(agent2_result.get("event_type", agent1_result.get("event_type", "")))
	company_name = str(agent3_result.get("company", ""))
	priority = str(agent2_result.get("priority", ""))
	inputs = {
		"news_article": news_article or {},
		"agent1_event_detection": agent1_result,
		"agent2_research_assessment": agent2_result,
		"agent3_company_impact_assessment": agent3_result,
	}

	# Tell the model to distinguish source data from AI hypotheses and unknowns.
	prompt = f"""
You are Agent 4, a human-in-the-loop analyst copilot. Create a concise
investigation brief for the event and company below using ONLY the supplied data.

The brief must cover: executive summary, market event, company exposure, potential
impact mechanism, potential financial areas, known evidence, uncertainties,
questions for the rating analyst, and next investigation steps.

Clearly distinguish source facts from AI-generated interpretations and items that
need analyst verification. Treat news headline/source/date as the only direct news
evidence supplied; Agent 1, Agent 2, and Agent 3 fields are AI-generated hypotheses,
not independently verified facts. Do not invent company facts, financial numbers,
or event details. State uncertainty explicitly where evidence is limited.

This tool must NOT make or recommend a credit-rating decision. Do not recommend
an upgrade, downgrade, or rating change. Next steps must be investigative actions
only, such as reviewing financial statements, cost/revenue exposure, pass-through
ability, liquidity, debt maturities, hedging, historical periods, or management
commentary. Do not assert that any of these items are known unless supplied.

Return ONLY valid JSON matching the supplied schema. Use these trusted identity
values: event={json.dumps(event_name)}, company={json.dumps(company_name)},
priority={json.dumps(priority)}.

Available inputs:
{json.dumps(inputs, ensure_ascii=False, indent=2, default=str)}
""".strip()

	# Request one structured brief from the local Ollama model.
	try:
		response = requests.post(
			OLLAMA_URL,
			json={
				"model": OLLAMA_MODEL,
				"prompt": prompt,
				"format": BRIEF_SCHEMA,
				"stream": False,
				"options": {"temperature": 0},
			},
			timeout=REQUEST_TIMEOUT_SECONDS,
		)
		response.raise_for_status()
	except requests.Timeout:
		return _error_brief(agent2_result, agent3_result, "Ollama request timed out.")
	except requests.RequestException as error:
		return _error_brief(agent2_result, agent3_result, f"Ollama request failed: {error}")

	# Parse and validate the model result before showing it to the analyst.
	try:
		brief = json.loads(response.json()["response"])
	except (KeyError, TypeError, ValueError):
		return _error_brief(agent2_result, agent3_result, "Ollama returned invalid JSON.")

	if not isinstance(brief, dict) or set(brief) != set(BRIEF_SCHEMA["required"]):
		return _error_brief(agent2_result, agent3_result, "Ollama response did not match the brief schema.")
	if not all(isinstance(brief[field], str) for field in BRIEF_TEXT_FIELDS):
		return _error_brief(agent2_result, agent3_result, "Ollama returned invalid text fields.")
	if not all(
		isinstance(brief[field], list) and all(isinstance(item, str) for item in brief[field])
		for field in BRIEF_LIST_FIELDS
	):
		return _error_brief(agent2_result, agent3_result, "Ollama returned invalid list fields.")

	# Enforce upstream identities even if the model returns different values.
	brief["event"] = event_name
	brief["company"] = company_name
	brief["priority"] = priority
	# Separate supplied source text from interpretations generated by the models.
	if news_article and news_article.get("headline"):
		source_details = ", ".join(
			value
			for value in (news_article.get("source"), news_article.get("date"))
			if value
		)
		source_note = f" (source/date: {source_details})" if source_details else ""
		brief["known_evidence"] = [
			f'Supplied news headline{source_note} (not independently verified): '
			f'"{news_article["headline"]}"'
		]
	else:
		brief["known_evidence"] = [
			"Original news headline/source metadata unavailable; verify against the source article."
		]
	brief["executive_summary"] = (
		"AI-generated interpretation (unverified): " + brief["executive_summary"]
	)
	brief["market_event"] = (
		"AI-generated interpretation; verify against original reporting: "
		+ brief["market_event"]
	)
	brief["company_exposure"] = (
		"AI-generated interpretation (unverified): " + brief["company_exposure"]
	)
	brief["potential_impact_mechanism"] = (
		"AI-generated interpretation (unverified): " + brief["potential_impact_mechanism"]
	)
	brief["potential_financial_areas"] = [
		f"Potential area to investigate: {area}"
		for area in brief["potential_financial_areas"]
	]
	brief["uncertainties"] = [
		f"Requires analyst verification: {item}"
		for item in brief["uncertainties"]
	]
	return brief


def _print_brief(brief: dict[str, Any]) -> None:
	"""Print the complete brief and the requested analyst handoff excerpt."""
	print("\n" + "=" * 60)
	print("ANALYST BRIEF")
	print("=" * 60)
	for field in BRIEF_TEXT_FIELDS:
		label = field.replace("_", " ").title()
		print(f"\n{label}:\n{brief[field]}")
	for field in BRIEF_LIST_FIELDS:
		label = field.replace("_", " ").title()
		print(f"\n{label}:")
		if brief[field]:
			for item in brief[field]:
				print(f"- {item}")
		else:
			print("- None provided")

	print("\n" + "=" * 60)
	print("ANALYST HANDOFF")
	print("=" * 60)
	print(f"\nEvent: {brief['event']}")
	print(f"Company: {brief['company']}")
	print(f"Priority: {brief['priority']}")
	print(f"\nExecutive Summary:\n{brief['executive_summary']}")
	print(f"\nPotential Impact:\n{brief['potential_impact_mechanism']}")
	print("\nAnalyst Questions:")
	for item in brief["analyst_questions"]:
		print(f"- {item}")
	print("\nNext Investigation Steps:")
	for item in brief["next_investigation_steps"]:
		print(f"- {item}")


def build_analyst_briefs(
	agent1_file: Path = AGENT1_FILE,
	agent2_file: Path = AGENT2_FILE,
	agent3_file: Path = AGENT3_FILE,
	news_file: Path = NEWS_FILE,
) -> list[dict[str, Any]]:
	"""Join upstream output records and create one brief per Agent 3 record."""
	missing_files = [path for path in (agent1_file, agent2_file, agent3_file) if not path.exists()]
	if missing_files:
		for path in missing_files:
			print(f"Missing required input file: {path}")
		return []

	try:
		agent1_results = _load_json_list(agent1_file)
		agent2_results = _load_json_list(agent2_file)
		agent3_results = _load_json_list(agent3_file)
		news_by_id = _load_news_by_id(news_file)
	except (OSError, json.JSONDecodeError, ValueError, pd.errors.ParserError) as error:
		print(f"Could not load Agent 4 input data: {error}")
		return []

	# Index the event analyses for joining each company exposure to its event.
	agent1_by_id = {str(item.get("article_id", "")): item for item in agent1_results}
	agent2_by_id = {str(item.get("article_id", "")): item for item in agent2_results}
	brief_inputs = []
	for agent3_result in agent3_results:
		event_id = str(agent3_result.get("event_id", ""))
		agent1_result = agent1_by_id.get(event_id)
		agent2_result = agent2_by_id.get(event_id)
		if agent1_result is None or agent2_result is None:
			print(f"Skipping Agent 3 result for event {event_id}: matching Agent 1/2 result missing.")
			continue
		brief_inputs.append(
			(
				agent1_result,
				agent2_result,
				agent3_result,
				news_by_id.get(event_id),
			)
		)

	briefs = []
	for index, (agent1_result, agent2_result, agent3_result, news_article) in enumerate(brief_inputs, start=1):
		print(
			f"\nCreating analyst brief {index}/{len(brief_inputs)}: "
			f"{agent3_result.get('company', '')} / {agent2_result.get('event_type', '')}",
			flush=True,
		)
		brief = create_analyst_brief(agent1_result, agent2_result, agent3_result, news_article)
		briefs.append(brief)
		_print_brief(brief)
	return briefs


def main() -> None:
	"""Build briefs, report missing input files, and save output JSON."""
	briefs = build_analyst_briefs()
	OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
	OUTPUT_FILE.write_text(json.dumps(briefs, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"\nAnalyst briefs saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
	main()
