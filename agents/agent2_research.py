"""Agent 2: prepare an initial research assessment for rating analysts."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
	from .agent1_event_detection import detect_event as detect_agent1_event
except ImportError:
	from agent1_event_detection import detect_event as detect_agent1_event


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT1_RESULTS_FILE = PROJECT_ROOT / "outputs" / "agent1_results.json"
NEWS_FILE = PROJECT_ROOT / "data" / "news.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "agent2_results.json"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
REQUEST_TIMEOUT_SECONDS = 60

# Constrain Ollama to the exact research result fields expected by Agent 2.
RESEARCH_SCHEMA = {
	"type": "object",
	"properties": {
		"article_id": {"type": "string"},
		"event_type": {"type": "string"},
		"assessment": {"type": "string"},
		"background": {"type": "string"},
		"significance": {"type": "string"},
		"potentially_affected_industries": {
			"type": "array",
			"items": {"type": "string"},
		},
		"analyst_investigation_points": {
			"type": "array",
			"items": {"type": "string"},
		},
		"priority": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
		"reason_for_priority": {"type": "string"},
	},
	"required": [
		"article_id",
		"event_type",
		"assessment",
		"background",
		"significance",
		"potentially_affected_industries",
		"analyst_investigation_points",
		"priority",
		"reason_for_priority",
	],
	"additionalProperties": False,
}


def _error_result(agent1_result: dict[str, Any], message: str) -> dict[str, Any]:
	"""Keep errors in the same JSON shape as successful research results."""
	return {
		"article_id": str(agent1_result.get("article_id", "")),
		"event_type": str(agent1_result.get("event_type", "")),
		"assessment": "",
		"background": "",
		"significance": "",
		"potentially_affected_industries": [],
		"analyst_investigation_points": [],
		"priority": "LOW",
		"reason_for_priority": f"ERROR: {message}",
	}


def research_event(
	article: dict[str, Any],
	agent1_result: dict[str, Any],
) -> dict[str, Any]:
	"""Ask Ollama for an initial, evidence-grounded research assessment."""
	article_id = str(agent1_result.get("article_id", article.get("id", "")))
	event_type = str(agent1_result.get("event_type", ""))
	article_json = json.dumps(article, ensure_ascii=False, default=str)
	agent1_json = json.dumps(agent1_result, ensure_ascii=False, default=str)

	# The article and Agent 1 analysis are evidence; investigation points are questions.
	prompt = f"""
You are Agent 2, a research assistant preparing an initial assessment for a human
credit rating analyst. Agent 1 flagged this event as relevant for investigation.

Assess:
1. What happened?
2. What is the underlying cause?
3. Why could this matter?
4. How significant could the event be?
5. Which industries could potentially be affected?
6. What additional information should the analyst investigate?
7. Assign an investigation priority: HIGH, MEDIUM, or LOW.

Use only information in the original article and Agent 1 analysis. Do not invent
facts or present assumptions as verified. Clearly identify missing information
and uncertainty. Investigation points must be questions or checks for the analyst.
Priority is only the urgency of human investigation, not the probability of a
rating change. Do not make a credit-rating decision or recommend an upgrade or
downgrade. This is an initial research assessment, not a verified fact.

Return ONLY valid JSON matching the supplied schema, with no markdown or extra
text. Use article_id {json.dumps(article_id)} and event_type
{json.dumps(event_type)} from the supplied data.

Original article:
{article_json}

Agent 1 structured analysis:
{agent1_json}
""".strip()

	# Request a complete JSON response from the local Ollama model.
	try:
		response = requests.post(
			OLLAMA_URL,
			json={
				"model": OLLAMA_MODEL,
				"prompt": prompt,
				"format": RESEARCH_SCHEMA,
				"stream": False,
				"options": {"temperature": 0},
			},
			timeout=REQUEST_TIMEOUT_SECONDS,
		)
		response.raise_for_status()
	except requests.Timeout:
		return _error_result(agent1_result, "Ollama request timed out.")
	except requests.RequestException as error:
		return _error_result(agent1_result, f"Ollama request failed: {error}")

	# Parse Ollama's response envelope, then the JSON produced by the model.
	try:
		result = json.loads(response.json()["response"])
	except (KeyError, TypeError, ValueError):
		return _error_result(agent1_result, "Ollama returned invalid JSON.")

	# Reject malformed output rather than passing incomplete data to analysts.
	if not isinstance(result, dict) or set(result) != set(RESEARCH_SCHEMA["required"]):
		return _error_result(agent1_result, "Ollama response did not match the research schema.")
	if result["priority"] not in {"HIGH", "MEDIUM", "LOW"}:
		return _error_result(agent1_result, "Ollama returned an invalid priority.")
	if not all(
		isinstance(result[field], list)
		and all(isinstance(item, str) for item in result[field])
		for field in ("potentially_affected_industries", "analyst_investigation_points")
	):
		return _error_result(agent1_result, "Ollama returned invalid list values.")
	if not all(
		isinstance(result[field], str)
		for field in (
			"article_id",
			"event_type",
			"assessment",
			"background",
			"significance",
			"reason_for_priority",
		)
	):
		return _error_result(agent1_result, "Ollama returned invalid text values.")

	# Keep the identifiers consistent with the trusted inputs, not model output.
	result["article_id"] = article_id
	result["event_type"] = event_type
	return result


def _load_news_articles(news_file: Path) -> dict[str, dict[str, str]]:
	"""Read original article rows and index them by their CSV id."""
	news = pd.read_csv(news_file, dtype=str).fillna("")
	return {
		str(row.get("id", "")): row
		for row in news.to_dict(orient="records")
		if row.get("id", "")
	}


def _load_agent1_results(
	results_file: Path,
	news_articles: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
	"""Load Agent 1 output, or run Agent 1 on one CSV row for a fallback demo."""
	if results_file.exists():
		with results_file.open(encoding="utf-8") as result_file:
			data = json.load(result_file)
		if isinstance(data, dict):
			data = data.get("results", [])
		if not isinstance(data, list):
			raise ValueError("Agent 1 results must be a JSON list.")
		return data

	# Reuse Agent 1 for a real structured fallback instead of inventing its analysis.
	first_article = next(iter(news_articles.values()), None)
	if first_article is None:
		return []
	print("Agent 1 results not found; creating a fallback from the first news article.")
	return [detect_agent1_event(first_article)]


def process_agent1_results(
	results_file: Path = AGENT1_RESULTS_FILE,
	news_file: Path = NEWS_FILE,
) -> list[dict[str, Any]]:
	"""Research every relevant Agent 1 result and print a concise summary."""
	news_articles = _load_news_articles(news_file)
	agent1_results = _load_agent1_results(results_file, news_articles)
	relevant_results = [
		result
		for result in agent1_results
		if isinstance(result, dict) and result.get("relevant") is True
	]

	print("=" * 60)
	print("AGENT 2 - RESEARCH ASSESSMENT")
	print("=" * 60)
	results = []
	error_count = 0
	priority_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

	for index, agent1_result in enumerate(relevant_results, start=1):
		article_id = str(agent1_result.get("article_id", ""))
		article = news_articles.get(article_id)
		print(f"\nResearching event {index}/{len(relevant_results)} (article {article_id})...")

		if article is None:
			result = _error_result(agent1_result, "Original article was not found in news.csv.")
		else:
			result = research_event(article, agent1_result)
		results.append(result)
		print(json.dumps(result, ensure_ascii=False, indent=2))

		if result["reason_for_priority"].startswith("ERROR:"):
			error_count += 1
		else:
			priority_counts[result["priority"]] += 1

	print("\n" + "=" * 60)
	print("AGENT 2 SUMMARY")
	print("=" * 60)
	print(f"\nEvents reviewed: {len(relevant_results)}")
	print(f"HIGH priority: {priority_counts['HIGH']}")
	print(f"MEDIUM priority: {priority_counts['MEDIUM']}")
	print(f"LOW priority: {priority_counts['LOW']}")
	print(f"Errors: {error_count}")
	return results


def main() -> None:
	results = process_agent1_results()
	OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
	OUTPUT_FILE.write_text(
		json.dumps(results, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)
	print(f"\nStructured results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
	main()
