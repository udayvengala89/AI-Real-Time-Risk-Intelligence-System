"""Read-only Streamlit dashboard for the AI Credit Risk POC outputs."""

import html
import csv
import json
import re
import uuid
from pathlib import Path
from typing import Any

import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILES = {
	"Agent 1": PROJECT_ROOT / "outputs" / "agent1_results.json",
	"Agent 2": PROJECT_ROOT / "outputs" / "agent2_results.json",
	"Agent 3": PROJECT_ROOT / "outputs" / "agent3_results.json",
	"Agent 4": PROJECT_ROOT / "outputs" / "agent4_analyst_brief.json",
}
PAGE_NAMES = ["Executive Dashboard", "News Feed", "Event Analysis", "Client Impact", "Analyst Copilot"]
NEWS_PATHS = {
	"CSV News": PROJECT_ROOT / "data" / "news.csv",
	"Live News (manual CSV)": PROJECT_ROOT / "data" / "live_news.csv",
}
COMPANIES_FILE = PROJECT_ROOT / "data" / "companies.json"

st.set_page_config(
	page_title="AI-Powered Credit Risk Early Warning System",
	page_icon="📊",
	layout="wide",
	initial_sidebar_state="expanded",
)


@st.cache_data(ttl=60)
def load_json_file(path_string: str) -> tuple[Any, str | None]:
	"""Read one JSON file and return a friendly error rather than raising."""
	path = Path(path_string)
	if not path.exists():
		return None, f"Missing file: {path.name}"
	try:
		with path.open(encoding="utf-8") as json_file:
			return json.load(json_file), None
	except (OSError, json.JSONDecodeError) as error:
		return None, f"Could not read {path.name}: {error}"


@st.cache_data(ttl=10)
def ollama_is_up() -> bool:
	"""Check the local Ollama tags endpoint without making a generation call."""
	try:
		return requests.get("http://127.0.0.1:11434/api/tags", timeout=2).ok
	except requests.RequestException:
		return False


def load_companies() -> list[dict[str, Any]]:
	"""Load supported entities from the existing companies JSON file."""
	if not COMPANIES_FILE.exists():
		return []
	try:
		with COMPANIES_FILE.open(encoding="utf-8") as company_file:
			data = json.load(company_file)
		if isinstance(data, dict):
			data = data.get("companies", [])
		return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
	except (OSError, json.JSONDecodeError):
		return []


def append_news_row(source: str, values: dict[str, str]) -> str:
	"""Assign an ID and append a user-entered article using the existing schema."""
	path = NEWS_PATHS[source]
	fields = (
		["id", "headline", "source", "date", "url"]
		if source == "CSV News"
		else ["article_id", "headline", "description", "source", "published_at", "url"]
	)
	path.parent.mkdir(parents=True, exist_ok=True)
	write_header = not path.exists() or path.stat().st_size == 0
	if source == "CSV News":
		used_ids = set()
		if not write_header:
			with path.open(encoding="utf-8-sig", newline="") as existing_file:
				used_ids = {row.get("id", "") for row in csv.DictReader(existing_file)}
		new_id = 1
		while str(new_id) in used_ids:
			new_id += 1
		id_field, article_id = "id", str(new_id)
	else:
		id_field, article_id = "article_id", str(uuid.uuid4())
	with path.open("a", encoding="utf-8", newline="") as news_file:
		writer = csv.DictWriter(news_file, fieldnames=fields, extrasaction="ignore")
		if write_header:
			writer.writeheader()
		writer.writerow({id_field: article_id, **values})
	return article_id


def as_records(value: Any) -> list[dict[str, Any]]:
	"""Accept a JSON list or a wrapper object containing a results list."""
	if isinstance(value, dict):
		value = value.get("results", [])
	if not isinstance(value, list):
		return []
	return [record for record in value if isinstance(record, dict)]


def text(value: Any, fallback: str = "Not available") -> str:
	"""Return safe, display-ready text for partially populated POC records."""
	if value is None or value == "":
		return fallback
	if isinstance(value, (dict, list)):
		return json.dumps(value, ensure_ascii=False)
	return str(value)


def joined(values: Any, fallback: str = "Not specified") -> str:
	"""Format a list of labels without exposing Python list syntax."""
	if isinstance(values, list):
		items = [str(item) for item in values if item not in (None, "")]
		return ", ".join(items) if items else fallback
	return text(values, fallback)


def values_list(values: Any) -> list[str]:
	"""Return non-empty string values from a JSON list."""
	if not isinstance(values, list):
		return []
	return [str(value) for value in values if value not in (None, "")]


def safe_number(value: Any, default: float = 0.0) -> float:
	"""Convert optional confidence values safely and keep progress values valid."""
	try:
		return min(1.0, max(0.0, float(value)))
	except (TypeError, ValueError):
		return default


def event_summary(event: dict[str, Any]) -> str:
	"""Use only available Agent 1/2 descriptions for the event summary."""
	detection = event.get("detection", {})
	research = event.get("research", {})
	return text(
		detection.get("event_summary") or research.get("assessment") or research.get("background"),
		"Event summary not available in POC data",
	)


def display_event_title(event: dict[str, Any]) -> str:
	"""Replace generic event labels with descriptive text already in JSON."""
	detection = event.get("detection", {})
	research = event.get("research", {})
	title = str(research.get("event_type") or detection.get("event_type") or "Event")
	if title.strip().lower() in {"supply shock", "oil supply shock", "event", "oil event"}:
		summary = event_summary(event)
		if summary != "Event summary not available in POC data":
			return summary.rstrip(". ")
	return title


def event_priority(event: dict[str, Any]) -> str:
	"""Return a normalized priority label from Agent 2."""
	return str(event.get("research", {}).get("priority") or "NOT AVAILABLE").upper()


def priority_tone(priority: Any) -> str:
	"""Map priorities to fixed CSS names, using neutral for unknown values."""
	return {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}.get(str(priority).upper(), "neutral")


def event_industries(event: dict[str, Any]) -> list[str]:
	"""Combine distinct industry labels supplied by Agents 1 and 2."""
	detection = event.get("detection", {})
	research = event.get("research", {})
	items = values_list(research.get("potentially_affected_industries"))
	items.extend(values_list(detection.get("potentially_affected_industries")))
	return list(dict.fromkeys(items))


def event_metadata(event: dict[str, Any], briefs: list[dict[str, Any]]) -> dict[str, str]:
	"""Read source and time metadata only when supplied in existing JSON."""
	records = [event.get("detection", {}), event.get("research", {})]
	metadata = {
		"source": "Not available in POC data",
		"published": "Not available in POC data",
		"detected": "Not available in POC data",
	}
	for record in records:
		for key in ("source", "publisher", "source_name"):
			if record.get(key):
				metadata["source"] = str(record[key])
				break
		for key in ("published_at", "published_timestamp", "published_date", "publication_date", "date"):
			if record.get(key):
				metadata["published"] = str(record[key])
				break
		for key in ("detected_at", "detected_timestamp", "detection_time", "created_at"):
			if record.get(key):
				metadata["detected"] = str(record[key])
				break

	# Agent 4 may preserve source/date inside its supplied-evidence note.
	if metadata["source"] == "Not available in POC data" or metadata["published"] == "Not available in POC data":
		event_names = {
			str(event.get("research", {}).get("event_type", "")).casefold(),
			display_event_title(event).casefold(),
		}
		for brief in briefs:
			if str(brief.get("event", "")).casefold() not in event_names:
				continue
			for evidence in values_list(brief.get("known_evidence")):
				match = re.search(r"source/date:\s*([^)]*)", evidence, flags=re.IGNORECASE)
				if not match:
					continue
				parts = [part.strip() for part in match.group(1).split(",", 1)]
				if metadata["source"] == "Not available in POC data" and parts and parts[0]:
					metadata["source"] = parts[0]
				if metadata["published"] == "Not available in POC data" and len(parts) > 1 and parts[1]:
					metadata["published"] = parts[1]
				break
	return metadata


def priority_rationale(event: dict[str, Any]) -> str:
	"""Summarize existing Agent 2 rationale without creating new analysis."""
	research = event.get("research", {})
	parts = []
	if research.get("significance"):
		parts.append(f"Significance: {research['significance']}")
	if research.get("reason_for_priority"):
		parts.append(str(research["reason_for_priority"]))
	if not parts:
		for field in ("assessment", "background"):
			if research.get(field):
				parts.append(str(research[field]))
				break
	return " ".join(parts) if parts else "Prioritization rationale not available in POC data."


def build_events(agent1: list[dict[str, Any]], agent2: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Join event detections and research priorities using the article ID."""
	agent1_by_id = {str(row.get("article_id", "")): row for row in agent1}
	agent2_by_id = {str(row.get("article_id", "")): row for row in agent2}
	all_ids = list(dict.fromkeys([*agent2_by_id.keys(), *agent1_by_id.keys()]))
	events = []
	for event_id in all_ids:
		detection = agent1_by_id.get(event_id, {})
		research = agent2_by_id.get(event_id, {})
		if not detection.get("relevant", False) and not research:
			continue
		events.append({"event_id": event_id, "detection": detection, "research": research})
	return events


def event_label(event: dict[str, Any]) -> str:
	"""Create a concise event selector label."""
	priority = event_priority(event)
	return f"{priority}  |  {display_event_title(event)}  |  #{event['event_id']}"


def open_event(event_id: str) -> None:
	"""Select an event and navigate from a dashboard card to its analysis."""
	st.session_state["selected_event_id"] = event_id
	st.session_state["event_selector"] = event_id
	st.session_state["page_navigation"] = "Event Analysis"


def list_block(title: str, values: Any) -> None:
	"""Render a compact list used by client and analyst pages."""
	items = values if isinstance(values, list) else []
	st.markdown(f"**{title}**")
	if not items:
		st.caption("No items available")
		return
	for item in items:
		st.markdown(f"- {text(item)}")


def render_metric(label: str, value: int, note: str, accent: str) -> None:
	"""Render one restrained KPI card."""
	st.markdown(
		f"""
		<div class="metric-card" style="--metric-accent:{accent}">
		  <div class="metric-label">{html.escape(label)}</div>
		  <div class="metric-value">{value}</div>
		  <div class="metric-note">{html.escape(note)}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


def render_event_card(event: dict[str, Any], index: int, briefs: list[dict[str, Any]]) -> None:
	"""Render a traceable HIGH-priority event card and a direct analysis action."""
	priority = event_priority(event)
	title = display_event_title(event)
	summary = event_summary(event)
	industries = joined(event_industries(event), "Not available in POC data")
	metadata = event_metadata(event, briefs)
	rationale = priority_rationale(event)
	st.markdown(
		f"""
		<div class="event-card">
		  <div class="event-card-top">
		    <span class="priority-pill priority-{priority_tone(priority)}">{html.escape(priority)} PRIORITY</span>
		    <span class="event-id">EVENT #{html.escape(event['event_id'])}</span>
		  </div>
		  <h3>{html.escape(title)}</h3>
		  <p>{html.escape(text(summary))}</p>
		  <div class="event-industries"><b>Potentially affected industries</b><br>{html.escape(industries)}</div>
		  <div class="event-metadata">
		    <span><b>Source</b> {html.escape(metadata['source'])}</span>
		    <span><b>Published</b> {html.escape(metadata['published'])}</span>
		    <span><b>Detected</b> {html.escape(metadata['detected'])}</span>
		  </div>
		  <div class="priority-rationale"><b>Why this was prioritized</b><br>{html.escape(rationale)}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)
	st.button(
		"Open event analysis",
		key=f"open_event_{index}_{event['event_id']}",
		use_container_width=True,
		on_click=open_event,
		args=(event["event_id"],),
	)


def render_sidebar(missing: list[str]) -> tuple[str, str]:
	"""Show source selector, manual news entry, navigation, and service status."""
	with st.sidebar:
		st.markdown('<div class="rail-kicker">RISK INTELLIGENCE / POC</div>', unsafe_allow_html=True)
		st.markdown('<div class="rail-title">Credit Risk<br>Early Warning</div>', unsafe_allow_html=True)
		st.markdown('<div class="rail-rule"></div>', unsafe_allow_html=True)
		st.markdown("<div class='rail-section-label'>NEWS SOURCE</div>", unsafe_allow_html=True)
		selected_source = st.selectbox("News source", list(NEWS_PATHS), key="news_source")
		st.caption(f"Manual file: {NEWS_PATHS[selected_source].relative_to(PROJECT_ROOT)}")
		with st.expander("Manually add news"):
			with st.form("manual_news_form", clear_on_submit=True):
				headline = st.text_input("Headline *")
				if selected_source == "CSV News":
					source_name = st.text_input("Source")
					published = st.text_input("Date")
					description = ""
				else:
					description = st.text_area("Description")
					source_name = st.text_input("Source")
					published = st.text_input("Published at")
				url = st.text_input("URL")
				submitted = st.form_submit_button("Add news row", use_container_width=True)
			if submitted:
				if not headline.strip():
					st.error("Headline is required.")
				else:
					row = (
						{"headline": headline.strip(), "source": source_name.strip(), "date": published.strip(), "url": url.strip()}
						if selected_source == "CSV News"
						else {"headline": headline.strip(), "description": description.strip(), "source": source_name.strip(), "published_at": published.strip(), "url": url.strip()}
					)
					try:
						created_id = append_news_row(selected_source, row)
						st.success(f"News row saved with ID {created_id}. Run the agents manually to refresh analysis.")
					except OSError:
						st.error("Could not write to the selected news CSV.")
		st.markdown("<div class='rail-section-label'>OLLAMA STATUS</div>", unsafe_allow_html=True)
		if ollama_is_up():
			st.success("Ollama is up")
		else:
			st.error("Ollama is down")
		st.markdown('<div class="rail-rule"></div>', unsafe_allow_html=True)
		page = st.radio("Workspace", PAGE_NAMES, label_visibility="collapsed", key="page_navigation")
		st.markdown("<div class='rail-rule'></div>", unsafe_allow_html=True)
		if st.button("Refresh Data", use_container_width=True):
			st.cache_data.clear()
			st.rerun()
		st.markdown("<div class='rail-section-label'>DATA STATUS</div>", unsafe_allow_html=True)
		if missing:
			st.markdown(f"<div class='rail-status warning'>{len(missing)} input file(s) missing</div>", unsafe_allow_html=True)
		else:
			st.markdown("<div class='rail-status ready'>4 JSON sources available</div>", unsafe_allow_html=True)
		st.caption("Manual data entry · no agents run automatically")
	return page, selected_source


def render_news_feed(source: str) -> None:
	"""Display rows currently stored in the selected manual news CSV."""
	path = NEWS_PATHS[source]
	st.markdown('<div class="section-eyebrow">MANUAL INPUT / NEWS</div>', unsafe_allow_html=True)
	st.title("News Feed")
	st.caption(f"Showing manually maintained rows from {path.relative_to(PROJECT_ROOT)}. No agents are run here.")
	if not path.exists():
		st.info("This news file does not exist yet. Add a row using the sidebar form.")
		return
	try:
		with path.open(encoding="utf-8-sig", newline="") as news_file:
			rows = list(csv.DictReader(news_file))
	except (OSError, UnicodeError, csv.Error) as error:
		st.error(f"Could not read the selected news file: {error}")
		return
	if not rows:
		st.info("No news rows are available. Add one using the sidebar form.")
		return
	st.dataframe(rows, use_container_width=True, hide_index=True)


def render_dashboard(events: list[dict[str, Any]], impacts: list[dict[str, Any]], briefs: list[dict[str, Any]], agent1: list[dict[str, Any]]) -> None:
	"""Page 1: portfolio-level KPIs and HIGH-priority event cards."""
	relevant_count = sum(row.get("relevant") is True for row in agent1)
	high_events = [event for event in events if event_priority(event) == "HIGH"]
	impacted_clients = {
		str(row.get("company", ""))
		for row in impacts
		if row.get("potentially_impacted") is True and not str(row.get("reason", "")).startswith("ERROR:")
	}
	investigation_count = sum(
		len(values_list(row.get("analyst_questions"))) + len(values_list(row.get("next_investigation_steps")))
		for row in briefs
	)

	st.markdown('<div class="section-eyebrow">OVERVIEW / LIVE FILE SNAPSHOT</div>', unsafe_allow_html=True)
	st.title("Executive Dashboard")
	metrics = st.columns(4, gap="medium")
	metric_values = [
		("News Events Screened", len(agent1), f"{relevant_count} potentially relevant", "#11796f"),
		("High Priority Events", len(high_events), "flagged for analyst research", "#c48837"),
		("Potentially Impacted Clients", len(impacted_clients), "unique companies in Agent 3", "#bd604e"),
		("Analyst Investigation Items", investigation_count, "questions and next steps in briefs", "#566d8c"),
	]
	for column, metric in zip(metrics, metric_values):
		with column:
			render_metric(*metric)

	st.markdown("<div class='section-heading-row'><div><div class='section-eyebrow'>PRIORITY QUEUE</div><h2>Priority Events</h2></div><div class='section-note'>HIGH priority · select an event to inspect</div></div>", unsafe_allow_html=True)
	if not high_events:
		st.info("No HIGH priority events are available in Agent 2 results.")
	else:
		for start in range(0, len(high_events), 2):
			columns = st.columns(2, gap="medium")
			for offset, (column, event) in enumerate(zip(columns, high_events[start:start + 2])):
				with column:
					render_event_card(event, start + offset, briefs)


def selected_event_control(events: list[dict[str, Any]]) -> dict[str, Any] | None:
	"""Render a stable event selector shared by the two analysis pages."""
	if not events:
		st.info("No event records are available yet. Add Agent 1/2 JSON outputs and refresh.")
		return None
	event_ids = [event["event_id"] for event in events]
	selected = st.session_state.get("selected_event_id")
	if selected not in event_ids:
		selected = event_ids[0]
	if st.session_state.get("event_selector") not in event_ids:
		st.session_state["event_selector"] = selected
	chosen_id = st.selectbox(
		"Select event",
		event_ids,
		format_func=lambda event_id: event_label(next(event for event in events if event["event_id"] == event_id)),
		key="event_selector",
	)
	st.session_state["selected_event_id"] = chosen_id
	return next(event for event in events if event["event_id"] == chosen_id)


def render_event_analysis(events: list[dict[str, Any]], briefs: list[dict[str, Any]]) -> None:
	"""Page 2: market context and Agent 2 prioritization for one selected event."""
	st.markdown('<div class="section-eyebrow">SIGNAL REVIEW / AGENTS 1 + 2</div>', unsafe_allow_html=True)
	st.title("Event Analysis")
	event = selected_event_control(events)
	if event is None:
		return
	detection = event["detection"]
	research = event["research"]
	priority = event_priority(event)
	metadata = event_metadata(event, briefs)
	st.markdown(
		f"<div class='detail-banner'><span class='priority-pill priority-{priority_tone(priority)}'>{html.escape(priority)} PRIORITY</span>"
		f"<span class='banner-title'>{html.escape(display_event_title(event))}</span>"
		f"<span class='event-id'>EVENT #{html.escape(event['event_id'])}</span></div>",
		unsafe_allow_html=True,
	)
	left, right = st.columns([1.4, 1], gap="large")
	with left:
		st.markdown("### 01 / Market Event")
		st.write(event_summary(event))
		st.markdown("### 02 / Why It Matters")
		st.write(priority_rationale(event))
		st.markdown("### 03 / Background")
		background = detection.get("cause") or research.get("background")
		st.write(text(background))
	with right:
		st.markdown("### 04 / Priority")
		priority_class = priority.lower() if priority in {"HIGH", "MEDIUM", "LOW"} else "neutral"
		st.markdown(f"<div class='priority-panel priority-{priority_class}'>{html.escape(priority)}</div>", unsafe_allow_html=True)
		st.markdown("### 05 / Potentially affected industries")
		industries = event_industries(event)
		st.markdown("".join(f"<span class='industry-chip'>{html.escape(str(item))}</span>" for item in industries) or "Not available in POC data", unsafe_allow_html=True)
		st.caption(f"Agent 1 signal confidence: {safe_number(detection.get('confidence')):.0%}")
		st.markdown("### Event Provenance")
		st.caption(f"Source: {metadata['source']}")
		st.caption(f"Published: {metadata['published']}")
		st.caption(f"Detected: {metadata['detected']}")


def render_impact_chain(
	event: dict[str, Any],
	impact: dict[str, Any],
	brief: dict[str, Any] | None,
) -> None:
	"""Show how the supplied event analysis could pass through to investigation."""
	detection = event.get("detection", {})
	market_effect = {
		"up": "Agent 1 indicates upward crude-price direction; potential market effect, not verified.",
		"down": "Agent 1 indicates downward crude-price direction; potential market effect, not verified.",
		"mixed": "Agent 1 indicates a mixed price direction; market effect requires validation.",
	}.get(str(detection.get("price_direction", "")).lower())
	if not market_effect:
		market_effect = text(detection.get("cause") or event.get("research", {}).get("significance"), "Market effect not available in POC data")
		if market_effect != "Market effect not available in POC data":
			market_effect = f"Potential market effect requiring validation: {market_effect}"

	industries = event_industries(event)
	company_industry = str(impact.get("industry", ""))
	if company_industry and company_industry not in industries:
		industries.append(company_industry)
	client = text(impact.get("company"), "Client not available in POC data")
	if impact.get("business_area"):
		client += f" · Business area: {impact['business_area']}"

	financial_areas = values_list(impact.get("potential_financial_impact"))
	if not financial_areas and brief:
		financial_areas = values_list(brief.get("potential_financial_areas"))
	financial_text = joined(
		[f"Potential area to investigate: {area}" if not area.lower().startswith("potential") else area for area in financial_areas],
		"No financial areas available in POC data",
	)
	if not financial_areas:
		financial_text = "No financial areas available in POC data; analyst validation required."

	investigation = values_list(brief.get("next_investigation_steps")) if brief else []
	if not investigation:
		investigation = values_list(impact.get("analyst_investigation_points"))
	if brief:
		investigation.extend(values_list(brief.get("analyst_questions")))
	investigation = list(dict.fromkeys(investigation))
	investigation_text = joined(investigation, "No investigation steps available in POC data")

	chain = [
		("MARKET EVENT", event_summary(event)),
		("MARKET EFFECT", market_effect),
		("AFFECTED INDUSTRIES", joined(industries, "Not available in POC data")),
		("CLIENT EXPOSURE", f"Potential exposure: {client}"),
		("POTENTIAL FINANCIAL TRANSMISSION", financial_text),
		("ANALYST INVESTIGATION", investigation_text),
	]
	st.markdown("## IMPACT CHAIN")
	for index, (label, detail) in enumerate(chain):
		st.markdown(
			f"<div class='chain-node'><div class='chain-label'>{html.escape(label)}</div>"
			f"<div class='chain-detail'>{html.escape(detail)}</div></div>",
			unsafe_allow_html=True,
		)
		if index < len(chain) - 1:
			st.markdown("<div class='chain-arrow' aria-hidden='true'>↓</div>", unsafe_allow_html=True)


def render_client_impact(
	events: list[dict[str, Any]],
	impacts: list[dict[str, Any]],
	briefs: list[dict[str, Any]],
) -> None:
	"""Page 3: company-level potential exposure and financial transmission."""
	st.markdown('<div class="section-eyebrow">PORTFOLIO EXPOSURE / AGENT 3</div>', unsafe_allow_html=True)
	st.title("Client Impact")
	st.markdown("### SUPPORTED ENTITIES")
	companies = load_companies()
	if companies:
		for company in companies:
			st.markdown(
				f"**{html.escape(text(company.get('company'), 'Unnamed entity'))}** · "
				f"{html.escape(text(company.get('industry'), 'Industry not specified'))}  \n"
				f"{html.escape(text(company.get('business'), 'Business not specified'))}"
			)
	else:
		st.caption("No supported entities added yet.")

	with st.expander("Add supported entity"):
		with st.form("supported_entity_form", clear_on_submit=True):
			company_name = st.text_input("Company name *")
			industry = st.text_input("Industry *")
			business = st.text_input("Business description *")
			add_company = st.form_submit_button("Add entity")
		if add_company:
			if not all(value.strip() for value in (company_name, industry, business)):
				st.error("Company name, Industry, and Business description are required.")
			else:
				companies.append({"company": company_name.strip(), "industry": industry.strip(), "business": business.strip()})
				try:
					COMPANIES_FILE.parent.mkdir(parents=True, exist_ok=True)
					COMPANIES_FILE.write_text(json.dumps(companies, ensure_ascii=False, indent=2), encoding="utf-8")
					st.success("Supported entity saved.")
					st.rerun()
				except OSError:
					st.error("Could not save the supported entity.")

	assessed_event_ids = {
		str(row.get("event_id", ""))
		for row in impacts
		if row.get("event_id") not in (None, "")
	}
	impact_events = [event for event in events if str(event.get("event_id", "")) in assessed_event_ids]
	if not impact_events:
		st.info("No company impact assessments are currently available.")
		return
	event = selected_event_control(impact_events)
	if event is None:
		return
	event_id = event["event_id"]
	event_impacts = [row for row in impacts if str(row.get("event_id", "")) == event_id]
	if not event_impacts:
		st.info("No Agent 3 company assessments are available for this event.")
		return

	st.caption("Potential exposure hypotheses only · Requires analyst validation")
	columns = st.columns(min(3, len(event_impacts)), gap="medium")
	for index, impact in enumerate(event_impacts):
		with columns[index % len(columns)]:
			is_error = str(impact.get("reason", "")).startswith("ERROR:")
			is_impacted = impact.get("potentially_impacted") is True and not is_error
			badge = "Assessment error" if is_error else ("Potentially impacted" if is_impacted else "No material exposure")
			badge_class = "badge-error" if is_error else ("badge-impact" if is_impacted else "badge-neutral")
			st.markdown(
				f"<div class='client-card'><div class='client-card-top'><span class='impact-badge {badge_class}'>{badge}</span></div>"
				f"<h3>{html.escape(text(impact.get('company')))}</h3><div class='client-industry'>{html.escape(text(impact.get('industry')))}</div></div>",
				unsafe_allow_html=True,
			)
			st.markdown(f"**Potentially Impacted**  \n{'Yes' if is_impacted else ('Error' if is_error else 'No')}")
			st.markdown(f"**Impact Direction**  \n{text(impact.get('impact_direction'), 'Not available in POC data')}")
			st.markdown(f"**Impact Mechanism**  \n{text(impact.get('impact_mechanism'), 'Not available in POC data')}")
			st.markdown(f"**Business Area**  \n{text(impact.get('business_area'), 'Not available in POC data')}")
			list_block("Potential Financial Impact", impact.get("potential_financial_impact"))
			confidence = safe_number(impact.get("confidence"))
			st.progress(confidence, text=f"Confidence · {confidence:.0%}")
			st.markdown("### WHY THIS COMPANY MAY BE EXPOSED")
			st.write(text(impact.get("reason"), "Reason not available in POC data"))
			st.caption(f"Impact mechanism: {text(impact.get('impact_mechanism'), 'Not available in POC data')}")
			st.markdown("### POTENTIAL FINANCIAL TRANSMISSION")
			areas = values_list(impact.get("potential_financial_impact"))
			if areas:
				for area in areas:
					st.markdown(f"- Potential area to investigate: {text(area)}")
			else:
				st.caption("No financial areas available in POC data")

	company_names = [str(row.get("company", "")) for row in event_impacts]
	selected_company = st.selectbox("Impact chain company", company_names, key=f"impact_company_{event_id}")
	selected_impact = next(row for row in event_impacts if str(row.get("company", "")) == selected_company)
	brief = next(
		(row for row in briefs if row.get("company") == selected_company and row.get("event") == (event["research"].get("event_type") or event["detection"].get("event_type"))),
		None,
	)
	render_impact_chain(event, selected_impact, brief)


def render_analyst_copilot(
	briefs: list[dict[str, Any]],
	events: list[dict[str, Any]],
	impacts: list[dict[str, Any]],
) -> None:
	"""Page 4: analyst brief, evidence trace, and clear human handoff."""
	st.markdown('<div class="section-eyebrow">HUMAN-IN-THE-LOOP / AGENT 4</div>', unsafe_allow_html=True)
	st.title("Analyst Copilot")
	if not briefs:
		st.info("No Agent 4 analyst briefs are available yet.")
		return
	labels = [f"{brief.get('company', 'Company')} · {brief.get('event', 'Event')} · {brief.get('priority', 'Priority')}" for brief in briefs]
	selected_index = st.selectbox("Select analyst brief", range(len(briefs)), format_func=lambda index: labels[index])
	brief = briefs[selected_index]
	matching_event = next(
		(
			event for event in events
			if str(event.get("research", {}).get("event_type", "")).casefold() == str(brief.get("event", "")).casefold()
			or display_event_title(event).casefold() == str(brief.get("event", "")).casefold()
		),
		None,
	)
	matching_impact = next(
		(
			impact for impact in impacts
			if impact.get("company") == brief.get("company")
			and (matching_event is None or str(impact.get("event_id", "")) == matching_event["event_id"])
		),
		None,
	)
	potential_identified = bool(matching_impact and matching_impact.get("potentially_impacted") is True)
	assessment_label = "Potential impact identified" if potential_identified else "Exposure assessment requires review"
	st.markdown("### AI ASSESSMENT")
	st.markdown(
		f"<div class='assessment-banner'><span class='assessment-mark'>◈</span>"
		f"<span>{html.escape(assessment_label)}</span></div>",
		unsafe_allow_html=True,
	)
	st.warning("⚠ HUMAN REVIEW REQUIRED")
	st.caption("AI-generated assessment for analyst investigation. It does not determine or recommend a credit-rating action.")
	st.markdown(
		f"<div class='brief-header'><span class='priority-pill priority-{priority_tone(brief.get('priority'))}'>{html.escape(text(brief.get('priority')))} PRIORITY</span>"
		f"<span class='banner-title'>{html.escape(text(brief.get('company')))}</span>"
		f"<span class='brief-event'>{html.escape(text(brief.get('event')))}</span></div>",
		unsafe_allow_html=True,
	)

	st.markdown("### EXECUTIVE SUMMARY")
	st.write(text(brief.get("executive_summary"), "Not available in POC data"))
	left, right = st.columns(2, gap="large")
	with left:
		st.markdown("### MARKET EVENT")
		st.write(text(brief.get("market_event"), "Not available in POC data"))
		st.markdown("### COMPANY EXPOSURE")
		st.write(text(brief.get("company_exposure"), "Not available in POC data"))
	with right:
		st.markdown("### POTENTIAL IMPACT MECHANISM")
		st.write(text(brief.get("potential_impact_mechanism"), "Not available in POC data"))
		list_block("POTENTIAL FINANCIAL AREAS", brief.get("potential_financial_areas"))
	evidence, uncertainties = st.columns(2, gap="large")
	with evidence:
		list_block("KNOWN EVIDENCE", brief.get("known_evidence"))
	with uncertainties:
		list_block("UNCERTAINTIES", brief.get("uncertainties"))
	questions, steps = st.columns(2, gap="medium")
	with questions:
		st.markdown("<div class='action-panel'><div class='action-kicker'>ANALYST REVIEW</div><h3>ANALYST QUESTIONS</h3>", unsafe_allow_html=True)
		questions_list = values_list(brief.get("analyst_questions"))
		if questions_list:
			for question in questions_list:
				st.markdown(f"<div class='action-item'>{html.escape(question)}</div>", unsafe_allow_html=True)
		else:
			st.markdown("<div class='action-item'>Not available in POC data</div>", unsafe_allow_html=True)
		st.markdown("</div>", unsafe_allow_html=True)
	with steps:
		st.markdown("<div class='action-panel next-steps'><div class='action-kicker'>FOLLOW-UP</div><h3>NEXT INVESTIGATION STEPS</h3>", unsafe_allow_html=True)
		steps_list = values_list(brief.get("next_investigation_steps"))
		if steps_list:
			for step in steps_list:
				st.markdown(f"<div class='action-item'>{html.escape(step)}</div>", unsafe_allow_html=True)
		else:
			st.markdown("<div class='action-item'>Not available in POC data</div>", unsafe_allow_html=True)
		st.markdown("</div>", unsafe_allow_html=True)

	# Show exactly which upstream outputs support the selected handoff.
	st.markdown("## AI ANALYSIS TRACE")
	if matching_event is None:
		st.info("Matching Agent 1/2 event details are not available in POC data.")
	else:
		detection = matching_event.get("detection", {})
		research = matching_event.get("research", {})
		trace = [
			("AGENT 1 · EVENT DETECTED", detection.get("event_summary")),
			("AGENT 2 · RESEARCH & PRIORITY", research.get("reason_for_priority") or research.get("significance")),
			("AGENT 3 · CLIENT EXPOSURE", (matching_impact or {}).get("reason")),
			("AGENT 4 · ANALYST BRIEF", brief.get("executive_summary")),
		]
		trace_columns = st.columns(4, gap="small")
		for column, (label, conclusion) in zip(trace_columns, trace):
			with column:
				st.markdown(
					f"<div class='trace-card'><div class='trace-label'>{html.escape(label)}</div>"
					f"<div class='trace-conclusion'>{html.escape(text(conclusion, 'Not available in POC data'))}</div></div>",
					unsafe_allow_html=True,
				)

	st.markdown("## HUMAN-IN-THE-LOOP HANDOFF")
	handoff = [
		"AI DETECTION",
		"AI RESEARCH",
		"AI CLIENT IMPACT ASSESSMENT",
		"⚠ HUMAN ANALYST REVIEW",
		"DETAILED CREDIT ANALYSIS",
	]
	handoff_columns = st.columns(len(handoff), gap="small")
	for index, (column, stage) in enumerate(zip(handoff_columns, handoff)):
		with column:
			style = "handoff-human" if "HUMAN" in stage else "handoff-stage"
			st.markdown(f"<div class='{style}'>{html.escape(stage)}</div>", unsafe_allow_html=True)
			if index < len(handoff) - 1:
				st.markdown("<div class='handoff-arrow'>↓</div>", unsafe_allow_html=True)
	st.info("AI identifies potential risk signals; the rating analyst performs the final assessment.")


def main() -> None:
	"""Load read-only JSON data and render the selected dashboard page."""
	# Establish a restrained institutional palette with clear risk-status accents.
	st.markdown(
		"""
		<style>
		:root { --ink:#1c2b32; --muted:#617078; --line:#d9e1e2; --paper:#f3f6f5; --white:#ffffff; --teal:#116b61; --gold:#c58b3d; --rust:#aa5145; }
		.stApp { background:var(--paper); color:var(--ink); font-family:'Aptos','Segoe UI',sans-serif; }
		[data-testid="stHeader"] { background:transparent; }
		.block-container { max-width:1440px; padding-top:2.1rem; padding-bottom:3rem; }
		h1,h2,h3 { color:var(--ink); font-family:Georgia,'Times New Roman',serif; letter-spacing:0; }
		h1 { font-size:2rem; margin-bottom:.15rem; }
		h2 { font-size:1.45rem; margin:0; }
		h3 { font-size:1.05rem; }
		[data-testid="stSidebar"] { background:#172d32; border-right:1px solid #294247; }
		[data-testid="stSidebar"] * { color:#eef4f1; }
		[data-testid="stSidebar"] [data-testid="stRadio"] label { font-size:.94rem; }
		[data-testid="stSidebar"] input,
		[data-testid="stSidebar"] textarea,
		[data-testid="stSidebar"] [data-baseweb="select"] > div {
			background:#f8fbfa !important; color:#172d32 !important; -webkit-text-fill-color:#172d32 !important;
			border-color:#aabbbb !important;
		}
		[data-testid="stSidebar"] input::placeholder,
		[data-testid="stSidebar"] textarea::placeholder { color:#58696e !important; -webkit-text-fill-color:#58696e !important; opacity:1; }
		[data-testid="stSidebar"] .stButton button { background:#c78e42; color:#152a2e; border:0; font-weight:700; }
		.rail-kicker,.section-eyebrow,.rail-section-label,.action-kicker { font-size:.68rem; letter-spacing:.09em; font-weight:700; }
		.rail-kicker { color:#b8c8c2; margin:.25rem 0 .55rem; }
		.rail-title { color:white; font-family:Georgia,'Times New Roman',serif; font-size:1.45rem; line-height:1.18; }
		.rail-rule { height:1px; background:#395056; margin:1.1rem 0; }
		.rail-section-label { color:#9eb3ac; margin:1.1rem 0 .4rem; }
		.rail-status { border-radius:4px; padding:.45rem .6rem; font-size:.78rem; }
		.rail-status.ready { background:#244841; color:#d9f2e4; }
		.rail-status.warning { background:#59452e; color:#ffe6bc; }
		.section-eyebrow { color:var(--teal); margin:0 0 .35rem; }
		.metric-card,.event-card,.client-card { background:var(--white); border:1px solid var(--line); border-radius:6px; }
		.metric-card { min-height:123px; padding:1rem 1.05rem .85rem; border-top:3px solid var(--metric-accent); }
		.metric-label { color:var(--muted); font-size:.78rem; font-weight:650; }
		.metric-value { color:var(--ink); font-size:2rem; font-weight:750; line-height:1.25; margin:.2rem 0; }
		.metric-note,.section-note,.event-id { color:var(--muted); font-size:.73rem; }
		.section-heading-row { display:flex; align-items:end; justify-content:space-between; margin:2.1rem 0 .8rem; }
		.event-card { min-height:245px; padding:1rem 1.05rem; }
		.event-card-top,.detail-banner,.brief-header { display:flex; align-items:center; gap:.65rem; flex-wrap:wrap; }
		.priority-pill { display:inline-block; border-radius:3px; background:#f6e9d4; color:#80551e; padding:.22rem .42rem; font-size:.65rem; font-weight:800; letter-spacing:.04em; }
		.priority-pill.priority-high { background:#f5e2dc; color:#913f32; }
		.priority-pill.priority-medium { background:#f7edd8; color:#805b25; }
		.priority-pill.priority-low { background:#e2eee7; color:#356b52; }
		.priority-pill.priority-not-available,.priority-pill.priority-neutral { background:#e8edee; color:#59676d; }
		.event-card h3 { margin:.8rem 0 .4rem; }
		.event-card p { color:#37464d; font-size:.92rem; min-height:2.7rem; }
		.event-industries { border-top:1px solid var(--line); padding-top:.65rem; color:#526168; font-size:.8rem; }
		.event-metadata { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.3rem .65rem; padding:.65rem 0; color:#53636a; font-size:.72rem; }
		.event-metadata b { display:block; color:#26383e; font-size:.68rem; text-transform:uppercase; }
		.priority-rationale { border-top:1px solid var(--line); padding-top:.6rem; color:#405056; font-size:.78rem; line-height:1.45; }
		.priority-rationale b { color:#25383d; }
		.stButton button { border-radius:4px; border:1px solid #b8c7c7; color:var(--teal); font-weight:650; }
		.detail-banner,.brief-header { background:#fff; border:1px solid var(--line); border-radius:6px; padding:.85rem 1rem; margin:.5rem 0 1.2rem; }
		.banner-title { color:var(--ink); font-family:Georgia,'Times New Roman',serif; font-size:1.18rem; }
		.priority-panel { font-size:1.2rem; font-weight:750; border-left:4px solid var(--gold); background:#fff; padding:.65rem .85rem; }
		.priority-high { color:#8a4e32; border-left-color:#b65e46; }
		.priority-medium { color:#855e29; border-left-color:#c58b3d; }
		.priority-low { color:#3e6d61; border-left-color:#588b76; }
		.priority-neutral { color:#5c6a70; border-left-color:#9ba9ab; }
		.industry-chip { display:inline-block; background:#e1ece8; color:#255950; border-radius:3px; padding:.3rem .48rem; margin:.1rem .25rem .25rem 0; font-size:.78rem; }
		.client-card { min-height:105px; padding:.85rem; margin-bottom:.75rem; }
		.client-card h3 { margin:.7rem 0 .1rem; }
		.client-industry { color:var(--muted); font-size:.83rem; }
		.impact-badge { display:inline-block; border-radius:3px; padding:.25rem .45rem; font-size:.67rem; font-weight:750; }
		.badge-impact { background:#f5e5dc; color:#8c4638; }
		.badge-neutral { background:#dcece4; color:#356b52; }
		.badge-error { background:#f4dedd; color:#8e3531; }
		.brief-event { color:var(--muted); font-size:.85rem; }
		.action-panel { min-height:220px; background:#fff; border:1px solid var(--line); border-left:4px solid var(--teal); border-radius:6px; padding:1rem 1.05rem; }
		.action-panel.next-steps { border-left-color:var(--gold); }
		.action-panel h3 { margin:.25rem 0 .9rem; }
		.action-kicker { color:var(--teal); }
		.next-steps .action-kicker { color:#986725; }
		.action-item { padding:.55rem 0; border-top:1px solid #e9eeee; color:#35454c; font-size:.9rem; }
		.brand-subtitle { color:#2d7068; font-size:1rem; font-weight:700; margin:.2rem 0 .45rem; }
		.brand-description { color:#586970; max-width:900px; margin:0 0 1.2rem; line-height:1.55; }
		.assessment-banner { display:flex; align-items:center; gap:.7rem; padding:.85rem 1rem; background:#e4efea; border:1px solid #c5d8d0; border-left:4px solid #287256; border-radius:5px; color:#24543f; font-weight:750; margin:.35rem 0 .6rem; }
		.assessment-mark { font-size:1.25rem; }
		.chain-node { background:#fff; border:1px solid var(--line); border-left:4px solid #39786f; border-radius:5px; padding:.75rem .9rem; }
		.chain-label { color:#2a645b; font-size:.68rem; font-weight:800; letter-spacing:.07em; margin-bottom:.25rem; }
		.chain-detail { color:#35454c; font-size:.88rem; line-height:1.45; }
		.chain-arrow { color:#b08038; text-align:center; font-size:1.15rem; font-weight:800; line-height:1.5; }
		.trace-card { height:100%; min-height:142px; background:#fff; border:1px solid var(--line); border-top:3px solid #57857e; border-radius:5px; padding:.75rem; }
		.trace-label { color:#2c645c; font-size:.66rem; font-weight:800; letter-spacing:.04em; margin-bottom:.45rem; }
		.trace-conclusion { color:#415158; font-size:.78rem; line-height:1.4; }
		.handoff-stage,.handoff-human { min-height:66px; display:flex; align-items:center; justify-content:center; text-align:center; padding:.55rem; border-radius:4px; font-size:.72rem; font-weight:800; }
		.handoff-stage { background:#e4ecea; color:#285e55; border:1px solid #ccdbd7; }
		.handoff-human { background:#f7e8db; color:#8a4935; border:1px solid #e8cbb8; }
		.handoff-arrow { color:#78868a; text-align:center; padding:.25rem 0; font-size:.78rem; }
		[data-testid="stMetricValue"] { color:var(--ink); }
		@media (max-width: 700px) { .block-container { padding:1.2rem 1rem 2rem; } .section-heading-row { display:block; } .metric-card { min-height:112px; } }
		@media (max-width: 760px) { .event-metadata { grid-template-columns:1fr; } .trace-card { min-height:unset; margin-bottom:.4rem; } }
		</style>
		""",
		unsafe_allow_html=True,
	)

	st.markdown("# AI-Powered Credit Risk Early Warning System")
	st.markdown("<div class='brand-subtitle'>Detect → Prioritize → Map → Investigate</div>", unsafe_allow_html=True)
	st.markdown(
		"<div class='brand-description'>Identify potentially material market events, connect them to exposed clients, "
		"and help rating analysts prioritize where deeper investigation should begin.</div>",
		unsafe_allow_html=True,
	)
	# Read each requested JSON source independently, so one missing file does not crash the app.
	loaded = {name: load_json_file(str(path)) for name, path in DATA_FILES.items()}
	missing_or_invalid = [error for _, error in loaded.values() if error]
	if missing_or_invalid:
		for message in missing_or_invalid:
			st.warning(message)

	data = {name: as_records(value) for name, (value, _) in loaded.items()}
	events = build_events(data["Agent 1"], data["Agent 2"])
	page, selected_source = render_sidebar(missing_or_invalid)
	st.caption(f"Manual news source selected: {selected_source} · {NEWS_PATHS[selected_source].relative_to(PROJECT_ROOT)}")

	if page == "Executive Dashboard":
		render_dashboard(events, data["Agent 3"], data["Agent 4"], data["Agent 1"])
	elif page == "News Feed":
		render_news_feed(selected_source)
	elif page == "Event Analysis":
		render_event_analysis(events, data["Agent 4"])
	elif page == "Client Impact":
		render_client_impact(events, data["Agent 3"], data["Agent 4"])
	else:
		render_analyst_copilot(data["Agent 4"], events, data["Agent 3"])

	st.markdown("<div class='rail-rule'></div>", unsafe_allow_html=True)
	st.caption("AI identifies potential risk signals; the rating analyst performs the final assessment.")


main()
