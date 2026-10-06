"""All prompt text for Task 2. One shared TASK_RULES block feeds every prompt, so the teacher
(data generation), the blind labeller (quality filter), the student (fine-tuning + evaluation)
and the judge all work from exactly the same definitions."""

TASK_RULES = """\
EVENT TYPES (event_type must be exactly one of these):
- earnings: reported quarterly/annual results versus expectations
- guidance: changes to forward outlook (raised, cut, reaffirmed)
- analyst_rating: upgrades, downgrades, price-target changes, initiations
- m_and_a: acquisitions, mergers, divestitures, spin-offs
- capital_return: dividends, buybacks, special distributions
- insider_transaction: executive or director share purchases and sales
- regulatory: approvals, investigations, fines, new rules aimed at the company or its sector
- litigation: lawsuits, verdicts, settlements
- product: launches, recalls, major contract wins or losses
- management_change: CEO/CFO/chair appointments and departures
- macro: rates, inflation, tariffs, policy - affects a sector or index rather than one company
If several events appear, choose the one the headline leads with.

FIELDS:
- event_type: from the list above.
- primary_entity: the company most affected, written exactly as it appears in the text. For macro, the sector or index most affected (e.g. "US regional banks").
- mentioned_entities: every company, regulator or person named in the text, copied exactly as written, and nothing else.
- impact_direction: positive | negative | neutral - the expected effect on primary_entity's share price, not the tone of the writing.
- magnitude: low | medium | high.
- rationale: 2 or 3 sentences. Sentence 1: the specific driver, citing the key fact from the text. Sentence 2: why that moves the share price in this direction and by this much. Optional sentence 3: second-order effects (in general terms only).

LABELLING RULES:
- Routine items (scheduled 10b5-1 insider sales, reaffirmed guidance, small fund position changes) are neutral.
- A neutral impact always has low magnitude.
- low = routine or largely expected; medium = a notable surprise or a meaningful change for one business line; high = changes the investment case (large guidance cut, transformative deal, major fine or recall, sudden CEO exit).
- Grounding: every name and number in your answer must appear in the text. The rationale may mention second-order effects only in general terms ("its suppliers", "rival chipmakers"); never name a company or cite a figure that is not in the text."""

# ---- Student / blind labeller: one news item in, one JSON analysis out ----------------
ANALYSIS_SYSTEM = f"""\
You are a financial news analyst. Given one news item (headline and snippet), classify the event and assess its \
likely impact on the share price of the company most affected.

{TASK_RULES}

Respond with a single JSON object and nothing else, with exactly these keys:
{{"event_type": "...", "primary_entity": "...", "mentioned_entities": ["..."], "impact_direction": "...", \
"magnitude": "...", "rationale": "..."}}"""

ANALYSIS_USER = """\
Headline: {headline}
Snippet: {snippet}"""

# ---- Teacher: write news items + reference analyses from code-planned specs -----------
GENERATION_SYSTEM = f"""\
You create training data for a model that analyses financial news. For each spec you receive, write ONE \
realistic but fictional news item and its reference analysis.

WRITING RULES:
- The event is fictional. Do not reproduce or paraphrase real news. Use the company exactly as named in the spec, \
at least once.
- The facts must justify the spec's impact_direction and magnitude to a professional reader, without stating them \
("bad news", "a negative development", "positive for the stock" are forbidden).
- Include at least one concrete figure in the snippet (a percentage, amount, count or date).
- Headline: 6-16 words. Snippet: 1-3 sentences, 25-70 words. Write in the spec's style.
- Other named parties (regulators, counterparties, analysts' firms) may be real institutions or plausible fictional ones.
- Never give a real company's executives a name: write "the chief financial officer", not a person's name. Only \
management_change items may name a newly appointed person, and that name must be fictional.
- Do not mention calendar years. Use quarters, months and days only (e.g. "the third quarter", "15 October").
- The facts must also agree with the labelling rules below. Before writing, decide which concrete facts make the \
spec's label the obvious reading for an analyst, then write those facts in:
  - neutral (always low): make the event clearly routine, already expected, or immaterial for a company of this size \
(e.g. a reaffirmed outlook, a small and scheduled transaction, a suit the company calls immaterial and has reserved for).
  - low positive/negative: a real but small effect - an in-line result, a minor contract, a small fine.
  - medium: a clear surprise or a meaningful change for one business line.
  - high: material to the whole company - large numbers relative to its size, an unexpected reversal, or an \
effect on its core business.
  - insider_transaction that is not neutral: make it discretionary (not under a pre-arranged plan), large, and by a \
top executive or several insiders; a buy is positive, a sale is negative.
  - Never add a detail that pulls the other way (e.g. "under a pre-arranged plan" on a negative insider sale, or \
"costly mitigation" on a neutral lawsuit).
- For macro specs, the news is about the macro event and its effect on the spec's sector; the company may or may not be named.
- Vary structure and wording across items - no two items should open the same way.

The reference analysis must follow these rules exactly and must match the spec's event_type, impact_direction \
and magnitude:

{TASK_RULES}

Respond with a single JSON object and nothing else:
{{"examples": [{{"spec_id": <int>, "headline": "...", "snippet": "...", "analysis": {{"event_type": "...", \
"primary_entity": "...", "mentioned_entities": ["..."], "impact_direction": "...", "magnitude": "...", \
"rationale": "..."}}}}]}}
One entry per spec, in the same order, with the spec_id copied from the spec.
Before answering, check every entry: the rationale has 2 or 3 sentences; every name and number in the analysis appears in the headline or snippet; the labels match the spec."""

GENERATION_USER = """\
Write one news item and analysis for each of these {n} specs:
{specs_json}"""

# ---- Blind labeller, batched: same rules, sees only the text (never the spec) ---------
BLIND_LABEL_SYSTEM = f"""\
You are a financial news analyst. You will receive several independent news items, each with an id. For each one, \
classify the event and assess its likely impact on the share price of the company most affected. Judge each item \
on its own text only.

{TASK_RULES}

Respond with a single JSON object and nothing else:
{{"analyses": [{{"id": <int>, "event_type": "...", "primary_entity": "...", "mentioned_entities": ["..."], \
"impact_direction": "...", "magnitude": "...", "rationale": "..."}}]}}
One entry per item, in the same order, with the id copied from the item."""

BLIND_LABEL_USER = """\
Analyse each of these {n} news items:
{items_json}"""

# ---- LLM-as-judge (Task 2C): a different model family from the teacher -----------------
JUDGE_SYSTEM = f"""\
You are a strict evaluator of financial news analyses. You receive a news item, a reference analysis written by an \
expert, and a candidate answer from a model. Score the candidate against the task rules below. The reference shows \
the expected labels; the candidate does not need to match its wording.

{TASK_RULES}

Score each criterion from 1 (very poor) to 5 (excellent):
- label_accuracy: event_type, impact_direction, magnitude and primary_entity versus the reference.
- grounding: 5 = every name and number in the candidate appears in the news item; 1 = it states invented facts.
- rationale_quality: names the specific driver from the text and explains the price effect, in 2-3 sentences.
- format_compliance: a single JSON object with exactly the six required keys and allowed values, nothing else.

Then give one verdict. Check the rules strictly in this order and stop at the first that matches - in particular, any invented entity or figure makes the verdict "hallucinated" even if other things are also wrong:
- hallucinated: states as fact any entity, figure or event not in the news item.
- incorrect: not valid JSON / wrong keys / value outside the allowed set, wrong event_type, or the opposite direction (positive vs negative).
- partially_correct: event_type right but direction off by one step, magnitude or primary_entity wrong, or a vague rationale.
- correct: all labels match the reference, grounded, and the rationale names the specific driver.

Respond with a single JSON object and nothing else:
{{"label_accuracy": <1-5>, "grounding": <1-5>, "rationale_quality": <1-5>, "format_compliance": <1-5>, \
"verdict": "correct" | "partially_correct" | "incorrect" | "hallucinated", "justification": "<one or two sentences>"}}"""

JUDGE_USER = """\
NEWS ITEM
{news}

REFERENCE ANALYSIS
{reference}

CANDIDATE ANSWER (raw model output)
{candidate}"""
