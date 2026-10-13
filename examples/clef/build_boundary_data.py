# /// script
# requires-python = ">=3.10"
# dependencies = ["typed-argument-parser>=1.10"]
# ///
"""Build reproducible, counterfactual decision-boundary examples.

This corpus is programmatically authored, not LLM-generated. Public evaluation
examples are never used as generation inputs. Labels follow executable rules.
"""

import collections
import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tap import Tap


class Args(Tap):
    output_dir: str
    seed: int = 261012


TASKS = [
    ("shipment tracking", "track the parcel", "tracking_number", "carrier"),
    ("flight status", "check whether the flight is delayed", "flight_number", "departure_date"),
    ("audio beat analysis", "find the strong beats in the recording", "audio_file", "threshold"),
    ("video scene analysis", "locate the scene changes in the clip", "video_file", "minimum_duration"),
    ("invoice lookup", "retrieve the invoice total", "invoice_id", "account_id"),
    ("inventory lookup", "check how many units are available", "sku", "warehouse"),
    ("reservation lookup", "check the reservation details", "reservation_id", "surname"),
    ("warranty lookup", "check the remaining warranty", "serial_number", "purchase_date"),
    ("weather forecast", "check tomorrow's forecast", "city", "forecast_date"),
    ("exchange rate lookup", "look up the exchange rate", "currency_pair", "rate_date"),
    ("usage lookup", "check usage for the billing period", "account_id", "billing_period"),
    ("appointment availability", "find the open appointment slots", "clinic", "appointment_date"),
    ("subscription lookup", "check when the subscription expires", "subscription_id", "account_id"),
    ("transaction lookup", "inspect the transaction status", "transaction_id", "account_id"),
    ("document text extraction", "read the text from the scanned document", "document_file", "page_number"),
    ("image size inspection", "measure the dimensions of the image", "image_file", "measurement_unit"),
    ("delivery estimate", "check the estimated delivery date", "order_id", "postal_code"),
    ("refund lookup", "check whether the refund was processed", "refund_id", "account_id"),
    ("hotel availability", "check the available hotel rooms", "hotel_id", "stay_date"),
    ("support ticket lookup", "check the ticket's current status", "ticket_id", "workspace"),
    ("access audit", "check the access history", "resource_id", "audit_date"),
    ("backup lookup", "check whether the backup completed", "backup_id", "workspace"),
    ("payment fee estimate", "estimate the payment fee", "amount", "currency"),
    ("document translation", "translate the supplied document", "document_file", "target_language"),
    # Held-out operation families, not just different numeric IDs.
    ("sensor history", "retrieve the sensor readings", "sensor_id", "measurement_date"),
    ("rail departure lookup", "check the train's departure time", "train_number", "departure_date"),
    ("certificate lookup", "check the certificate expiry date", "certificate_id", "domain"),
    ("parking availability", "check how many parking spaces remain", "car_park", "arrival_date"),
    ("medical lab lookup", "retrieve the published laboratory result", "result_id", "patient_id"),
    ("rental quote", "look up the rental price", "equipment_id", "rental_date"),
    ("network latency lookup", "check the recorded network latency", "connection_id", "measurement_date"),
    ("course availability", "check the remaining course places", "course_id", "term"),
]

INTENTS = [
    ("schedule a visit", "Book a new appointment; not merely inspect availability.", "book me an appointment", "show the open appointment times"),
    ("inspect availability", "List open appointment times without booking one.", "which appointment times are available", "reserve an appointment for me"),
    ("change an existing booking", "Move the time of an existing reservation; do not create a new reservation.", "move my current reservation", "create another reservation"),
    ("cancel a booking", "Cancel an existing reservation; do not ask whether cancellation is allowed.", "cancel my existing reservation", "explain the cancellation policy"),
    ("read the account balance", "Read the settled account balance; not pending payments or spending limits.", "what is my settled balance", "what payments are still pending"),
    ("read pending payments", "List unsettled payments; not the settled balance.", "show payments that have not settled", "show my settled account balance"),
    ("change a spending limit", "Request a new spending limit; not read the current limit.", "increase my spending limit", "tell me my existing spending limit"),
    ("inspect a spending limit", "Read the existing limit without changing it.", "what is my spending limit now", "raise my spending limit"),
    ("replace a damaged card", "Replace a physically damaged card; lost cards are different.", "my bent card needs replacing", "my card disappeared"),
    ("report a missing card", "Report a lost or stolen card; damaged cards are different.", "I lost my card", "my card is cracked"),
    ("inspect delivery status", "Read the progress of an already placed order.", "where is my current order", "when would a new order arrive"),
    ("estimate a new delivery", "Estimate delivery before placing an order.", "when would a new order reach me", "track my already placed order"),
    ("change a reminder", "Modify an already existing reminder.", "move my existing reminder to tomorrow", "create a new reminder for tomorrow"),
    ("create a reminder", "Create a new reminder, not change an existing one.", "set up a new reminder", "update my current reminder"),
    ("inspect software version", "Read the installed version without installing anything.", "which software version is installed", "install the newest software version"),
    ("install a software update", "Install a newer version; not merely read the installed version.", "install the newest version", "tell me which version I have"),
    ("change an address", "Update the delivery address, not the billing address.", "change my delivery address", "change my billing address"),
    ("change billing address", "Update the billing address, not the delivery address.", "change the address used on invoices", "change the parcel destination"),
    ("read refund status", "Read the status of a requested refund, not request a new refund.", "has my refund been processed", "start a refund request"),
    ("request a refund", "Start a new refund request.", "request a refund for this payment", "check my existing refund request"),
    ("inspect account name", "Read the profile name, not change it.", "what name is on my profile", "rename my profile"),
    ("rename an account", "Change the profile name.", "change the name shown on my profile", "tell me the name on my profile"),
    ("read an invoice", "Read an already issued invoice, not create one.", "show my existing invoice", "issue a new invoice"),
    ("issue an invoice", "Create a new invoice, not inspect an existing one.", "create a new invoice", "read an invoice already issued"),
    # Validation intent semantics never occur as positive training labels.
    ("inspect certificate", "Read a certificate without renewing it.", "show the current certificate", "renew the certificate"),
    ("renew certificate", "Renew a certificate rather than read it.", "renew my certificate", "show my current certificate"),
    ("check exam registration", "Read existing exam registration, not register for an exam.", "am I already registered for the exam", "register me for the exam"),
    ("register for an exam", "Create a new exam registration.", "register me for the exam", "show my current exam registration"),
    ("read sensor history", "Read previous sensor readings, not reconfigure the sensor.", "show past sensor measurements", "change the sensor sampling rate"),
    ("configure a sensor", "Change sensor configuration, not read previous measurements.", "change the sampling interval", "retrieve old sensor measurements"),
    ("inspect rental", "Read an existing rental, not reserve a new rental.", "show my current equipment rental", "book a new equipment rental"),
    ("reserve rental", "Create a new equipment rental booking.", "book a new equipment rental", "show the rental I already booked"),
]

PRODUCTS = [
    ("headphones", "wireless over-ear headphones", "wired earphones", "replacement ear cushions"),
    ("keyboard", "wireless computer keyboard", "wired computer keyboard", "keyboard wrist rest"),
    ("mouse", "wireless computer mouse", "wired computer mouse", "mouse pad"),
    ("cleaner", "sonic facial cleansing brush", "manual facial cleansing brush", "replacement brush head"),
    ("coffee maker", "electric coffee maker", "manual coffee brewer", "coffee filter"),
    ("camera", "digital still camera", "instant film camera", "camera carrying case"),
    ("printer", "laser printer", "inkjet printer", "replacement toner cartridge"),
    ("lamp", "desk lamp with dimming", "desk lamp without dimming", "replacement lamp shade"),
    ("vacuum", "cordless vacuum cleaner", "corded vacuum cleaner", "replacement vacuum filter"),
    ("charger", "wireless phone charger", "wired phone charger", "charger travel pouch"),
    ("speaker", "portable battery-powered speaker", "mains-powered speaker", "speaker mounting bracket"),
    ("toothbrush", "electric toothbrush", "manual toothbrush", "replacement toothbrush head"),
    ("watch", "digital wristwatch", "analogue wristwatch", "replacement watch strap"),
    ("blender", "electric blender", "manual hand whisk", "replacement blender cup"),
    ("fan", "battery-powered portable fan", "mains-powered fan", "replacement fan blade"),
    ("scale", "digital kitchen scale", "mechanical kitchen scale", "scale protective cover"),
    ("projector", "portable video projector", "desktop video projector", "projector carrying bag"),
    ("shaver", "electric shaver", "manual razor", "replacement shaver foil"),
    ("doorbell", "wireless video doorbell", "wired video doorbell", "doorbell mounting plate"),
    ("thermometer", "digital food thermometer", "analogue food thermometer", "thermometer storage sleeve"),
    ("router", "portable cellular router", "fixed cellular router", "router mounting bracket"),
    ("microphone", "wireless microphone", "wired microphone", "microphone foam cover"),
    ("monitor", "portable computer monitor", "desktop computer monitor", "monitor carrying sleeve"),
    ("iron", "cordless clothes iron", "corded clothes iron", "replacement ironing pad"),
    ("watering timer", "automatic watering timer", "manual watering valve", "timer wall mount"),
    ("bike light", "rechargeable bicycle light", "disposable-battery bicycle light", "bicycle light mounting clip"),
    ("air pump", "electric air pump", "manual air pump", "replacement pump nozzle"),
    ("label maker", "electronic label maker", "manual embossing label maker", "replacement label tape"),
    ("door lock", "electronic door lock", "mechanical door lock", "door lock mounting plate"),
    ("water filter", "portable water filter", "fixed water filter", "replacement water filter cartridge"),
    ("tripod", "lightweight camera tripod", "heavy studio camera tripod", "replacement tripod head"),
    ("kettle", "electric kettle", "stovetop kettle", "kettle storage cover"),
]


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def wrapped(identifier: str, source: str, state: Any, questions: dict, gold: dict, metadata: dict) -> dict:
    targets = {f: {o: float(o == gold[f]) for o in q["criteria"]} for f, q in questions.items()}
    return {"record": {"id": identifier, "state": state, "questions": questions}, "targets": targets, "source": source, "metadata": {**metadata, "generator": "counterfactual-boundaries-v1", "display_only": False}}


def options(rng: random.Random, descriptions: Mapping[str, str]) -> tuple[dict, dict]:
    pairs = list(descriptions.items())
    rng.shuffle(pairs)
    schema = {chr(65 + i): text for i, (_, text) in enumerate(pairs)} if len(pairs) <= 26 else {f"o{i}": text for i, (_, text) in enumerate(pairs)}
    mapping = {key: name for key, (name, _) in zip(schema, pairs, strict=True)}
    return schema, mapping


def tool_group(index: int, split: str, seed: int) -> list[dict]:
    rng = random.Random(f"{seed}:{split}:tools:{index}")
    bank = TASKS[:24] if split == "train" else TASKS[24:]
    task = bank[index % len(bank)]
    operation, request, first, second = task
    prefix = "ticket" if split == "train" else "case"
    values = {first: f"{prefix}-{rng.randrange(100000, 999999)}", second: f"value-{rng.randrange(1000, 9999)}"}
    region = rng.choice(["EU", "US", "APAC"])
    name = f"{operation.replace(' ', '_')}_{rng.randrange(100, 999)}"
    distractor_task = rng.choice([t for t in bank if t != task])
    blocker = index % 8
    group_id = f"boundary-{split}-tool-family-{index:05d}"
    rows = []
    for mode in ("tool_call", "request_for_info", "cannot_answer", "direct"):
        tools = []
        for j in range(rng.choice([0, 3, 7, 15, 31])):
            t = distractor_task if j % 2 == 0 else rng.choice([x for x in bank if x != task])
            tools.append({"name": f"{t[0].replace(' ', '_')}_{j}_{rng.randrange(100, 999)}", "description": f"Supports {t[0]} only. Does not support {operation}.", "parameters": {"type": "object", "properties": {t[2]: {"type": "string"}}, "required": [t[2]]}})
        definition = {
            "name": name,
            "description": f"Supports {operation} for records in {region}. Required parameters must be provided by the user, not guessed. Returns the requested information.",
            "parameters": {"type": "object", "properties": {first: {"type": "string"}, second: {"type": "string"}, "language": {"type": "string", "default": "en"}}, "required": [first, second]},
        }
        available = mode in ("tool_call", "request_for_info")
        reason = ""
        if mode == "cannot_answer":
            reasons = ["no_tools", "wrong_operation", "wrong_region", "read_only_preview", "no_permission", "unsupported_format", "missing_dependency", "wrong_account_scope"]
            reason = reasons[blocker]
            if blocker == 0:
                tools = []
            elif blocker == 1:
                definition["description"] = f"Supports {distractor_task[0]} only; cannot perform {operation}."
            elif blocker == 2:
                definition["description"] = f"Supports {operation} only for records in {'US' if region != 'US' else 'EU'}."
            else:
                constraints = [
                    "",
                    "",
                    "",
                    f"Only previews documentation; cannot execute {operation}.",
                    f"The current account has no permission to use this tool for {operation}.",
                    f"Does not accept the input format supplied in this request for {operation}.",
                    f"Disabled: its mandatory upstream dependency is unavailable, so {operation} cannot be executed.",
                    f"Supports {operation} only in workspace alpha; this request concerns workspace beta.",
                ]
                definition["description"] = constraints[blocker]
            if blocker > 0:
                tools.append(definition)
        elif available:
            tools.append(definition)
        elif mode == "direct" and index % 2:
            tools.append(definition)
        rng.shuffle(tools)
        missing = second if index % 2 else first
        supplied = {k: v for k, v in values.items() if mode != "request_for_info" or k != missing}
        intro = rng.choice(["Could you", "Please", "I need you to", "Would you"])
        utterance = f"{intro} {request} for the record in {region}?"
        if split == "validation":
            utterance = f"For the record in {region}, can you help me {request}?"
        utterance += " " + "; ".join(f"{k} is {v}" for k, v in supplied.items()) + "."
        if reason == "wrong_account_scope":
            utterance += " This is for workspace beta."
        if index % 3 == 0 and mode == "request_for_info":
            # A previous identifier must not be silently reused for a different request.
            history = [{"role": "user", "content": f"An earlier unrelated record used {missing}=old-{rng.randrange(1000, 9999)}."}, {"role": "user", "content": utterance + " The earlier identifier does not apply to this request."}]
        else:
            history = [{"role": "user", "content": utterance}]
        answer = f"The requested {operation} result for {values[first]} is confirmed: reference {rng.randrange(10000, 99999)}."
        evidence = {"source": "validated tool result", "request_parameters": values, "region": region, "result": answer} if mode == "direct" else None
        responses = {"tool_call": json.dumps({"name": name, "arguments": {**values, "language": "en"}}), "request_for_info": f"Please provide {missing} for this request before I proceed.", "cannot_answer": f"I cannot perform {operation} with the tools and permissions available in this request.", "direct": answer}
        if index % 4 == 0:
            responses["wrong_arguments"] = json.dumps({"name": name, "arguments": {**values, first: "previous-unrelated-record", "language": "en"}})
        criteria, mapping = options(rng, responses)
        state = {"tools": tools, "conversation": history}
        if evidence:
            state["prior_result"] = evidence
        instruction = "Which response should the assistant give to the latest user request, given the available tools and evidence? Do not invent tools, parameters, or results."
        if index % 3 == 1:
            # Like CLINC, some tasks put the request in the schema, not the state.
            state.pop("conversation")
            instruction += " User request: " + utterance
        questions = {"next_action": {"type": "choice", "instructions": instruction, "criteria": criteria}}
        label = next(o for o, action in mapping.items() if action == mode)
        metadata = {"family_id": group_id, "mode": mode, "unavailability_reason": reason, "oracle": {"operation": operation, "region": region, "required": [first, second], "values": values, "supplied": supplied, "tool_name": name, "capable": available, "has_evidence": bool(evidence), "action_by_option": mapping}}
        rows.append(wrapped(f"{group_id}-{mode}", "boundary_tool_use", state, questions, {"next_action": label}, metadata))
    return rows


def multiselect(index: int, split: str, seed: int) -> dict:
    rng = random.Random(f"{seed}:{split}:multi:{index}")
    k = rng.choice([4, 8, 12, 16])
    positive_count = (0, 1, k // 2, k)[index % 4]
    positive = set(rng.sample(range(k), positive_count))
    content_kind = (index // 4) % 4
    entity = f"{'Facility' if split == 'train' else 'Depot'}-{rng.randrange(10000, 99999)}"
    other = entity + "-annex"
    facts = []
    passages = []
    questions = {}
    labels = {}
    for j in range(k):
        threshold = rng.randrange(2, 60)
        high = j in positive
        quantity = threshold + rng.randrange(0, 9) if high else max(0, threshold - rng.randrange(1, min(9, threshold) + 1))
        code = f"{'bin' if split == 'train' else 'rack'}-{rng.randrange(1000, 9999)}-{j}"
        phrasing = j % 4
        if phrasing == 0:
            text = f"At {entity}, {code} has {quantity} usable units. {other} has {threshold + 25} in its corresponding location."
        elif phrasing == 1:
            broken = rng.randrange(1, 12)
            text = f"{entity}'s {code} holds {quantity + broken} units, of which {broken} cannot be used."
        elif phrasing == 2:
            text = f"The previous count of {code} at {entity} was {threshold + 40}. The corrected current usable count is {quantity}."
        else:
            incoming = rng.randrange(1, 20)
            text = f"At {entity}, {quantity} units in {code} are usable now. Another {incoming} are due next week and are not part of today's stock."
        if split == "validation":
            text = text.replace("usable units", "units ready for use").replace("cannot be used", "are unusable")
        instruction = f"Candidate: {code} meets the immediate availability requirement at {entity}. This means at least {threshold} usable units CURRENTLY at this facility. Use the corrected count; exclude broken units, future deliveries, and other facilities."
        if content_kind == 1:
            text = (
                f"For import category {code}, {entity} will issue licences for no more than {threshold} tonnes; requests above that ceiling must be refused."
                if high
                else f"For import category {code}, {entity} requires an inspection certificate but does not limit the quantity imported. The quota at {other} does not apply here."
            )
            instruction = f"Candidate: quantitative restriction on {code} at {entity}. A quantitative restriction limits the amount that may be imported; a documentation requirement alone is not a quantitative restriction. Does this concept apply to this document?"
        elif content_kind == 2:
            text = f"With {entity}'s feature {code}, tasks that once took me an hour now take minutes. I would gladly use it again." if high else f"{entity}'s feature {code} kept interrupting my work. I regretted relying on it and stopped using it."
            text += f" Someone else praised the corresponding feature at {other}; that is not my assessment of {entity}."
            instruction = f"Candidate: the reviewer expresses a positive opinion about feature {code} of {entity}. Infer sentiment from the described experience, not the presence of a sentiment word or someone else's opinion."
        elif content_kind == 3:
            text = f"Application {code} at {entity} has {quantity} verified qualifying hours. The rule requires at least {threshold} verified hours. Its identity check passed; the applicant has signed the form. An earlier unverified claim of {quantity + 50} hours does not count."
            instruction = f"Candidate: application {code} is eligible. The conditions are a passed identity check, a signed form, and at least {threshold} VERIFIED qualifying hours. All three conditions must hold."
        passages.append(text)
        field = f"ready_{j}"
        criteria, mapping = options(rng, {"yes": "Yes, this candidate applies.", "no": "No, this candidate does not apply."})
        questions[field] = {"type": "choice", "instructions": "Select all applicable candidates by judging each separately. " + instruction, "criteria": criteria}
        labels[field] = next(o for o, v in mapping.items() if v == ("yes" if high else "no"))
        facts.append({"field": field, "code": code, "threshold": threshold, "usable": quantity, "yes_by_option": {o: v == "yes" for o, v in mapping.items()}, "paragraph": text})
    rng.shuffle(passages)
    state = {"document": "\n".join(passages), "facility": entity}
    return wrapped(f"boundary-{split}-multi-{index:05d}", "boundary_multilabel", state, questions, labels, {"family_id": f"multi-{split}-{index}", "positive_count": positive_count, "content_kind": content_kind, "oracle": facts})


def intent_group(index: int, split: str, seed: int) -> list[dict]:
    rng = random.Random(f"{seed}:{split}:intent:{index}")
    bank = INTENTS[:24] if split == "train" else INTENTS[24:]
    intent = bank[index % len(bank)]
    name, definition, positive, near_miss = intent
    # Exclude the near-miss intent from this catalog. Otherwise OOS would be wrong.
    excluded = {other[0] for other in bank if other[2] == near_miss or other[3] == positive}
    excluded.add(bank[(index % len(bank)) ^ 1][0])
    distractors = [t for t in bank if t != intent and t[0] not in excluded]
    rng.shuffle(distractors)
    catalog = {x[0]: x[1] for x in distractors[: rng.choice([3, 7, 11])]} | {name: definition}
    catalog["out_of_scope"] = "None of the available intent descriptions fits this request. Do not pick a merely related intent."
    rows = []
    for label, utterance in ((name, positive), ("out_of_scope", near_miss)):
        suffix = f". The reference for this request is {'T' if split == 'train' else 'V'}-{rng.randrange(100000, 999999)}."
        phrase = rng.choice(["Can you help me: ", "My request is: ", "Please ", "I want to "])
        if split == "validation":
            phrase = "What I need is to "
        criteria, mapping = options(rng, catalog)
        field = {"intent": {"type": "choice", "instructions": "Choose the exact matching intent, or out of scope. User request: " + phrase + utterance + suffix, "criteria": criteria}}
        state = {} if index % 2 else {"request": phrase + utterance + suffix}
        rows.append(
            wrapped(
                f"boundary-{split}-intent-{index:05d}-{'in' if label == name else 'out'}",
                "boundary_intent",
                state,
                field,
                {"intent": next(o for o, v in mapping.items() if v == label)},
                {"family_id": f"intent-{split}-{index}", "oracle": {"requested_intent": label, "phrase": utterance, "catalog": catalog, "intent_by_option": mapping}},
            )
        )
    return rows


def product_group(index: int, split: str, seed: int) -> list[dict]:
    rng = random.Random(f"{seed}:{split}:product:{index}")
    bank = PRODUCTS[:24] if split == "train" else PRODUCTS[24:]
    category, exact, substitute, accessory = bank[index % len(bank)]
    capacity = rng.randrange(2, 80)
    model = f"Series-{rng.randrange(100, 999)}"
    meanings = {
        "exact": "Exact: a complete product meeting all explicit requirements.",
        "substitute": "Substitute: a complete product for the same primary need, but differing in a requested feature.",
        "complement": "Complement: a compatible accessory for the requested product; it cannot fulfill the primary need by itself.",
        "irrelevant": "Irrelevant: neither a usable product for the primary need nor a compatible accessory.",
    }
    rows = []
    for kind in meanings:
        main = exact if kind == "exact" else substitute if kind == "substitute" else accessory if kind == "complement" else bank[(index + 7) % len(bank)][1]
        if kind == "irrelevant" and index % 2:
            main = accessory
        fits = model if kind != "irrelevant" or index % 2 == 0 else model + "-different"
        description = f"A {main}. Designed for {fits}. Capacity rating: {capacity} units. Stock code {'P' if split == 'train' else 'Z'}-{rng.randrange(100000, 999999)}."
        if kind == "substitute" and index % 2:
            description = f"A {exact}. Designed for {model}. Capacity rating: {capacity - 1} units. Stock code P-{rng.randrange(100000, 999999)}."
        criteria, mapping = options(rng, meanings)
        state = {"search_query": f"I need a {exact} for {model} with capacity of at least {capacity} units, not just a replacement part.", "product_description": description}
        questions = {"relevance": {"type": "choice", "instructions": "Judge the product's relationship to the requested need. Compatible accessories are complements, not exact products; an incompatible accessory is irrelevant.", "criteria": criteria}}
        rows.append(
            wrapped(
                f"boundary-{split}-product-{index:05d}-{kind}",
                "boundary_product_relevance",
                state,
                questions,
                {"relevance": next(o for o, v in mapping.items() if v == kind)},
                {"family_id": f"product-{split}-{index}", "oracle": {"relationship": kind, "option_mapping": mapping, "category": category, "capacity": capacity, "requested_model": model}},
            )
        )
    return rows


def verify_row(row: dict) -> None:
    record = row["record"]
    oracle = row["metadata"]["oracle"]
    source = row["source"]
    expected = {}
    if source == "boundary_tool_use":
        mode = "direct" if oracle["has_evidence"] else "cannot_answer" if not oracle["capable"] else "request_for_info" if not set(oracle["required"]) <= oracle["supplied"].keys() else "tool_call"
        assert mode == row["metadata"]["mode"]
        expected["next_action"] = next(o for o, m in oracle["action_by_option"].items() if m == mode)
        if mode == "tool_call":
            call = json.loads(record["questions"]["next_action"]["criteria"][expected["next_action"]])
            assert call["name"] in {t["name"] for t in record["state"]["tools"]}
            assert all(call["arguments"][k] == v for k, v in oracle["values"].items())
        if mode == "direct":
            assert record["state"]["prior_result"]["result"] == record["questions"]["next_action"]["criteria"][expected["next_action"]]
    elif source == "boundary_multilabel":
        for fact in oracle:
            applies = fact["usable"] >= fact["threshold"]
            assert fact["paragraph"] in record["state"]["document"]
            expected[fact["field"]] = next(o for o, yes in fact["yes_by_option"].items() if yes == applies)
        assert sum(f["usable"] >= f["threshold"] for f in oracle) == row["metadata"]["positive_count"]
    elif source == "boundary_intent":
        expected["intent"] = next(o for o, label in oracle["intent_by_option"].items() if label == oracle["requested_intent"])
        assert oracle["phrase"] in record["questions"]["intent"]["instructions"]
    else:
        expected["relevance"] = next(o for o, kind in oracle["option_mapping"].items() if kind == oracle["relationship"])
    assert set(record["questions"]) == set(row["targets"]) == set(expected)
    for field, label in expected.items():
        assert row["targets"][field] == {o: float(o == label) for o in record["questions"][field]["criteria"]}


def make_split(split: str, seed: int) -> list[dict]:
    rows = []
    for i in range(2560 if split == "train" else 160):
        rows.extend(tool_group(i, split, seed))
    for i in range(3072 if split == "train" else 192):
        rows.append(multiselect(i, split, seed))
    for i in range(768 if split == "train" else 48):
        rows.extend(intent_group(i, split, seed))
    for i in range(384 if split == "train" else 24):
        rows.extend(product_group(i, split, seed))
    random.Random(f"{seed}:shuffle:{split}").shuffle(rows)
    return rows


def main(args: Args) -> None:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    seen = set()
    signatures = set()
    manifest = {
        "seed": args.seed,
        "generation": "programmatic counterfactual templates",
        "benchmark_inputs": "none",
        "validation": "held-out tool operation, intent, product families and wording",
        "splits": {},
        "limitations": "Executable-oracle checks do not constitute independent linguistic review or proof of public benchmark improvement.",
    }
    for split, expected in (("train", 16384), ("validation", 1024)):
        rows = make_split(split, args.seed)
        assert len(rows) == expected
        counts = collections.Counter()
        modes = collections.Counter()
        positive_bins = collections.Counter()
        for row in rows:
            verify_row(row)
            identifier = row["record"]["id"]
            assert identifier not in seen
            seen.add(identifier)
            signature = sha({"state": row["record"]["state"], "questions": row["record"]["questions"]})
            assert signature not in signatures
            signatures.add(signature)
            counts[row["source"]] += 1
            if row["source"] == "boundary_tool_use":
                modes[row["metadata"]["mode"]] += 1
            if row["source"] == "boundary_multilabel":
                n = row["metadata"]["positive_count"]
                positive_bins["none" if n == 0 else "one" if n == 1 else "all" if n == len(row["targets"]) else "several"] += 1
        path = root / f"{split}.jsonl"
        with path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sources": dict(counts), "tool_actions": dict(modes), "positive_count_bins": dict(positive_bins)}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main(Args(underscores_to_dashes=True).parse_args())
