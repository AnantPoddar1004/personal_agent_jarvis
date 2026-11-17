import json
import random
from datetime import datetime, timedelta

random.seed(42)

# ── pools ──────────────────────────────────────────────────────────────────────

NAMES = [
    "Aanand", "Priya", "James", "Sarah", "Mike", "Rahul", "Emily",
    "David", "Nina", "Carlos", "Aisha", "Tom", "Lily", "Omar", "Zoe",
    "Kevin", "Maya", "Chris", "Fatima", "Alex"
]

CONTACTS_DB = {
    name: {
        "phone": f"98304{i:05d}",
        "email": f"{name.lower()}@gmail.com"
    }
    for i, name in enumerate(NAMES)
}

ZOOM_TOPICS = [
    "standup", "team sync", "project review", "onboarding", "retrospective",
    "planning session", "design review", "client call", "1-on-1", "demo",
    "sprint planning", "brainstorm", "kickoff meeting", "check-in"
]

CALENDAR_TITLES = [
    "lunch", "dentist appointment", "gym", "team lunch", "coffee chat",
    "doctor appointment", "interview", "workshop", "training session", "dinner"
]

EMAIL_SUBJECTS = [
    "project update", "meeting notes", "follow up", "quick question",
    "report ready", "action items", "proposal", "feedback needed",
    "reminder", "introduction"
]

EMAIL_BODIES = [
    "Hi {name}, just wanted to send over the latest update. Let me know if you have any questions.",
    "Hey {name}, here are the meeting notes from today. Please review when you get a chance.",
    "Hi {name}, following up on our last conversation. Looking forward to hearing from you.",
    "Hey {name}, the report is ready for your review. Let me know your thoughts.",
    "Hi {name}, please find the action items from our last meeting below.",
    "Hey {name}, wanted to share the proposal with you. Happy to discuss further.",
    "Hi {name}, just a quick reminder about our upcoming meeting. See you then.",
]

TEXT_BODIES = [
    "Hey {name}, just checking in — let me know when you're free.",
    "Hi {name}, can you check your inbox when you get a chance?",
    "Hey {name}, just sent you an email — take a look when you can.",
    "Hi {name}, confirming our meeting — see you then!",
    "Hey {name}, are you available for a quick call today?",
    "Hi {name}, just wanted to give you a heads up about tomorrow.",
    "Hey {name}, the Zoom link is in your inbox — see you soon!",
]

RELATIVE_TIMES = [
    ("tomorrow at 9am",   1,  9, 0),
    ("tomorrow at 2pm",   1, 14, 0),
    ("tomorrow at 3pm",   1, 15, 0),
    ("tomorrow at 11am",  1, 11, 0),
    ("today at noon",     0, 12, 0),
    ("today at 4pm",      0, 16, 0),
    ("this Friday at 3pm",    "friday",      15, 0),
    ("next Monday at 10am",   "next_monday", 10, 0),
]

DURATIONS      = [30, 45, 60, 90, 120]
DURATION_LABELS = {30: "30 minutes", 45: "45 minutes", 60: "an hour", 90: "90 minutes", 120: "2 hours"}

BASE_DT        = datetime(2026, 3, 4, 22, 13, 0, 123456)
CLOCK_FEEDBACK = "Retrieved current datetime: 2026-03-04 22:13:00.123456"
ZOOM_LINK      = "https://zoom.us/j/1234567890"

def resolve_datetime(rel):
    _, day_offset, hour, minute = rel
    if day_offset == "friday":
        days = (4 - BASE_DT.weekday()) % 7 or 7
        dt = BASE_DT + timedelta(days=days)
    elif day_offset == "next_monday":
        days = (0 - BASE_DT.weekday()) % 7 or 7
        dt = BASE_DT + timedelta(days=days)
    else:
        dt = BASE_DT + timedelta(days=day_offset)
    return dt.replace(hour=hour, minute=minute, second=0, microsecond=0)

def fmt_dt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S.000000")

def pick(lst):  return random.choice(lst)
def pick_name(): return random.choice(NAMES)

def contact_feedback(name):
    c = CONTACTS_DB[name]
    return f'Retrieved email address and phone number: {{phone: "{c["phone"]}", email: "{c["email"]}"}}'

# ── pair builder ───────────────────────────────────────────────────────────────

def make_pair(user_input, history, reasoning, tool_call):
    if history:
        context_parts = []
        for h in history:
            context_parts.append(f"{h['action']}\n{h['env_feedback']}")
        context = "\n".join(context_parts)
    else:
        context = "None"
    return {
        "input":  f"<input>{user_input}</input>\nContext: {context}",
        "output": f"<reasoning>\n{reasoning}\n</reasoning>\n<tool>{tool_call}</tool>",
    }

def last_hist(action, env_feedback):
    return {"action": f"<tool>{action}</tool>", "env_feedback": env_feedback}

# ── single-step generators ─────────────────────────────────────────────────────

def gen_single_clock():
    q = pick([
        "What time is it right now?", "What's today's date?",
        "What's the current date and time?", "Can you tell me what time it is?",
        "What day is it today?",
    ])
    pairs = [
        make_pair(q, [], "The user wants the current date and time. I just need to call the clock tool.", "clock|get_date_time"),
        make_pair(q, [last_hist("clock|get_date_time", CLOCK_FEEDBACK)],
                  "Got the current datetime. That's all the user needed so I'm done.", "stop"),
    ]
    return pairs

def gen_single_contacts():
    name  = pick_name()
    field = pick(["phone number", "email address", "contact info", "number", "email"])
    q = pick([
        f"What is {name}'s {field}?", f"Can you look up {name}'s {field}?",
        f"Find {name}'s {field} for me", f"I need {name}'s {field}",
    ])
    return [
        make_pair(q, [], f"The user wants {name}'s contact details. I'll look them up in contacts.", f"contacts|get_info|{name}"),
        make_pair(q, [last_hist(f"contacts|get_info|{name}", contact_feedback(name))],
                  f"Retrieved {name}'s contact info successfully. Task is done.", "stop"),
    ]

def gen_single_email():
    name    = pick_name()
    subject = pick(EMAIL_SUBJECTS)
    body    = pick(EMAIL_BODIES).format(name=name)
    email   = CONTACTS_DB[name]["email"]
    q = pick([
        f"Email {name} about the {subject}", f"Send {name} an email about the {subject}",
        f"Can you email {name} regarding the {subject}?", f"Drop {name} an email about the {subject}",
    ])
    tool = f"email|send|{email}|{subject.title()}|{body}"
    return [
        make_pair(q, [], f"The user wants to email {name}. I have enough info to send directly.", tool),
        make_pair(q, [last_hist(tool, "Email sent successfully")],
                  f"Email sent to {name} successfully. Task is done.", "stop"),
    ]

def gen_single_text():
    name  = pick_name()
    phone = CONTACTS_DB[name]["phone"]
    body  = pick(TEXT_BODIES).format(name=name)
    q = pick([
        f"Text {name} to let them know I'm on my way",
        f"Send {name} a text saying I'll be there soon",
        f"Text {name} that I'm running late",
        f"Can you text {name} for me?",
    ])
    tool = f"text|send|{phone}|{body}"
    return [
        make_pair(q, [], f"The user wants to text {name}. I have their number so I can send directly.", tool),
        make_pair(q, [last_hist(tool, "Text sent successfully")],
                  f"Text sent to {name} successfully. Task is done.", "stop"),
    ]

def gen_single_zoom():
    topic = pick(ZOOM_TOPICS)
    rel   = pick(RELATIVE_TIMES)
    dur   = pick(DURATIONS)
    dt    = resolve_datetime(rel)
    tl, dl = rel[0], DURATION_LABELS[dur]
    q = pick([
        f"Create a Zoom link for a {dl} {topic} {tl}",
        f"Schedule a {topic} on Zoom for {tl}, {dl}",
        f"Set up a {dl} Zoom call for the {topic} {tl}",
    ])
    tool = f"zoom|schedule|{topic.title()}|{fmt_dt(dt)}|{dur}"
    return [
        make_pair(q, [], f"The user wants a Zoom meeting for {tl}. I need the current date first to resolve the exact datetime.", "clock|get_date_time"),
        make_pair(q, [last_hist("clock|get_date_time", CLOCK_FEEDBACK)],
                  f"Got the current date. {tl} resolves to {fmt_dt(dt)}. Now I'll create the Zoom meeting.", tool),
        make_pair(q, [last_hist(tool, f"Retrieved zoom meeting link: {ZOOM_LINK}")],
                  f"Zoom meeting created successfully for the {topic}. Task is done.", "stop"),
    ]

def gen_single_calendar():
    title = pick(CALENDAR_TITLES)
    rel   = pick(RELATIVE_TIMES)
    dur   = pick(DURATIONS)
    dt    = resolve_datetime(rel)
    tl, dl = rel[0], DURATION_LABELS[dur]
    q = pick([
        f"Add {title} to my calendar for {tl} for {dl}",
        f"Put {title} on my calendar {tl}, {dl}",
        f"Schedule {title} in my calendar for {tl}",
    ])
    tool = f"calendar|add_event|{fmt_dt(dt)}|{dur}|{title}"
    return [
        make_pair(q, [], f"The user wants to add {title} to their calendar for {tl}. I need to get the current date first to resolve that.", "clock|get_date_time"),
        make_pair(q, [last_hist("clock|get_date_time", CLOCK_FEEDBACK)],
                  f"Got the current date. {tl} resolves to {fmt_dt(dt)}. Now I'll add the calendar event.", tool),
        make_pair(q, [last_hist(tool, "Added event successfully")],
                  f"Calendar event for {title} added successfully. Task is done.", "stop"),
    ]

# ── 2-step generators ──────────────────────────────────────────────────────────

def gen_contacts_then_email():
    name    = pick_name()
    subject = pick(EMAIL_SUBJECTS)
    body    = pick(EMAIL_BODIES).format(name=name)
    email   = CONTACTS_DB[name]["email"]
    q = pick([
        f"Email {name} about the {subject}",
        f"Send an email to {name} regarding {subject}",
        f"Can you email {name} the {subject}?",
    ])
    tool = f"email|send|{email}|{subject.title()}|{body}"
    return [
        make_pair(q, [], f"I need to email {name} but I don't have their email address so I should look them up in contacts first.", f"contacts|get_info|{name}"),
        make_pair(q, [last_hist(f"contacts|get_info|{name}", contact_feedback(name))],
                  f"Got {name}'s email address which is {email}. Now I'll send the email.", tool),
        make_pair(q, [last_hist(tool, "Email sent successfully")],
                  f"Email sent to {name} at {email} successfully. Task is done.", "stop"),
    ]

def gen_contacts_then_text():
    name  = pick_name()
    phone = CONTACTS_DB[name]["phone"]
    body  = pick(TEXT_BODIES).format(name=name)
    q = pick([
        f"Text {name} to confirm our meeting tomorrow",
        f"Send {name} a text saying I'm on my way",
        f"Can you text {name} for me?",
        f"Text {name} that I'll be there in 10 minutes",
    ])
    tool = f"text|send|{phone}|{body}"
    return [
        make_pair(q, [], f"I need to text {name} but don't have their number. I'll look them up in contacts first.", f"contacts|get_info|{name}"),
        make_pair(q, [last_hist(f"contacts|get_info|{name}", contact_feedback(name))],
                  f"Got {name}'s phone number which is {phone}. Now I'll send the text.", tool),
        make_pair(q, [last_hist(tool, "Text sent successfully")],
                  f"Text sent to {name} successfully. Task is done.", "stop"),
    ]

def gen_clock_then_zoom():
    topic = pick(ZOOM_TOPICS)
    rel   = pick(RELATIVE_TIMES)
    dur   = pick(DURATIONS)
    dt    = resolve_datetime(rel)
    tl    = rel[0]
    q = pick([
        f"Schedule a {DURATION_LABELS[dur]} {topic} on Zoom for {tl}",
        f"Create a Zoom meeting for the {topic} {tl}, {DURATION_LABELS[dur]}",
    ])
    tool = f"zoom|schedule|{topic.title()}|{fmt_dt(dt)}|{dur}"
    return [
        make_pair(q, [], f"Need to create a Zoom meeting for {tl}. Getting current date first to resolve that.", "clock|get_date_time"),
        make_pair(q, [last_hist("clock|get_date_time", CLOCK_FEEDBACK)],
                  f"Today is 2026-03-04 so {tl} is {fmt_dt(dt)}. Creating the Zoom meeting now.", tool),
        make_pair(q, [last_hist(tool, f"Retrieved zoom meeting link: {ZOOM_LINK}")],
                  f"Zoom meeting for {topic} created successfully. Task is done.", "stop"),
    ]

def gen_clock_then_calendar():
    title = pick(CALENDAR_TITLES)
    rel   = pick(RELATIVE_TIMES)
    dur   = pick(DURATIONS)
    dt    = resolve_datetime(rel)
    tl    = rel[0]
    q = pick([
        f"Add {title} to my calendar for {tl}",
        f"Put {title} on my calendar {tl} for {DURATION_LABELS[dur]}",
    ])
    tool = f"calendar|add_event|{fmt_dt(dt)}|{dur}|{title}"
    return [
        make_pair(q, [], f"Need to add {title} to the calendar for {tl}. Getting current date first.", "clock|get_date_time"),
        make_pair(q, [last_hist("clock|get_date_time", CLOCK_FEEDBACK)],
                  f"Today is 2026-03-04 so {tl} is {fmt_dt(dt)}. Adding the event now.", tool),
        make_pair(q, [last_hist(tool, "Added event successfully")],
                  f"Calendar event for {title} added successfully. Task is done.", "stop"),
    ]

# ── 3-step generators ──────────────────────────────────────────────────────────

def gen_contacts_email_then_text():
    name      = pick_name()
    subject   = pick(EMAIL_SUBJECTS)
    body      = pick(EMAIL_BODIES).format(name=name)
    text_body = pick(TEXT_BODIES).format(name=name)
    email     = CONTACTS_DB[name]["email"]
    phone     = CONTACTS_DB[name]["phone"]
    q = pick([
        f"Email {name} the {subject} and then text them to check their inbox",
        f"Send {name} an email about {subject} and text them after",
        f"Email {name} regarding {subject} then send them a text",
    ])
    etool = f"email|send|{email}|{subject.title()}|{body}"
    ttool = f"text|send|{phone}|{text_body}"
    h0 = last_hist(f"contacts|get_info|{name}", contact_feedback(name))
    h1 = last_hist(etool, "Email sent successfully")
    h2 = last_hist(ttool, "Text sent successfully")
    return [
        make_pair(q, [], f"I need to email and then text {name}. I don't have their details so I'll look them up first.", f"contacts|get_info|{name}"),
        make_pair(q, [h0],
                  f"Got {name}'s email {email} and phone {phone}. Sending the email first.", etool),
        make_pair(q, [h0, h1],
                  f"Email sent to {name}. Now sending the follow-up text.", ttool),
        make_pair(q, [h0, h1, h2],
                  f"Both email and text sent to {name} successfully. Task is done.", "stop"),
    ]

def gen_clock_zoom_then_calendar():
    topic = pick(ZOOM_TOPICS)
    rel   = pick(RELATIVE_TIMES)
    dur   = pick(DURATIONS)
    dt    = resolve_datetime(rel)
    tl    = rel[0]
    q = pick([
        f"Schedule a {DURATION_LABELS[dur]} {topic} on Zoom for {tl} and add it to my calendar",
        f"Create a Zoom meeting for the {topic} {tl} and put it in my calendar",
        f"Set up the {topic} Zoom call for {tl} and block my calendar",
    ])
    ztool = f"zoom|schedule|{topic.title()}|{fmt_dt(dt)}|{dur}"
    ctool = f"calendar|add_event|{fmt_dt(dt)}|{dur}|{topic.title()}"
    h0 = last_hist("clock|get_date_time", CLOCK_FEEDBACK)
    h1 = last_hist(ztool, f"Retrieved zoom meeting link: {ZOOM_LINK}")
    h2 = last_hist(ctool, "Added event successfully")
    return [
        make_pair(q, [], f"I need to create a Zoom meeting and add it to the calendar for {tl}. Getting current date first.", "clock|get_date_time"),
        make_pair(q, [h0],
                  f"Today is 2026-03-04 so {tl} is {fmt_dt(dt)}. Creating the Zoom meeting first.", ztool),
        make_pair(q, [h0, h1],
                  f"Zoom meeting created. Now adding it to the calendar at the same time.", ctool),
        make_pair(q, [h0, h1, h2],
                  f"Zoom meeting created and calendar event added for {topic}. Task is done.", "stop"),
    ]

def gen_contacts_zoom_then_email():
    name  = pick_name()
    topic = pick(ZOOM_TOPICS)
    rel   = pick(RELATIVE_TIMES)
    dur   = pick(DURATIONS)
    dt    = resolve_datetime(rel)
    tl    = rel[0]
    email = CONTACTS_DB[name]["email"]
    q = pick([
        f"Schedule a Zoom {topic} with {name} for {tl} and email them the link",
        f"Create a Zoom meeting with {name} for {tl} then send them the link",
    ])
    ztool = f"zoom|schedule|{topic.title()}|{fmt_dt(dt)}|{dur}"
    ctool = f"contacts|get_info|{name}"
    etool = f"email|send|{email}|Zoom Link for {topic.title()}|Hi {name}, here is the Zoom link for our {topic}: {ZOOM_LINK}"
    h0 = last_hist("clock|get_date_time", CLOCK_FEEDBACK)
    h1 = last_hist(ztool, f"Retrieved zoom meeting link: {ZOOM_LINK}")
    h2 = last_hist(ctool, contact_feedback(name))
    h3 = last_hist(etool, "Email sent successfully")
    return [
        make_pair(q, [], f"I need to create a Zoom meeting for {tl} and email the link to {name}. Getting current date first.", "clock|get_date_time"),
        make_pair(q, [h0],
                  f"Today is 2026-03-04 so {tl} resolves to {fmt_dt(dt)}. Creating Zoom meeting now.", ztool),
        make_pair(q, [h0, h1],
                  f"Zoom link is {ZOOM_LINK}. Now I need {name}'s email to send it to them.", ctool),
        make_pair(q, [h0, h1, h2],
                  f"Got {name}'s email {email}. Sending them the Zoom link now.", etool),
        make_pair(q, [h0, h1, h2, h3],
                  f"Zoom meeting created and link emailed to {name} successfully. Task is done.", "stop"),
    ]

# ── dataset assembly ───────────────────────────────────────────────────────────

TRAJECTORY_TYPES = [
    (gen_single_clock,             150),
    (gen_single_contacts,          150),
    (gen_single_email,             150),
    (gen_single_text,              150),
    (gen_single_zoom,              200),
    (gen_single_calendar,          200),
    (gen_contacts_then_email,      400),
    (gen_contacts_then_text,       400),
    (gen_clock_then_zoom,          400),
    (gen_clock_then_calendar,      400),
    (gen_contacts_email_then_text, 600),
    (gen_clock_zoom_then_calendar, 600),
    (gen_contacts_zoom_then_email, 200),
]

def generate_dataset():
    all_pairs = []
    for gen_fn, n_trajectories in TRAJECTORY_TYPES:
        traj_pairs = []
        for _ in range(n_trajectories):
            traj_pairs.extend(gen_fn())
        all_pairs.extend(traj_pairs)
        print(f"  {gen_fn.__name__:<35s}: {n_trajectories} trajectories → {len(traj_pairs)} pairs")
    random.shuffle(all_pairs)
    return all_pairs

def main():
    print("Generating trajectory-augmented SFT dataset...")
    dataset = generate_dataset()
    print(f"\nTotal training pairs: {len(dataset)}")

    out = "data/trajectory_sft_examples.json"
    with open(out, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved to {out}")

    print("\n── Sample examples ──────────────────────────────────────────")
    for i in [0, 50, 200, 500, 1000]:
        ex = dataset[i]
        print(f"\n[Example {i}]")
        print(f"Input:\n{ex['input']}")
        print(f"Output:\n{ex['output']}")
        print("─" * 60)

if __name__ == "__main__":
    main()
