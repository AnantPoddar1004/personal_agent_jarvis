"""
training_tools.py
Each tool validates its inputs and returns (result, reward).

Reward structure per tool call:
    All params valid:       +0.5  (called at the right time with right info)
    Per invalid param:      -1.0  (subtracted for each invalid param)
    All params invalid:     -3.0  (floor)

Validation logic:
    - email    : must contain @ and a dot after @, AND must be in CONTACTS_DB
    - phone    : must be all digits 7-15 chars, AND must be in CONTACTS_DB
    - datetime : must match YYYY-MM-DD HH:MM:SS format AND must be in the future
    - duration : must be a positive integer (or string of digits)
    - name     : must be a non-empty string with no digits
    - topic    : must be a non-empty string
    - subject  : must be a non-empty string
    - body     : must be a non-empty string
    - title    : must be a non-empty string

Note: All parameters are received as raw strings (no type enforcement in
signatures) so that invalid inputs are caught by validation logic below
rather than raising Python TypeErrors before we can penalize them.
"""

import re
import random
import datetime

# ── Contacts database ─────────────────────────────────────────────────────────

CONTACTS_DB = {
    "Aanand": {"phone": "2025550100", "email": "aanand@example.org"},
    "David":  {"phone": "2025550101", "email": "david.chen@example.org"},
    "Emily":  {"phone": "2025550102", "email": "emily.jones@example.org"},
    "Lily":   {"phone": "2025550103", "email": "lily.wang@example.org"},
    "Mike":   {"phone": "2025550104", "email": "mike.ross@example.org"},
    "Omar":   {"phone": "2025550105", "email": "omar.hassan@example.org"},
    "Priya":  {"phone": "2025550106", "email": "priya.sharma@example.org"},
    "Rahul":  {"phone": "2025550107", "email": "rahul.gupta@example.org"},
    "Sarah":  {"phone": "2025550108", "email": "sarah.miller@example.org"},
}

# ── Reward constants ──────────────────────────────────────────────────────────
REWARD_ALL_VALID    =  0.5   # every param passed validation
REWARD_PER_INVALID  = -1.0   # subtracted for each invalid param
REWARD_FLOOR        = -3.0   # minimum possible reward per tool call


# ── Validators ────────────────────────────────────────────────────────────────

def _is_valid_email(value) -> bool:
    """Must be a valid email format AND exist in CONTACTS_DB."""
    if not isinstance(value, str):
        return False
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    if not re.match(pattern, value.strip()):
        return False
    known_emails = {v["email"] for v in CONTACTS_DB.values()}
    return value.strip() in known_emails


def _is_valid_phone(value) -> bool:
    """Must be a valid phone format AND exist in CONTACTS_DB."""
    if not isinstance(value, str):
        return False
    digits_only = re.sub(r"[\s\-\(\)]", "", value.strip())
    if not (digits_only.isdigit() and 7 <= len(digits_only) <= 15):
        return False
    known_phones = {v["phone"] for v in CONTACTS_DB.values()}
    return digits_only in known_phones


def _is_valid_datetime(value) -> bool:
    """
    Must match YYYY-MM-DD HH:MM:SS or YYYY-MM-DD HH:MM:SS.ffffff
    AND must be in the future relative to current time.
    """
    if not isinstance(value, str):
        return False
    pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?$"
    if not re.match(pattern, value.strip()):
        return False
    try:
        fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in value else "%Y-%m-%d %H:%M:%S"
        proposed_dt = datetime.datetime.strptime(value.strip(), fmt)
        return proposed_dt > datetime.datetime.now()
    except ValueError:
        return False


def _is_valid_duration(value) -> bool:
    """Must be a positive integer or a string representation of one."""
    try:
        return int(value) > 0
    except (ValueError, TypeError):
        return False


def _is_valid_name(value) -> bool:
    """Must be a non-empty string containing only letters, spaces, hyphens."""
    if not isinstance(value, str) or not value.strip():
        return False
    return bool(re.match(r"^[A-Za-z\s\-']+$", value.strip()))


def _is_valid_nonempty_string(value) -> bool:
    """Must be a non-empty string."""
    return isinstance(value, str) and bool(value.strip())


# ── Reward calculator ─────────────────────────────────────────────────────────

def _calculate_reward(invalid_count: int, total_params: int) -> float:
    """
    Starts at REWARD_ALL_VALID and subtracts REWARD_PER_INVALID per bad param.
    Floors at REWARD_FLOOR.
    """
    if invalid_count == 0:
        return REWARD_ALL_VALID
    reward = REWARD_ALL_VALID + (invalid_count * REWARD_PER_INVALID)
    return max(reward, REWARD_FLOOR)


# ── Tools ─────────────────────────────────────────────────────────────────────

def zoom_schedule(topic, start_time, duration):
    """
    Params:
        topic      (string)   : meeting title
        start_time (datetime) : YYYY-MM-DD HH:MM:SS format, must be in the future
        duration   (int)      : meeting length in minutes
    Returns:
        (result, reward)
    """
    validations = {
        "topic":      _is_valid_nonempty_string(topic),
        "start_time": _is_valid_datetime(start_time),
        "duration":   _is_valid_duration(duration),
    }
    invalid_count = sum(1 for v in validations.values() if not v)
    reward = _calculate_reward(invalid_count, len(validations))

    invalid_fields = [k for k, v in validations.items() if not v]
    if invalid_fields:
        result = f"[zoom|schedule] Invalid fields: {invalid_fields}"
    else:
        meeting_id = "".join([str(random.randint(0, 9)) for _ in range(10)])
        result = f"https://zoom.us/j/{meeting_id}"

    return result, reward


def email_send(to, subject, body):
    """
    Params:
        to      (email string) : recipient email — must exist in CONTACTS_DB
        subject (string)       : email subject line
        body    (string)       : email body content
    Returns:
        (result, reward)
    """
    validations = {
        "to":      _is_valid_email(to),
        "subject": _is_valid_nonempty_string(subject),
        "body":    _is_valid_nonempty_string(body),
    }
    invalid_count = sum(1 for v in validations.values() if not v)
    reward = _calculate_reward(invalid_count, len(validations))

    invalid_fields = [k for k, v in validations.items() if not v]
    if invalid_fields:
        result = f"[email|send] Invalid fields: {invalid_fields}"
    else:
        result = "Email sent successfully"

    return result, reward


def text_send(to, body):
    """
    Params:
        to   (phone string) : recipient phone — must exist in CONTACTS_DB
        body (string)       : message content
    Returns:
        (result, reward)
    """
    validations = {
        "to":   _is_valid_phone(to),
        "body": _is_valid_nonempty_string(body),
    }
    invalid_count = sum(1 for v in validations.values() if not v)
    reward = _calculate_reward(invalid_count, len(validations))

    invalid_fields = [k for k, v in validations.items() if not v]
    if invalid_fields:
        result = f"[text|send] Invalid fields: {invalid_fields}"
    else:
        result = "Text message sent successfully"

    return result, reward


def calendar_add_event(dt, duration, title):
    """
    Params:
        dt       (datetime string) : YYYY-MM-DD HH:MM:SS format, must be in the future
        duration (int)             : event length in minutes
        title    (string)          : event name
    Returns:
        (result, reward)
    """
    validations = {
        "datetime": _is_valid_datetime(dt),
        "duration": _is_valid_duration(duration),
        "title":    _is_valid_nonempty_string(title),
    }
    invalid_count = sum(1 for v in validations.values() if not v)
    reward = _calculate_reward(invalid_count, len(validations))

    invalid_fields = [k for k, v in validations.items() if not v]
    if invalid_fields:
        result = f"[calendar|add_event] Invalid fields: {invalid_fields}"
    else:
        result = "Added event successfully"

    return result, reward


def contacts_get_info(name):
    """
    Params:
        name (string) : contact name to look up (letters only)
    Returns:
        (result, reward)
    """
    validations = {
        "name":       _is_valid_name(name),
        "name_in_db": name.strip().title() in CONTACTS_DB,
    }
    invalid_count = sum(1 for v in validations.values() if not v)
    reward = _calculate_reward(invalid_count, len(validations))

    invalid_fields = [k for k, v in validations.items() if not v]
    if invalid_fields:
        result = f"[contacts|get_info] Invalid fields: {invalid_fields}"
    else:
        contact = CONTACTS_DB.get(name.strip().title())
        result = {"phone": contact["phone"], "email": contact["email"]}

    return result, reward


def clock_get_date_time():
    """
    No params — always valid, always rewarded.
    Returns current real datetime as a string.
    Returns:
        (result, reward)
    """
    return str(datetime.datetime.now()), REWARD_ALL_VALID


def stop():
    """
    Signals task completion — always valid.
    Reward of 2.0 to strongly encourage completing the task.
    Returns:
        (None, reward)
    """
    return None, 2.0