import json
import random
from datetime import datetime, timedelta
from collections import Counter

random.seed(42)

# ---------------------------------------------------------------------------
# DATA POOLS
# ---------------------------------------------------------------------------

first_names = [
    "aanand","sarah","john","emily","michael","jessica","david","ashley","james","jennifer",
    "robert","amanda","william","melissa","richard","stephanie","joseph","nicole","thomas","heather",
    "charles","amber","christopher","megan","daniel","brittany","matthew","elizabeth","andrew","lauren",
    "joshua","samantha","kevin","rachel","brian","christina","george","diana","edward","anna",
    "ronald","maria","timothy","karen","jason","lisa","jeffrey","nancy","ryan","betty",
    "jacob","margaret","gary","sandra","nicholas","donna","eric","carol","jonathan","ruth",
    "stephen","sharon","larry","michelle","justin","laura","scott","brandon","kimberly",
    "raymond","deborah","frank","dorothy","gregory","amy","helen","patrick","kathleen",
    "alex","priya","raj","neha","arjun","divya","rohan","pooja","vikram","ananya",
    "liam","olivia","noah","emma","oliver","ava","elijah","sophia","lucas","isabella",
    "mason","mia","ethan","charlotte","aiden","amelia","logan","harper","jackson","evelyn",
    "sebastian","abigail","mateo","emily","jack","ella","owen","elizabeth","theodore","sofia",
    "henry","camila","wyatt","victoria","leo","madison","julian","luna","ezra","grace",
    "sam","taylor","jordan","morgan","casey","riley","quinn","avery","blake","cameron",
    "derek","fiona","grant","hailey","ivan","julia","kyle","leah","marcus","nina",
    "oscar","piper","quentin","rosa","sean","tina","ursula","vince","wendy","xander"
]

email_domains = [
    "gmail.com","yahoo.com","outlook.com","company.com","work.org","school.edu",
    "startup.io","corp.net","business.com","uni.edu","hotmail.com","icloud.com",
    "protonmail.com","me.com","live.com","msn.com","aol.com","mail.com","hey.com","fastmail.com"
]

def rand_email():
    name = random.choice(first_names)
    domain = random.choice(email_domains)
    pattern = random.randint(0, 5)
    if pattern == 0:
        return f"{name}@{domain}"
    elif pattern == 1:
        return f"{name}{random.randint(1,999)}@{domain}"
    elif pattern == 2:
        return f"{name}.{random.choice(first_names)}@{domain}"
    elif pattern == 3:
        return f"{name}_{random.choice(first_names)}@{domain}"
    elif pattern == 4:
        return f"{name[0]}{random.choice(first_names)}@{domain}"
    else:
        return f"{random.choice(first_names)}.{name}{random.randint(10,99)}@{domain}"

def rand_phone():
    # Formats: plain 10-digit, with dashes, with dots, with parens
    digits = f"{random.randint(200,999)}{random.randint(1000000,9999999)}"
    fmt = random.randint(0, 3)
    if fmt == 0:
        return digits
    elif fmt == 1:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    elif fmt == 2:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    else:
        return f"{digits[:3]}.{digits[3:6]}.{digits[6:]}"

def rand_datetime():
    base = datetime(2026, 1, 1)
    delta = timedelta(
        days=random.randint(0, 364),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    dt = base + delta
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{random.randint(0,999999):06d}"

def rand_duration():
    return random.choice([15, 20, 25, 30, 45, 60, 75, 90, 105, 120, 150, 180, 240])

meeting_topics = [
    "Team Standup","Product Demo","Intro Meeting","Project Kickoff","Design Review",
    "Sprint Planning","Retrospective","Client Onboarding","Sales Call","Budget Review",
    "Marketing Sync","Engineering All Hands","Interview","One on One","Quarterly Review",
    "User Research Session","Board Meeting","Strategy Session","Investor Update","Code Review",
    "Customer Check In","Vendor Meeting","Partnership Discussion","Hiring Review","Training Session",
    "Workshop","Brainstorming Session","Status Update","Performance Review","Onboarding Call",
    "Technical Deep Dive","Product Roadmap Review","Infrastructure Planning","Security Audit","Data Review",
    "Feedback Session","Launch Planning","Go To Market Meeting","Analytics Review","Support Sync",
    "Weekly Sync","Monthly Catchup","Q3 Planning","Annual Review","Team Offsite Prep",
    "Architecture Discussion","Feature Scoping","Bug Triage","Release Planning","Incident Postmortem",
    "Accessibility Review","Compliance Meeting","Finance Sync","Legal Review","HR Discussion",
    "Content Planning","SEO Strategy","Growth Review","Customer Success Sync","Partnership Call",
    "Pilot Review","Beta Feedback Session","Demo Day Prep","Investor Pitch Practice","Advisory Board Meeting"
]

calendar_events = [
    "Lunch","Doctor Appointment","Dentist Appointment","Gym","Team Lunch","Coffee Chat",
    "Study Session","Library Visit","Grocery Run","Hair Appointment","Flight","Hotel Checkout",
    "Birthday Party","Anniversary Dinner","Family Dinner","Movie Night","Date Night","Networking Event",
    "Conference Talk","Workshop","Yoga Class","Physical Therapy","Eye Exam","Car Service",
    "Volunteer Work","Book Club","Game Night","Hiking Trip","Spa Day","Concert",
    "Team Building","Offsite","Hackathon","Study Group","Tutoring Session","Piano Lesson",
    "Soccer Practice","Running Club","Cooking Class","Art Class","Therapy Session",
    "Job Interview","Networking Lunch","Alumni Meetup","Tax Appointment","Visa Appointment",
    "Blood Donation","Vaccine Appointment","Car Wash","Oil Change","Vet Appointment",
    "School Play","Graduation Ceremony","Wedding","Baby Shower","Retirement Party",
    "Morning Run","Meditation","Language Class","Swimming Practice","Tennis Match",
    "Podcast Recording","Photography Session","Pottery Class","Dance Rehearsal","Choir Practice"
]

email_subjects = [
    "Hello World","Project Update","Meeting Follow Up","Quick Question","Introduction",
    "Action Items","Weekly Recap","Important Announcement","Invitation","Thank You",
    "Feedback Request","Q1 Report","Q2 Summary","New Proposal","Contract Review",
    "Urgent: Please Respond","Reminder","Check In","Newsletter","Job Opportunity",
    "Invoice Attached","Approval Needed","Schedule Change","Welcome to the Team","Update on Progress",
    "Next Steps","Follow Up from Yesterday","Proposal for Review","Research Findings","Please Review",
    "Quick Update","FYI","Heads Up","Re: Our Last Call","Following Up",
    "Request for Feedback","Time Sensitive","New Feature Launch","Product Announcement","Bug Report",
    "Partnership Opportunity","Collaboration Request","Application Received","Interview Confirmation","Offer Letter",
    "Meeting Confirmation","Reschedule Request","Out of Office","Return to Office","Policy Update",
    "Team Announcement","Welcome Aboard","Monthly Newsletter","Event Invitation","Webinar Registration"
]

email_bodies = [
    "how are you","Please find the report attached","Let me know if you have any questions",
    "Looking forward to hearing from you","I wanted to follow up on our last conversation",
    "Please review the attached documents","Can we schedule a time to chat",
    "Just checking in to see how things are going","We are on track for the deadline",
    "Please confirm your availability for next week","Thank you for your time today",
    "I have attached the invoice for your review","Here is a summary of our discussion",
    "Please let me know if you need anything else","The project is progressing well",
    "I wanted to share some exciting news","We need your approval to move forward",
    "Please review and sign the attached contract","Looking forward to working with you",
    "I hope this email finds you well","Here are the action items from our meeting",
    "Please find the meeting notes below","We have completed the first phase",
    "I wanted to introduce myself","Please share your feedback at your earliest convenience",
    "The team has made significant progress","Kindly confirm receipt of this email",
    "We would love to have you join us","Please find the updated schedule attached",
    "I am writing to follow up on my previous email","Hope you had a great weekend",
    "Just a quick note to say thank you","I wanted to check in on the status",
    "Please let me know your thoughts","We are excited to share this update with you",
    "Could you please review the following","I have some questions about the project",
    "The meeting has been rescheduled","Please see the attached agenda",
    "I wanted to reach out regarding your inquiry","Thank you for your patience",
    "We appreciate your continued support","Please do not hesitate to reach out"
]

text_bodies = [
    "Happy Birthday","Can you call me later?","I'll be there in 10 minutes","On my way",
    "Running 5 minutes late","Are you free tonight?","Let's grab lunch","Did you see my email?",
    "Call me when you get a chance","Meeting is confirmed for tomorrow","Don't forget about tonight",
    "Can you send me the address?","I'm outside","See you soon","Thanks for everything",
    "Got it, thanks","Sure, sounds good","Can we reschedule?","Just landed","Heading home now",
    "The package arrived","Don't forget to bring your ID","I'll meet you there","Can you help me?",
    "Happy New Year","Happy Holidays","Congrats on the promotion","Hope you feel better soon",
    "Good luck today","Miss you","Thinking of you","Let me know when you're free",
    "Check your email","Did you get my voicemail?","I'll call you back","Almost there",
    "Stuck in traffic","The code is 4521","Meet me at the lobby","See you at 6",
    "What time works for you?","Can you pick me up?","I'll be right back","Leaving now",
    "Can you grab some groceries?","Dinner is ready","Did you make it home safe?",
    "Let me know when you arrive","I need your help with something","Can we talk later?",
    "Just finished the meeting","Heading to the office now","Lunch at the usual spot?",
    "Do you need anything?","Flight is delayed","Landed safely","Hotel is booked",
    "Great news, I got the job","Interview went really well","They offered me the position",
    "Can you cover for me today?","I'm taking a sick day","Working from home today",
    "The wifi is down","I need the password","Did you reset the router?","Power is back on",
    "Happy Anniversary","Merry Christmas","Happy Thanksgiving","Happy Halloween","Congrats on the baby"
]

# ---------------------------------------------------------------------------
# TEMPLATES  — 40+ per tool for maximum linguistic variety
# ---------------------------------------------------------------------------

zoom_templates = [
    # Direct imperatives
    lambda t,dt,d: f"Schedule a {d} minute zoom meeting called {t} at {dt}",
    lambda t,dt,d: f"Create a Zoom meeting for {t} starting at {dt} that lasts {d} minutes",
    lambda t,dt,d: f"Book a {d}-minute Zoom call titled {t} at {dt}",
    lambda t,dt,d: f"Set up a Zoom meeting called {t} at {dt} for {d} minutes",
    lambda t,dt,d: f"Make a Zoom meeting: {t} at {dt}, {d} minutes",
    lambda t,dt,d: f"Add a {d} minute Zoom session called {t} at {dt}",
    lambda t,dt,d: f"Organize a Zoom call titled {t} for {dt}, duration {d} minutes",
    lambda t,dt,d: f"Put together a {d} minute Zoom meeting called {t} at {dt}",
    # First person need/want
    lambda t,dt,d: f"I need to set up a {d} minute Zoom call titled {t} at {dt}",
    lambda t,dt,d: f"I have to schedule a {d} minute zoom meeting titled {t} at {dt}",
    lambda t,dt,d: f"I want to create a zoom meeting: {t}, at {dt}, {d} minutes long",
    lambda t,dt,d: f"I need a Zoom meeting set up for {t} at {dt}. It should be {d} minutes.",
    lambda t,dt,d: f"I'd like to schedule a zoom for {t} at {dt} that goes for {d} minutes",
    lambda t,dt,d: f"I'm trying to set up a Zoom call called {t} at {dt} for {d} minutes",
    lambda t,dt,d: f"I need you to schedule a {d}-minute Zoom meeting called {t} at {dt}",
    lambda t,dt,d: f"I want a {d} minute Zoom meeting called {t} scheduled for {dt}",
    # Polite requests
    lambda t,dt,d: f"Please book a {d} minute zoom meeting called {t} at {dt}",
    lambda t,dt,d: f"Could you schedule a zoom session titled {t} at {dt} for {d} minutes?",
    lambda t,dt,d: f"Can you create a Zoom meeting called {t} for {dt} lasting {d} minutes?",
    lambda t,dt,d: f"Would you set up a {d} minute Zoom call for {t} at {dt}?",
    lambda t,dt,d: f"Can you please schedule a zoom for {t} at {dt}? It needs to be {d} minutes.",
    lambda t,dt,d: f"Please set up a Zoom meeting titled {t} at {dt} for {d} minutes",
    lambda t,dt,d: f"Could you make a {d} minute Zoom meeting for {t} at {dt}?",
    # Casual / shorthand
    lambda t,dt,d: f"Book zoom for {t} at {dt}, {d} mins",
    lambda t,dt,d: f"Schedule zoom: {t} | {dt} | {d} minutes",
    lambda t,dt,d: f"Setup a {d} min zoom titled {t} starting at {dt}",
    lambda t,dt,d: f"Zoom meeting — {t}, {dt}, {d} min",
    lambda t,dt,d: f"{t} zoom call at {dt}, {d} minutes",
    lambda t,dt,d: f"New zoom: {t}, {d} mins, starts {dt}",
    lambda t,dt,d: f"zoom {t} at {dt} for {d} min",
    # Add-to-schedule framing
    lambda t,dt,d: f"Add a Zoom meeting called {t} at {dt} for {d} minutes to my schedule",
    lambda t,dt,d: f"Put a {d} minute Zoom call titled {t} on my calendar at {dt}",
    lambda t,dt,d: f"Block {d} minutes for a Zoom called {t} at {dt}",
    # Descriptive / verbose
    lambda t,dt,d: f"I have a meeting coming up called {t} and I need a Zoom link. It starts at {dt} and will last {d} minutes.",
    lambda t,dt,d: f"We're having a {t} zoom call at {dt}. Please schedule it for {d} minutes.",
    lambda t,dt,d: f"The {t} session is happening at {dt} and should be {d} minutes on Zoom",
    lambda t,dt,d: f"I need to get a Zoom room set up for {t}. Start time: {dt}. Duration: {d} minutes.",
    lambda t,dt,d: f"There's a {d}-minute call for {t} happening at {dt} — can you schedule it on Zoom?",
    lambda t,dt,d: f"My meeting for {t} is at {dt}. Set up a Zoom for {d} minutes please.",
    lambda t,dt,d: f"Can you get Zoom sorted for the {t} meeting at {dt}? It's {d} minutes long.",
    # Terse commands
    lambda t,dt,d: f"Zoom: {t}, {dt}, {d}min",
    lambda t,dt,d: f"Schedule {t} on Zoom at {dt} — {d} minutes",
    lambda t,dt,d: f"Create {d}min zoom for {t} at {dt}",
]

email_templates = [
    # Direct imperatives
    lambda to,s,b: f"Send an email to {to} with the subject {s} and the body {b}",
    lambda to,s,b: f"Email {to}, subject is {s}, body is {b}",
    lambda to,s,b: f"Compose and send an email to {to}. Subject: {s}. Body: {b}",
    lambda to,s,b: f"Draft an email to {to} with subject line {s} and message {b}",
    lambda to,s,b: f"Send a message to {to} via email. Subject: {s}. Body: {b}",
    lambda to,s,b: f"Fire off an email to {to}. Subject: {s}. Body: {b}",
    lambda to,s,b: f"Write an email to {to}. Title it {s} and say: {b}",
    lambda to,s,b: f"Shoot an email to {to} with subject {s} saying {b}",
    lambda to,s,b: f"Send {to} an email titled {s} with the following body: {b}",
    lambda to,s,b: f"Dispatch an email to {to}: subject {s}, content {b}",
    # First person need/want
    lambda to,s,b: f"I have to send an email to {to}. The subject is {s}. The body of the email is {b}.",
    lambda to,s,b: f"I need to email {to}. Use {s} as the subject and write {b} in the body",
    lambda to,s,b: f"I want to send an email to {to} titled {s}. The body is: {b}",
    lambda to,s,b: f"I'd like to shoot an email to {to} with subject {s} saying {b}",
    lambda to,s,b: f"I'm trying to reach {to} over email. Subject: {s}. Message: {b}",
    lambda to,s,b: f"I need you to send an email to {to}. Subject: {s}. Body: {b}",
    lambda to,s,b: f"I want you to email {to} with the subject {s} and body {b}",
    # Polite requests
    lambda to,s,b: f"Please send an email to {to} with subject {s}. The message should say: {b}",
    lambda to,s,b: f"Can you send {to} an email? Subject: {s}. Body: {b}",
    lambda to,s,b: f"Could you email {to}? The subject should be {s} and the body should say {b}",
    lambda to,s,b: f"Would you mind sending an email to {to} with subject {s} and body {b}?",
    lambda to,s,b: f"Please email {to} with the following — subject: {s}, body: {b}",
    lambda to,s,b: f"Can you please draft and send an email to {to}? Subject: {s}. Message: {b}",
    # Casual / shorthand
    lambda to,s,b: f"Send email: to {to}, subject {s}, message {b}",
    lambda to,s,b: f"Email {to} — {s} — {b}",
    lambda to,s,b: f"email to: {to}, re: {s}, body: {b}",
    lambda to,s,b: f"Quick email to {to}: subject {s}, body {b}",
    lambda to,s,b: f"{to} email, subject: {s}, message: {b}",
    # Descriptive / verbose
    lambda to,s,b: f"I need to get in touch with {to} by email. The subject of the email should be {s} and the body should read: {b}",
    lambda to,s,b: f"Please reach out to {to} via email. Use {s} as the subject line and write the following in the body: {b}",
    lambda to,s,b: f"There's an email I need to send to {to}. It should have {s} as the subject and the body text should be: {b}",
    lambda to,s,b: f"Can you handle sending an email to {to} for me? The subject is {s} and I want the body to say {b}",
    lambda to,s,b: f"I need an email sent to {to}. Subject line: {s}. The email body should read: {b}",
    lambda to,s,b: f"Draft and deliver an email to {to} with title {s} and body text reading {b}",
    # Terse commands
    lambda to,s,b: f"Mail {to}: {s} / {b}",
    lambda to,s,b: f"Send mail to {to}, subject: {s}, content: {b}",
    lambda to,s,b: f"compose email {to} | {s} | {b}",
    lambda to,s,b: f"Email → {to} | subject: {s} | {b}",
    lambda to,s,b: f"Outbound email: {to}, {s}, {b}",
    # Action-focused
    lambda to,s,b: f"Notify {to} by email. Subject: {s}. Tell them: {b}",
    lambda to,s,b: f"Reach out to {to} via email with the subject {s} and message {b}",
    lambda to,s,b: f"Ping {to} with an email — subject {s}, body {b}",
]

text_templates = [
    # Direct imperatives
    lambda to,b: f"Send a text to {to} saying {b}",
    lambda to,b: f"Text {to} the message {b}",
    lambda to,b: f"Send {to} a text that says {b}",
    lambda to,b: f"Message {to} via text: {b}",
    lambda to,b: f"Send a text message to {to}: {b}",
    lambda to,b: f"Shoot a text to {to} saying {b}",
    lambda to,b: f"Send an SMS to {to} with the message {b}",
    lambda to,b: f"Fire off a text to {to}: {b}",
    lambda to,b: f"Send {to} an SMS saying {b}",
    lambda to,b: f"Drop a text to {to}: {b}",
    # First person need/want
    lambda to,b: f"I have to send a text saying {b} to {to}",
    lambda to,b: f"I need to text {to}. The message should be: {b}",
    lambda to,b: f"I want to send a text to {to} saying {b}",
    lambda to,b: f"I'd like to text {to}: {b}",
    lambda to,b: f"I need you to text {to} and say {b}",
    lambda to,b: f"I'm trying to reach {to} by text. Send: {b}",
    lambda to,b: f"I want you to send a text to {to} saying {b}",
    # Polite requests
    lambda to,b: f"Can you text {to}? The message is {b}",
    lambda to,b: f"Please send a text message to {to}: {b}",
    lambda to,b: f"Could you send a text to {to} with the message {b}?",
    lambda to,b: f"Would you text {to} and say {b}?",
    lambda to,b: f"Can you please text {to}: {b}",
    lambda to,b: f"Please text {to} the following message: {b}",
    # Casual / shorthand
    lambda to,b: f"Text {to}: {b}",
    lambda to,b: f"SMS {to} — {b}",
    lambda to,b: f"text to {to}: {b}",
    lambda to,b: f"{to} text: {b}",
    lambda to,b: f"msg {to}: {b}",
    # Descriptive / verbose
    lambda to,b: f"I need a text sent to {to}. The content of the message should be: {b}",
    lambda to,b: f"Can you send a text message on my behalf to {to}? I want it to say: {b}",
    lambda to,b: f"There's a text I need to send to {to}. The message reads: {b}",
    lambda to,b: f"Please reach out to {to} by text and tell them: {b}",
    lambda to,b: f"I'd like to ping {to} with a text that says {b}",
    lambda to,b: f"Send the following SMS to {to}: {b}",
    # Terse commands
    lambda to,b: f"Send SMS to {to}: {b}",
    lambda to,b: f"iMessage {to}: {b}",
    lambda to,b: f"txt {to} — {b}",
    lambda to,b: f"Quick text to {to}: {b}",
    lambda to,b: f"Outbound text → {to}: {b}",
    # Action-focused
    lambda to,b: f"Notify {to} by text: {b}",
    lambda to,b: f"Ping {to} with a text saying {b}",
    lambda to,b: f"Reach out to {to} via SMS: {b}",
]

calendar_templates = [
    # Direct imperatives
    lambda dt,d,t: f"Add a {d} minute event called {t} to my calendar at {dt}",
    lambda dt,d,t: f"Block off {d} minutes on my calendar for {t} starting at {dt}",
    lambda dt,d,t: f"Schedule a {t} event at {dt} for {d} minutes",
    lambda dt,d,t: f"Create a calendar event: {t} at {dt} for {d} minutes",
    lambda dt,d,t: f"Put {t} on my calendar at {dt} for {d} minutes",
    lambda dt,d,t: f"Add {t} to my calendar starting at {dt} for {d} minutes",
    lambda dt,d,t: f"Book {t} on my calendar at {dt} for {d} minutes",
    lambda dt,d,t: f"Set a {d}-minute calendar block for {t} at {dt}",
    lambda dt,d,t: f"Create a {d} minute appointment called {t} at {dt}",
    lambda dt,d,t: f"Log a {d}-minute {t} event starting at {dt}",
    # First person need/want
    lambda dt,d,t: f"I need to add an event called {t} at {dt} that lasts {d} minutes",
    lambda dt,d,t: f"I want to add {t} to my calendar at {dt} for {d} minutes",
    lambda dt,d,t: f"I'd like to block off {d} minutes for {t} at {dt}",
    lambda dt,d,t: f"I need you to add {t} to my calendar at {dt} for {d} minutes",
    lambda dt,d,t: f"I want a {d} minute {t} event added to my calendar at {dt}",
    lambda dt,d,t: f"I'm trying to schedule {t} at {dt} for {d} minutes — can you add it?",
    # Polite requests
    lambda dt,d,t: f"Please add a {d} minute {t} to my calendar at {dt}",
    lambda dt,d,t: f"Can you add {t} to my schedule at {dt}? It's {d} minutes long.",
    lambda dt,d,t: f"Could you put {t} on my calendar at {dt} for {d} minutes?",
    lambda dt,d,t: f"Would you add a {d} minute {t} event to my calendar at {dt}?",
    lambda dt,d,t: f"Please schedule {t} on my calendar at {dt}, duration {d} minutes",
    lambda dt,d,t: f"Can you please create a {d} minute calendar event for {t} at {dt}?",
    # Casual / shorthand
    lambda dt,d,t: f"Calendar: add {t} at {dt} for {d} minutes",
    lambda dt,d,t: f"{t} — {dt} — {d} mins — add to calendar",
    lambda dt,d,t: f"cal event: {t}, {dt}, {d}min",
    lambda dt,d,t: f"add to cal: {t} at {dt}, {d} min",
    lambda dt,d,t: f"{t} on cal at {dt} for {d} min",
    # Descriptive / verbose
    lambda dt,d,t: f"I have a {t} coming up at {dt} and I need it on my calendar. It will last {d} minutes.",
    lambda dt,d,t: f"Can you make sure {t} is on my calendar? It starts at {dt} and goes for {d} minutes.",
    lambda dt,d,t: f"There's a {d}-minute {t} event at {dt} that I need added to my calendar.",
    lambda dt,d,t: f"Please make a note on my calendar for {t} at {dt}. Set the duration to {d} minutes.",
    lambda dt,d,t: f"I've got {t} at {dt} for {d} minutes — block that on my calendar for me.",
    lambda dt,d,t: f"My {t} is at {dt} and will take {d} minutes. Put that on the calendar please.",
    # Terse commands
    lambda dt,d,t: f"New calendar event: {t}, {dt}, {d}min",
    lambda dt,d,t: f"Schedule {t} at {dt} for {d} minutes",
    lambda dt,d,t: f"{d}min {t} at {dt} — add to calendar",
    lambda dt,d,t: f"Cal block: {t} | {dt} | {d} min",
    # Action-focused
    lambda dt,d,t: f"Mark my calendar for {t} at {dt}, it's {d} minutes",
    lambda dt,d,t: f"Reserve {d} minutes on my calendar for {t} starting at {dt}",
    lambda dt,d,t: f"Hold {dt} on my calendar for {t} — {d} minutes",
]

contacts_templates = [
    # Direct imperatives
    lambda n: f"Get the contact info for {n}",
    lambda n: f"Look up {n} in my contacts",
    lambda n: f"Find {n}'s contact details",
    lambda n: f"Pull up the contact info for {n}",
    lambda n: f"Fetch the contact information for {n}",
    lambda n: f"Show me {n}'s contact info",
    lambda n: f"Retrieve contact details for {n}",
    lambda n: f"Search for {n} in my contacts",
    lambda n: f"Get me the contact details of {n}",
    lambda n: f"Look {n} up in my contact list",
    lambda n: f"Find {n} in my address book",
    lambda n: f"Pull {n}'s contact information",
    # First person need/want
    lambda n: f"I need the contact information for {n}",
    lambda n: f"I want to find contact details for {n}",
    lambda n: f"I need to look up {n}'s contact info",
    lambda n: f"I'm trying to find {n}'s contact details",
    lambda n: f"I need {n}'s phone number or email from my contacts",
    lambda n: f"I'd like to see {n}'s contact info",
    lambda n: f"I need you to look up {n} in my contacts",
    # Polite requests
    lambda n: f"Can you look up {n} in my contacts?",
    lambda n: f"Could you find {n}'s contact details?",
    lambda n: f"Would you pull up {n}'s contact info for me?",
    lambda n: f"Can you search for {n} in my contact list?",
    lambda n: f"Please find {n}'s contact information for me",
    lambda n: f"Can you get {n}'s contact details?",
    # Casual / shorthand
    lambda n: f"Contact info for {n} please",
    lambda n: f"contacts: {n}",
    lambda n: f"{n} contact info",
    lambda n: f"look up {n}",
    lambda n: f"find {n}",
    lambda n: f"{n}'s details please",
    # Descriptive / verbose
    lambda n: f"I need to get in touch with {n} but I don't have their info. Can you pull it from my contacts?",
    lambda n: f"What contact details do I have saved for {n}?",
    lambda n: f"Can you check my contacts and get the info for {n}?",
    lambda n: f"I'm looking for {n}'s contact information — can you find it?",
    lambda n: f"Please look through my contacts and find the details for {n}",
    lambda n: f"Do I have {n} in my contacts? If so, what are their details?",
    # Action-focused
    lambda n: f"Bring up {n}'s contact card",
    lambda n: f"Open {n}'s contact info",
    lambda n: f"Get {n}'s details from my phonebook",
    lambda n: f"Check my contacts for {n}",
]

clock_templates = [
    # Simple time queries
    "What time is it right now?",
    "What time is it?",
    "What's the time?",
    "Tell me the time",
    "What is the current time?",
    "What time is it currently?",
    "Can you tell me the current time?",
    "Could you give me the current time?",
    "I need to know the current time",
    "I want to know the current time",
    "Check the time for me",
    "Can you get the time for me?",
    "Give me the time",
    "What is the time right now?",
    "What is today's time?",
    "Do you know what time it is?",
    "time?",
    "current time please",
    "What time is it at the moment?",
    "Quick — what time is it?",
    # Date queries
    "What's today's date?",
    "What is today's date?",
    "What is the current date?",
    "What's the current date?",
    "Tell me today's date",
    "Give me today's date",
    "What day is it today?",
    "What day is it?",
    "What's the date today?",
    "What date is it?",
    "Can you check the current date for me?",
    "What's today?",
    "What day of the week is it?",
    "What is the date today?",
    "date?",
    "today's date?",
    "I need to know today's date",
    "Can you tell me what day it is?",
    "What month is it?",
    "What year is it?",
    # Date and time combined
    "What is the date and time?",
    "What is the current date and time?",
    "What's today's date and time?",
    "What day and time is it?",
    "Get the current date and time",
    "Give me the current date and time",
    "Show me the current date and time",
    "Tell me the current date and time",
    "What's the date and time right now?",
    "Can you give me the date and time?",
    "I need the current date and time",
    "What is the exact date and time?",
    "date and time please",
    "current date and time?",
    "What's the time and date?",
    # Indirect / conversational
    "Do you know what today's date is?",
    "Do you have the current time?",
    "What's the time looking like right now?",
    "I've lost track — what day is it?",
    "I forgot — what's today's date?",
    "What does the clock say?",
    "What time does your clock show?",
    "Is it morning or afternoon right now?",
    "Can you check what time it is?",
    "I need to know the time and date",
    "Can you pull up the current date and time for me?",
    "I'd like to know the current time",
    "Could you check today's date?",
    "What's the current timestamp?",
    "Give me today's timestamp",
    # Terse
    "time now",
    "current time",
    "today's date",
    "date now",
    "time and date",
    "clock?",
    "what day",
    "timestamp",
    "now?",
    "current datetime",
]

# ---------------------------------------------------------------------------
# GENERATION
# ---------------------------------------------------------------------------

def generate_tool_examples(n, template_fn, output_fn):
    results = []
    for _ in range(n):
        inp, out = template_fn()
        results.append({"input": inp, "output": out})
    return results

examples = []

# Zoom — 1700 examples
for _ in range(1700):
    t  = random.choice(meeting_topics)
    dt = rand_datetime()
    d  = rand_duration()
    tmpl = random.choice(zoom_templates)
    examples.append({"input": tmpl(t, dt, d), "output": f"<tool>zoom|schedule|{t}|{dt}|{d}</tool>"})

# Email — 1700 examples
for _ in range(1700):
    to = rand_email()
    s  = random.choice(email_subjects)
    b  = random.choice(email_bodies)
    tmpl = random.choice(email_templates)
    examples.append({"input": tmpl(to, s, b), "output": f"<tool>email|send|{to}|{s}|{b}</tool>"})

# Text — 1700 examples
for _ in range(1700):
    to = rand_phone()
    b  = random.choice(text_bodies)
    tmpl = random.choice(text_templates)
    examples.append({"input": tmpl(to, b), "output": f"<tool>text|send|{to}|{b}</tool>"})

# Calendar — 1700 examples
for _ in range(1700):
    dt = rand_datetime()
    d  = rand_duration()
    t  = random.choice(calendar_events)
    tmpl = random.choice(calendar_templates)
    examples.append({"input": tmpl(dt, d, t), "output": f"<tool>calendar|add_event|{dt}|{d}|{t}</tool>"})

# Contacts — 1700 examples
for _ in range(1700):
    n    = random.choice(first_names)
    tmpl = random.choice(contacts_templates)
    examples.append({"input": tmpl(n), "output": f"<tool>contacts|get_info|{n}</tool>"})

# Clock — 1500 examples (slightly fewer to keep round 10k total)
for _ in range(1500):
    inp = random.choice(clock_templates)
    examples.append({"input": inp, "output": "<tool>clock|get_date_time</tool>"})

# Shuffle and trim to exactly 10,000
random.shuffle(examples)
examples = examples[:10000]

# ---------------------------------------------------------------------------
# STATS
# ---------------------------------------------------------------------------
print(f"Total examples: {len(examples)}")
tools = [e["output"].split("|")[0].replace("<tool>","") for e in examples]
print("Distribution:", Counter(tools))

# ---------------------------------------------------------------------------
# WRITE OUTPUT FILES
# ---------------------------------------------------------------------------
json_path   = "/mnt/user-data/outputs/sft_training_data_10k.json"
script_path = "/mnt/user-data/outputs/generate_sft_10k.py"

with open(json_path, "w") as f:
    json.dump(examples, f, indent=2)

print(f"JSON saved to {json_path}")

# Copy this script to outputs
import shutil, os
shutil.copy(__file__, script_path)
print(f"Script saved to {script_path}")
