"""
Generates a labeled synthetic dataset of support tickets for training/evaluating
the baseline classifier. Categories and priorities are assigned deterministically
based on template + keyword variation so the dataset is reproducible.

Run: python generate_dataset.py
Outputs: tickets.csv (text, category, priority)
"""
import csv
import random

random.seed(42)

CATEGORIES = ["billing", "technical", "account", "feature_request", "general_inquiry"]
PRIORITIES = ["low", "medium", "high", "urgent"]

TEMPLATES = {
    "billing": [
        ("I was charged twice for my subscription this month, please refund the duplicate charge.", "high"),
        ("My invoice shows the wrong amount, can you correct it?", "medium"),
        ("I want to cancel my subscription and get a refund for this billing cycle.", "medium"),
        ("Can you explain what this extra charge on my bill is for?", "low"),
        ("URGENT: I was billed $500 instead of $50, this needs to be fixed immediately.", "urgent"),
        ("How do I update my credit card on file?", "low"),
        ("My payment failed but I was still charged, please investigate.", "high"),
        ("I'd like to switch from monthly to annual billing.", "low"),
        ("The coupon code I used didn't apply the discount to my order.", "medium"),
        ("I need a receipt for my last three payments for tax purposes.", "low"),
    ],
    "technical": [
        ("The app crashes every time I try to upload a file larger than 10MB.", "high"),
        ("I'm getting a 500 error when I try to log in, this is blocking my whole team.", "urgent"),
        ("The dashboard is loading very slowly today, taking over 30 seconds.", "medium"),
        ("Search results are not returning any results even for common queries.", "high"),
        ("The mobile app keeps logging me out every few minutes.", "medium"),
        ("I can't reset my password, the reset email never arrives.", "high"),
        ("There's a small UI glitch where the button text overlaps on mobile.", "low"),
        ("API requests are returning 429 rate limit errors constantly, blocking production.", "urgent"),
        ("The export to PDF feature produces a corrupted file every time.", "medium"),
        ("Notifications aren't showing up even though they're enabled in settings.", "low"),
    ],
    "account": [
        ("I need to change the email address associated with my account.", "low"),
        ("I'm locked out of my account and need help verifying my identity.", "high"),
        ("Can you merge my two duplicate accounts into one?", "medium"),
        ("I want to delete my account and all associated data permanently.", "medium"),
        ("How do I add a new team member to my organization's account?", "low"),
        ("My account was suspended without explanation, I need this resolved urgently.", "urgent"),
        ("I need to transfer account ownership to a colleague.", "medium"),
        ("Two-factor authentication isn't sending me codes, I can't get in.", "high"),
        ("Please update the company name listed on my account profile.", "low"),
        ("I suspect someone else has accessed my account without permission.", "urgent"),
    ],
    "feature_request": [
        ("It would be great if the app supported dark mode.", "low"),
        ("Can you add the ability to export data to Excel, not just CSV?", "low"),
        ("Please consider adding integration with Slack for notifications.", "low"),
        ("We'd love a bulk-edit feature for managing many records at once.", "medium"),
        ("Could you add keyboard shortcuts for common actions?", "low"),
        ("Requesting support for multi-language interfaces.", "low"),
        ("Adding an undo button would really help prevent accidental deletions.", "medium"),
        ("Can the calendar view support week and day views, not just month?", "low"),
        ("It would help to have customizable dashboard widgets.", "low"),
        ("Please add single sign-on (SSO) support for enterprise customers.", "medium"),
    ],
    "general_inquiry": [
        ("What are your customer support hours?", "low"),
        ("Do you offer discounts for non-profit organizations?", "low"),
        ("I'm considering your product, can you tell me more about the enterprise plan?", "low"),
        ("Is there a mobile app available for iOS?", "low"),
        ("What's the difference between the pro and premium plans?", "low"),
        ("Do you have an affiliate or referral program?", "low"),
        ("Can I get a demo of the product before purchasing?", "low"),
        ("What regions do you currently support for data hosting?", "low"),
        ("Is there a public API and where can I find documentation?", "low"),
        ("Just wanted to say the onboarding experience was great, thank you!", "low"),
    ],
}

VARIATION_PREFIXES = ["", "Hi team, ", "Hello, ", "Hi, ", "To whom it may concern, ", "Quick question - "]
VARIATION_SUFFIXES = ["", " Thanks.", " Please advise.", " Appreciate the help.", " Let me know."]


def generate_rows(n_per_category=40):
    rows = []
    for category, examples in TEMPLATES.items():
        for i in range(n_per_category):
            base_text, priority = random.choice(examples)
            prefix = random.choice(VARIATION_PREFIXES)
            suffix = random.choice(VARIATION_SUFFIXES)
            text = f"{prefix}{base_text}{suffix}".strip()
            rows.append({"text": text, "category": category, "priority": priority})
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = generate_rows(n_per_category=40)
    with open("tickets.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category", "priority"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} labeled tickets -> tickets.csv")
