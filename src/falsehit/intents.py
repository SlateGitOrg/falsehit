"""The synthetic help-desk catalogue: 40 intents in 19 families, with paraphrase templates.

Intents in one family are near-misses of each other: they share every template and differ in a
single number, antonym or plan/object. Each family belongs to one `kind`, the class of near-miss
that cosine similarity cannot separate. The keyword table (`topic` and `match`, regular
expressions over normalized text) is what the fake upstream uses to pick a canned answer.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from functools import cache
from typing import Literal

from .keys import normalize_text

Kind = Literal["number", "negation", "object"]

SLOTS: dict[str, tuple[str, ...]] = {
    "greet": ("", "", "", "Hi, ", "Hello, ", "Hi there, "),
    "tail": ("", "", "", " please", " thanks", " thank you"),
    "my": ("my", "our"),
}


@dataclass(frozen=True)
class Variant:
    id: str
    surface: tuple[str, ...]
    """Alternative wordings for the `{k}` slot; every one must satisfy `match`."""
    match: tuple[str, ...]
    answer: str


@dataclass(frozen=True)
class Family:
    id: str
    kind: Kind
    category: str
    topic: tuple[str, ...]
    templates: tuple[str, ...]
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class Intent:
    id: str
    family: str
    kind: Kind
    category: str
    templates: tuple[str, ...]
    surface: tuple[str, ...]
    answer: str
    keywords: tuple[str, ...]

    @property
    def completion_tokens(self) -> int:
        """A fixed completion size per intent, proportional to its canned answer."""
        return 40 + math.ceil(len(self.answer) / 4)


def _v(id_: str, surface: tuple[str, ...], match: str, answer: str) -> Variant:
    return Variant(id_, surface, (match,), answer)


FAMILIES: tuple[Family, ...] = (
    # ------------------------------------------------------------------ number near-misses
    Family(
        "invoices_year",
        "number",
        "billing",
        (r"invoice",),
        (
            "{greet}how do I export {my} invoices for {k}{tail}?",
            "{greet}where can I download all invoices from {k}{tail}?",
            "{greet}I need a CSV of invoices for {k}{tail}",
            "{greet}can you export the invoices for {k}{tail}?",
            "{greet}export {k} invoices{tail}",
            "{greet}I want a copy of {my} {k} invoices{tail}",
            "{greet}how can I pull invoices for the year {k}{tail}?",
            "{greet}is there a way to bulk download invoices for {k}{tail}?",
            "{greet}need all {k} invoices as PDFs for the accountant{tail}",
            "{greet}what is the fastest way to get {k} invoices out of billing{tail}?",
        ),
        (
            _v(
                "2024",
                ("2024",),
                r"\b2024\b",
                "Open Billing, choose Invoices, set the date range to 2024 and press Export. "
                "You get a CSV, or a zip of PDFs, for every invoice dated in 2024.",
            ),
            _v(
                "2025",
                ("2025",),
                r"\b2025\b",
                "Open Billing, choose Invoices, set the date range to 2025 and press Export. "
                "Invoices for the current year are listed up to today and the export is "
                "regenerated each time you press it.",
            ),
        ),
    ),
    Family(
        "usage_report_year",
        "number",
        "analytics",
        (r"usage report",),
        (
            "{greet}where do I find the annual usage report for {k}{tail}?",
            "{greet}how do I download the usage report for {k}{tail}?",
            "{greet}can I get {my} usage report covering {k}{tail}?",
            "{greet}I need the {k} usage report for a budget review{tail}",
            "{greet}usage report {k}, where is it{tail}?",
            "{greet}how can I generate a usage report for the whole of {k}{tail}?",
            "{greet}is the usage report for {k} available as a spreadsheet{tail}?",
            "{greet}export the {k} usage report{tail}",
            "{greet}what is the way to see the usage report for {k}{tail}?",
        ),
        (
            _v(
                "2024",
                ("2024",),
                r"\b2024\b",
                "Go to Analytics, then Usage reports, and pick 2024 in the year selector. "
                "The report downloads as a spreadsheet with usage broken down by month.",
            ),
            _v(
                "2025",
                ("2025",),
                r"\b2025\b",
                "Go to Analytics, then Usage reports, and pick 2025 in the year selector. "
                "Months that have not closed yet show usage to date and are marked partial.",
            ),
        ),
    ),
    Family(
        "refund_window",
        "number",
        "billing",
        (r"refund",),
        (
            "{greet}can I get a refund within {k} days of paying{tail}?",
            "{greet}what is the refund policy if I ask within {k} days{tail}?",
            "{greet}I want a refund, it has been {k} days since I paid{tail}",
            "{greet}do you give a refund up to {k} days after purchase{tail}?",
            "{greet}how do I request a refund inside the {k} day window{tail}?",
            "{greet}is a refund possible {k} days after the charge{tail}?",
            "{greet}refund request, {k} days after payment{tail}",
            "{greet}am I still eligible for a refund after {k} days{tail}?",
            "{greet}how long do I have to ask for a refund, {k} days or more{tail}?",
        ),
        (
            _v(
                "14",
                ("14",),
                r"\b14\b",
                "Monthly plans can be refunded in full within 14 days of the charge. Open "
                "Billing, choose the payment and press Request refund.",
            ),
            _v(
                "30",
                ("30",),
                r"\b30\b",
                "Annual plans can be refunded in full within 30 days of the charge. Open "
                "Billing, choose the payment and press Request refund; after 30 days we "
                "credit the unused months instead.",
            ),
        ),
    ),
    Family(
        "seat_pricing",
        "number",
        "pricing",
        (r"\bseats?\b", r"price|pricing|cost|quote|pay|much|discount|run us"),
        (
            "{greet}how much do {k} seats cost per month{tail}?",
            "{greet}what is the price for {k} seats{tail}?",
            "{greet}we need {k} seats, what would we pay{tail}?",
            "{greet}can you quote me for {k} seats{tail}?",
            "{greet}pricing for a group of {k} seats{tail}",
            "{greet}cost of {k} seats on an annual contract{tail}?",
            "{greet}how much is it to buy {k} seats{tail}?",
            "{greet}{k} seats, what does that run us a year{tail}?",
            "{greet}is there a discount on {k} seats{tail}?",
        ),
        (
            _v(
                "10",
                ("10",),
                r"\b10\b",
                "Ten seats are billed at the list price per seat. Annual billing takes two "
                "months off and there is no volume discount below 25 seats.",
            ),
            _v(
                "50",
                ("50",),
                r"\b50\b",
                "Fifty seats qualify for the volume tier: 15 percent off the list price "
                "per seat, and a named account manager. Ask sales for a written quote.",
            ),
        ),
    ),
    Family(
        "file_recovery",
        "number",
        "files",
        (r"deleted file",),
        (
            "{greet}can I restore a deleted file within {k} days{tail}?",
            "{greet}how do I recover deleted files from the last {k} days{tail}?",
            "{greet}deleted file recovery, is {k} days the limit{tail}?",
            "{greet}is a deleted file still retrievable {k} days later{tail}?",
            "{greet}how long is a deleted file kept before it is gone for good, {k} days{tail}?",
            "{greet}where is the trash for a deleted file, and does it keep things {k} days{tail}?",
            "{greet}undo a deleted file after {k} days{tail}?",
            "{greet}retention for a deleted file, {k} days right{tail}?",
            "{greet}help me restore a deleted file, it was {k} days ago{tail}",
        ),
        (
            _v(
                "30",
                ("30",),
                r"\b30\b",
                "Deleted files stay in the Trash for 30 days. Open Trash, select the file "
                "and press Restore; after 30 days it is gone for good.",
            ),
            _v(
                "90",
                ("90",),
                r"\b90\b",
                "On Business storage, deleted files stay in the Trash for 90 days. Open "
                "Trash, select the file and press Restore; after 90 days it is gone for good.",
            ),
        ),
    ),
    Family(
        "api_rate_limit",
        "number",
        "developer",
        (r"rate limit",),
        (
            "{greet}how do I raise {my} API rate limit to {k} requests per minute{tail}?",
            "{greet}can the rate limit be increased to {k} requests a minute{tail}?",
            "{greet}we keep hitting the rate limit, we need {k} requests per minute{tail}",
            "{greet}what do I have to do to get a rate limit of {k} requests per minute{tail}?",
            "{greet}request a higher rate limit, {k} requests per minute{tail}",
            "{greet}is a rate limit of {k} requests per minute possible for us{tail}?",
            "{greet}who approves a rate limit change to {k} requests per minute{tail}?",
            "{greet}limit of {k} requests per minute, how do I ask for that rate limit{tail}?",
            "{greet}bump the rate limit to {k} rpm{tail}",
        ),
        (
            _v(
                "100",
                ("100",),
                r"\b100\b",
                "A limit of 100 requests per minute is self-service. Open Developer settings "
                "and set the rate limit; the change applies within five minutes.",
            ),
            _v(
                "1000",
                ("1000",),
                r"\b1000\b",
                "A limit of 1000 requests per minute needs approval. Open a Developer request, "
                "state the expected traffic and we answer within two business days.",
            ),
        ),
    ),
    # ------------------------------------------------------------- negation / antonym near-misses
    Family(
        "sso_toggle",
        "negation",
        "security",
        (r"\bsso\b|single sign on",),
        (
            "{greet}how do I {k} SSO for {my} company{tail}?",
            "{greet}where can I {k} single sign-on{tail}?",
            "{greet}I want to {k} SSO{tail}",
            "{greet}can you {k} SSO on {my} company{tail}?",
            "{greet}steps to {k} single sign-on for everyone{tail}",
            "{greet}what do I need to {k} SSO{tail}?",
            "{greet}how to {k} SSO login for all staff{tail}?",
            "{greet}is it possible to {k} single sign-on later{tail}?",
            "{greet}need to {k} SSO today, who can do that{tail}?",
        ),
        (
            _v(
                "enable",
                ("enable", "turn on", "activate"),
                r"\benable|\bturn on\b|\bactivate",
                "Go to Security, then Single sign-on, paste your identity provider metadata "
                "and switch the toggle on. Members sign in through your provider from then on.",
            ),
            _v(
                "disable",
                ("disable", "turn off", "deactivate"),
                r"\bdisable|\bturn off\b|\bdeactivate",
                "Go to Security, then Single sign-on, and switch the toggle off. Members fall "
                "back to email and password until you turn it on again.",
            ),
        ),
    ),
    Family(
        "two_factor_toggle",
        "negation",
        "security",
        (r"two factor|\b2fa\b",),
        (
            "{greet}how do I {k} two-factor authentication on {my} login{tail}?",
            "{greet}where do I {k} 2FA{tail}?",
            "{greet}I'd like to {k} 2FA for the whole company{tail}",
            "{greet}can admins {k} two-factor authentication for all users{tail}?",
            "{greet}steps to {k} 2FA{tail}",
            "{greet}what is the setting to {k} two-factor login{tail}?",
            "{greet}how can I {k} 2FA myself{tail}?",
            "{greet}is it safe to {k} two-factor authentication{tail}?",
        ),
        (
            _v(
                "enable",
                ("enable", "turn on", "activate"),
                r"\benable|\bturn on\b|\bactivate",
                "Open Security, choose Two-factor authentication and press Enable. Scan the "
                "QR code with an authenticator app and save the recovery codes.",
            ),
            _v(
                "disable",
                ("disable", "turn off", "deactivate"),
                r"\bdisable|\bturn off\b|\bdeactivate",
                "Open Security, choose Two-factor authentication and press Disable. You "
                "confirm with a current code, and the recovery codes stop working.",
            ),
        ),
    ),
    Family(
        "auto_renew",
        "negation",
        "billing",
        (r"auto renew",),
        (
            "{greet}how do I {k} auto-renewal{tail}?",
            "{greet}where is the setting to {k} auto-renew{tail}?",
            "{greet}I want to {k} auto-renewal before the next billing date{tail}",
            "{greet}can you {k} auto-renewal for {my} contract{tail}?",
            "{greet}{k} auto-renew{tail}",
            "{greet}what happens when I {k} auto-renewal{tail}?",
            "{greet}is there a way to {k} auto-renewal in the app{tail}?",
            "{greet}how to {k} auto-renewal on the contract{tail}?",
        ),
        (
            _v(
                "enable",
                ("turn on", "enable", "switch on"),
                r"\bturn on\b|\benable|\bswitch on\b",
                "Open Billing, then Contract, and switch Auto-renewal on. The contract renews "
                "on its end date at the current price and we email you seven days before.",
            ),
            _v(
                "disable",
                ("turn off", "disable", "switch off"),
                r"\bturn off\b|\bdisable|\bswitch off\b",
                "Open Billing, then Contract, and switch Auto-renewal off. The contract stays "
                "active until its end date and then lapses.",
            ),
        ),
    ),
    Family(
        "guest_access",
        "negation",
        "security",
        (r"guest", r"access"),
        (
            "{greet}how do I {k} guest access{tail}?",
            "{greet}where can admins {k} guest access to projects{tail}?",
            "{greet}I need to {k} guest access for contractors{tail}",
            "{greet}can you {k} guest access on {my} company{tail}?",
            "{greet}steps to {k} guest access{tail}",
            "{greet}what is the setting to {k} guest access{tail}?",
            "{greet}is there a toggle to {k} guest access company-wide{tail}?",
            "{greet}{k} guest access, how{tail}?",
        ),
        (
            _v(
                "allow",
                ("allow", "permit"),
                r"\ballow|\bpermit",
                "Open Admin settings, then Sharing, and set Guest access to Allowed. Guests "
                "can then be added to individual projects.",
            ),
            _v(
                "block",
                ("block", "deny"),
                r"\bblock|\bdeny",
                "Open Admin settings, then Sharing, and set Guest access to Blocked. Existing "
                "guests lose access at once and new guest requests are refused.",
            ),
        ),
    ),
    Family(
        "plan_change",
        "negation",
        "billing",
        (r"\bplan\b",),
        (
            "{greet}how do I {k} {my} plan{tail}?",
            "{greet}I want to {k} {my} plan today{tail}",
            "{greet}where can I {k} the plan{tail}?",
            "{greet}can I {k} {my} plan in the middle of the month{tail}?",
            "{greet}steps to {k} {my} plan{tail}",
            "{greet}what happens to billing if I {k} {my} plan{tail}?",
            "{greet}is it possible to {k} the plan online{tail}?",
            "{greet}how to {k} plan{tail}?",
        ),
        (
            _v(
                "upgrade",
                ("upgrade",),
                r"\bupgrade",
                "Open Billing, choose Change plan and pick a higher tier. The new plan "
                "starts immediately and we charge the prorated difference.",
            ),
            _v(
                "downgrade",
                ("downgrade",),
                r"\bdowngrade",
                "Open Billing, choose Change plan and pick a lower tier. The change takes "
                "effect at the end of the billing period and nothing is credited.",
            ),
        ),
    ),
    Family(
        "account_lock",
        "negation",
        "security",
        (r"account",),
        (
            "{greet}how do I {k} a user account{tail}?",
            "{greet}an admin needs to {k} an account, where is that{tail}?",
            "{greet}can I {k} {my} account{tail}?",
            "{greet}steps to {k} an account after failed logins{tail}",
            "{greet}where do I {k} an account in the admin panel{tail}?",
            "{greet}{k} account, how{tail}?",
            "{greet}is there a button to {k} an account{tail}?",
            "{greet}how can I {k} one account quickly{tail}?",
        ),
        (
            _v(
                "lock",
                ("lock", "freeze", "suspend"),
                r"\block|\bfreeze|\bsuspend",
                "Open Admin, then Members, select the account and press Lock. The member is "
                "signed out at once and cannot sign in until an admin restores it.",
            ),
            _v(
                "unlock",
                ("unlock", "unfreeze", "reinstate"),
                r"\bunlock|\bunfreeze|\breinstate",
                "Open Admin, then Members, select the locked account and press Unlock. The "
                "member can sign in again straight away.",
            ),
        ),
    ),
    Family(
        "member_change",
        "negation",
        "team",
        (r"\bmembers?\b",),
        (
            "{greet}how do I {k} a member{tail}?",
            "{greet}where can I {k} a member on {my} company{tail}?",
            "{greet}I need to {k} a member today{tail}",
            "{greet}steps to {k} a member{tail}",
            "{greet}can an admin {k} a member{tail}?",
            "{greet}what is the fastest way to {k} members{tail}?",
            "{greet}{k} member, where is the button{tail}?",
            "{greet}is there a limit when I {k} a member{tail}?",
        ),
        (
            _v(
                "add",
                ("add",),
                r"\badd",
                "Open Admin, then Members, press Add member and enter an email address. The "
                "member appears in the list once they accept the email.",
            ),
            _v(
                "remove",
                ("remove", "delete"),
                r"\bremove|\bdelete",
                "Open Admin, then Members, open the member's menu and press Remove. Their "
                "projects stay and are reassigned to you.",
            ),
        ),
    ),
    # --------------------------------------------------------------- plan / object near-misses
    Family(
        "plan_pricing",
        "object",
        "pricing",
        (r"price|pricing|cost|much|pay|charge",),
        (
            "{greet}how much does the {k} cost per month{tail}?",
            "{greet}what is the price of the {k}{tail}?",
            "{greet}{k} pricing{tail}",
            "{greet}can you tell me what the {k} costs{tail}?",
            "{greet}how much would we pay for the {k} each year{tail}?",
            "{greet}what do you charge for the {k}{tail}?",
            "{greet}looking for the monthly price of the {k}{tail}",
            "{greet}is the {k} cheaper if billed yearly, what is the price{tail}?",
            "{greet}{k}: what does it cost{tail}?",
        ),
        (
            _v(
                "starter",
                ("Starter plan", "Starter"),
                r"starter",
                "The Starter plan is 12 dollars per user per month, or 10 billed yearly. It "
                "includes five projects and email support.",
            ),
            _v(
                "pro",
                ("Pro plan", "Pro"),
                r"\bpro\b",
                "The Pro plan is 24 dollars per user per month, or 20 billed yearly. It adds "
                "unlimited projects, automations and priority support.",
            ),
            _v(
                "team",
                ("Team plan", "Team tier"),
                r"team (?:plan|tier)",
                "The Team plan is 39 dollars per user per month, or 33 billed yearly. It adds "
                "roles, audit history and shared dashboards.",
            ),
            _v(
                "enterprise",
                ("Enterprise plan", "Enterprise"),
                r"enterprise",
                "Enterprise is quoted per contract and includes single sign-on, a custom data "
                "retention period and a named success manager.",
            ),
        ),
    ),
    Family(
        "cancel_object",
        "object",
        "billing",
        (r"cancel",),
        (
            "{greet}how do I cancel {k}{tail}?",
            "{greet}I want to cancel {k}{tail}",
            "{greet}where can I cancel {k}{tail}?",
            "{greet}can you cancel {k} for me{tail}?",
            "{greet}steps to cancel {k}{tail}",
            "{greet}what is the process to cancel {k}{tail}?",
            "{greet}cancel {k}, how{tail}?",
            "{greet}is there a way to cancel {k} online{tail}?",
            "{greet}need to cancel {k} before the next charge{tail}",
        ),
        (
            _v(
                "subscription",
                ("my subscription", "our subscription", "the subscription"),
                r"subscription",
                "Open Billing, choose Cancel subscription and confirm. Access continues until "
                "the end of the paid period and no further charges are made.",
            ),
            _v(
                "seat",
                ("one seat", "a single seat", "a seat"),
                r"\bseat",
                "Open Admin, then Members, and press Release seat on the member you no longer "
                "need. The seat ends at the next renewal and you are billed for one fewer.",
            ),
        ),
    ),
    Family(
        "reset_object",
        "object",
        "account",
        (r"reset",),
        (
            "{greet}how do I reset {k}{tail}?",
            "{greet}I need to reset {k}{tail}",
            "{greet}where can I reset {k}{tail}?",
            "{greet}can you reset {k} for me{tail}?",
            "{greet}steps to reset {k}{tail}",
            "{greet}what is the way to reset {k}{tail}?",
            "{greet}reset {k}, how{tail}?",
            "{greet}is there a self-service option to reset {k}{tail}?",
        ),
        (
            _v(
                "password",
                ("my password", "the password"),
                r"password",
                "On the sign-in page choose Forgot password and enter your email. We send a "
                "link that stays valid for one hour.",
            ),
            _v(
                "api_key",
                ("my API key", "the API key"),
                r"api key",
                "Open Developer settings, select the key and press Rotate. The old key stops "
                "working after five minutes, so update your integrations first.",
            ),
        ),
    ),
    Family(
        "delete_object",
        "object",
        "account",
        (r"delete",),
        (
            "{greet}how do I delete {k}{tail}?",
            "{greet}I need to delete {k}{tail}",
            "{greet}where can I delete {k}{tail}?",
            "{greet}can you delete {k} for me{tail}?",
            "{greet}steps to permanently delete {k}{tail}",
            "{greet}what is the process to delete {k}{tail}?",
            "{greet}delete {k}, how{tail}?",
            "{greet}is there a way to delete {k} online{tail}?",
        ),
        (
            _v(
                "account",
                ("my account", "the account"),
                r"account",
                "Open Profile, choose Delete account and confirm with your password. "
                "Projects you own must be transferred first.",
            ),
            _v(
                "workspace",
                ("my workspace", "the workspace"),
                r"workspace",
                "Open Admin, then General, and press Delete workspace. Everything in it is "
                "erased after a 14 day grace period.",
            ),
        ),
    ),
    Family(
        "invite_target",
        "object",
        "team",
        (r"invit",),
        (
            "{greet}how do I invite {k} to {my} company{tail}?",
            "{greet}I want to invite {k}{tail}",
            "{greet}where can I invite {k}{tail}?",
            "{greet}can you invite {k} for me{tail}?",
            "{greet}steps to invite {k}{tail}",
            "{greet}what is the process to invite {k}{tail}?",
            "{greet}invite {k}, how{tail}?",
            "{greet}is there a link I can send to invite {k}{tail}?",
        ),
        (
            _v(
                "teammate",
                ("a teammate", "a coworker"),
                r"teammate|coworker",
                "Open Admin, then Members, press Invite and enter their work email. They "
                "receive a link and join with the Member role.",
            ),
            _v(
                "guest",
                ("a guest", "an outside guest"),
                r"guest",
                "Open the project, press Share and invite them by email as a Guest. Guests only "
                "see that project and never see billing.",
            ),
        ),
    ),
    Family(
        "export_object",
        "object",
        "reporting",
        (r"export",),
        (
            "{greet}how do I export {k}{tail}?",
            "{greet}I need to export {k}{tail}",
            "{greet}where can I export {k}{tail}?",
            "{greet}can you export {k} for me{tail}?",
            "{greet}steps to export {k}{tail}",
            "{greet}what is the way to export {k} as a CSV{tail}?",
            "{greet}export {k}, how{tail}?",
            "{greet}is there a button to export {k}{tail}?",
        ),
        (
            _v(
                "user_list",
                ("the user list", "a user list"),
                r"user list",
                "Open Admin, then Members, and press Export. You get a CSV with name, email, "
                "role and last sign-in.",
            ),
            _v(
                "audit_log",
                ("the audit log", "an audit log"),
                r"audit log",
                "Open Security, then Audit log, set a date range and press Export. The file "
                "is a CSV with one row per event, kept for twelve months.",
            ),
        ),
    ),
)

UNKNOWN_ANSWER = (
    "I am not sure about that one. I have passed your question to the support team and "
    "they will reply by email."
)
UNKNOWN_COMPLETION_TOKENS = 40


def _build() -> tuple[Intent, ...]:
    intents = []
    for family in FAMILIES:
        for variant in family.variants:
            intents.append(
                Intent(
                    id=f"{family.id}/{variant.id}",
                    family=family.id,
                    kind=family.kind,
                    category=family.category,
                    templates=family.templates,
                    surface=variant.surface,
                    answer=variant.answer,
                    keywords=family.topic + variant.match,
                )
            )
    return tuple(intents)


INTENTS: tuple[Intent, ...] = _build()
INTENT_BY_ID: dict[str, Intent] = {i.id: i for i in INTENTS}
ANSWER_TO_INTENT: dict[str, str] = {i.answer: i.id for i in INTENTS}


@cache
def _compiled(intent_id: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p) for p in INTENT_BY_ID[intent_id].keywords)


def classify(text: str) -> str | None:
    """The generator's keyword table: the intent whose keyword patterns all match, most specific
    first. Returns None when no intent matches."""
    normalized = normalize_text(text)
    best: tuple[int, str] | None = None
    for intent in INTENTS:
        patterns = _compiled(intent.id)
        if all(p.search(normalized) for p in patterns) and (
            best is None or len(patterns) > best[0]
        ):
            best = (len(patterns), intent.id)
    return None if best is None else best[1]


def render(intent: Intent, rng: random.Random, template_index: int | None = None) -> str:
    """One surface form of an intent: a template with every slot filled."""
    template = (
        intent.templates[template_index]
        if template_index is not None
        else rng.choice(intent.templates)
    )
    fills = {name: rng.choice(options) for name, options in SLOTS.items()}
    fills["k"] = rng.choice(intent.surface)
    return template.format(**fills).strip()
