# Ethics and privacy statement

## What this project does and does not do

- It studies **behaviour patterns across many wallets**. It does not diagnose anyone. A
  wallet that scores high on several markers is a pattern in public data, not a person
  with a disorder.
- It never publishes wallet addresses, wallet-level scores, or lists of "risky" wallets.
  Reports, figures and any shared dataset contain aggregates or salted pseudonyms only.
- It never tries to link a wallet to a real-world identity.

## Why care, if the data is public?

A blockchain address is pseudonymous, not anonymous. Under the EU GDPR it becomes
personal data as soon as it can be linked to a person, and behavioural scores about
gambling-like conduct are sensitive. The safeguards below follow the research
exemption (GDPR Art. 89): data minimisation, pseudonymisation and aggregate reporting.

## Safeguards built into the code

| Safeguard | Where |
| --- | --- |
| Raw data stays local; `data/` is git-ignored | `.gitignore` |
| Every derived table uses a salted hash, never the address | `storage.wallet_id`, `features.build_feature_table` |
| The salt is generated locally and never committed | `storage.get_salt`, `data/private/` |
| Reports print aggregates only | `report.py` |
| Synthetic wallets for all public demos and figures | `synthetic.py` |
| Polite API use (throttling, retries, caching) | `ingest/api_client.py` |

## Planned Layer B (not in this repo)

Linking consenting traders' wallets to validated questionnaires (Trading Disorder Scale,
PGSI, PCTS) involves human participants and mental-health data. That part will only run
after approval by a research ethics committee, with explicit consent, a data protection
impact assessment, and pre-registration. It is part of the planned PhD, not of this
public pilot.
