# Roadmap

## Layer A pilot (this repo, public data only)

| Week | Milestone | Status |
| --- | --- | --- |
| 1 | Repo, packaging, CI, API client, address harvester | done |
| 2 to 3 | Round-trip and balance reconstruction, bot filter, synthetic traders | done |
| 4 to 5 | All behavioural markers with tests; synthetic validation report | done |
| 5 | Harvest a first wallet sample and fetch 90 days of data | done (30 wallets) |
| 5 | Fix issues surfaced by real data: order sweeps, window capital, passive fills | done (v0.2) |
| 5 | Balance across account modes (unified account, portfolio margin), builder markets | done (v0.3) |
| 6 | Larger sample (several hundred wallets) | next |
| 6 | Layer A report on real wallets: prevalence, co-occurrence, profiles | |
| 6 | Harm link: do high-chasing wallets get liquidated more or lose more? | |
| 7 | Event study around a large market drop | |
| 7 to 8 | Working paper and preprint, tagged v1.0 release with a DOI | |

## Ideas that would strengthen the paper

- Survival analysis of time to ruin (balance under 10% of its peak) by marker level.
- Network analysis of the markers (do gambling-like signs form one connected cluster?).
- Compare small and large accounts: are gambling-like patterns concentrated in retail?
- Sensitivity of every result to the thresholds in `default_config.yaml`.

## Layer B (planned PhD study)

Consenting traders prove wallet ownership by signing a message and complete validated
questionnaires (TDS, PCTS, PGSI). This tests which markers actually track disordered
trading. See `docs/ethics.md`.
