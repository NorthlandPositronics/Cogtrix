# Budget — Project Nimbus

**Document ID:** NIMB-BUDG-001
**Version:** 2.3
**Last updated:** 2026-08-12 (end of Month 4)
**Owner:** Eberhard Lindqvist-Marais (Head of Finance), in concert with the Sponsor.

This is the authoritative budget tracking document. All amounts are USD. Burn is reported month-end. The envelope is the **$2,400,000** approved at version 1.2 of the charter (see `01_project_charter.md` section 13 and decision DEC-2026-04-02-01 in `14_decision_log.md`).

---

## 1. Envelope and Reserve

- **Approved envelope:** $2,400,000
- **Direct spend ceiling:** $2,112,000 (88% of envelope)
- **Contingency reserve:** $288,000 (12% of envelope, held by Sponsor)
- **PM standing approval authority:** cumulative variance up to $20,000 (charter section 10)
- **Sponsor approval required:** anything beyond standing authority.

## 2. Line-Item Budget vs Actuals (as of 2026-07-31)

| Line | Category | Budget (Plan) | Spent to M4 | Forecast at Programme End | Variance |
|---|---|---:|---:|---:|---:|
| L-01 | AcmeCloud committed-use spend (compute + network) | $720,000 | $314,200 | $735,000 | +$15,000 (+2.1%) |
| L-02 | AcmeCloud Object Storage | $96,000 | $38,900 | $97,500 | +$1,500 (+1.6%) |
| L-03 | AcmeDB Enterprise licences | $480,000 | $480,000 | $480,000 | 0 |
| L-04 | AcmeCloud Professional Services | $180,000 | $93,000 | $187,000 | +$7,000 (+3.9%) |
| L-05 | Contractor support (data cut-over specialists, 2 FTE × 5 months) | $260,000 | $104,000 | $260,000 | 0 |
| L-06 | Observability tooling licences (Grafana + Mimir managed) | $84,000 | $35,000 | $84,000 | 0 |
| L-07 | Training and certification (squad-wide AcmeCloud cert) | $36,000 | $32,000 | $36,000 | 0 |
| L-08 | Contingency reserve (Sponsor-held) | $288,000 | $0 | $48,000 | -$240,000 (-83%, reserve not yet drawn) |
| L-09 | Decommissioning (Frankfurt facility de-rack + transport) | $42,000 | $0 | $42,000 | 0 |
| L-10 | Programme administration (PM tooling, comms, retrospective) | $24,000 | $9,400 | $24,000 | 0 |
| L-11 | Buffer (unallocated) | $190,000 | $0 | $0 | -$190,000 (released to L-01 + L-04 forecast variance) |
| **Total** | | **$2,400,000** | **$1,106,500** | **$1,993,500 + $48,000 reserve = $2,041,500** | -$358,500 (-14.9%) |

## 3. Month-End Burn

| Month | Budget (Plan) | Spent | Variance |
|---|---:|---:|---:|
| M1 (April) | $310,000 | $298,400 | -$11,600 (-3.7%) |
| M2 (May) | $315,000 | $297,200 | -$17,800 (-5.7%) |
| M3 (June) | $278,000 | $259,800 | -$18,200 (-6.5%) |
| M4 (July) | $267,000 | $251,100 | -$15,900 (-6.0%) |
| **M1–M4 total** | **$1,170,000** | **$1,106,500** | **-$63,500 (-5.4%)** |

## 4. End-of-Programme Forecast

- **Forecast total spend (excluding reserve):** $1,993,500
- **Forecast envelope utilisation:** 83.1%
- **Forecast surplus (excluding reserve):** $118,500
- **Reserve forecast usage:** $48,000 of $288,000 (16.7% draw forecast)
- **Total forecast envelope draw:** $2,041,500 of $2,400,000 (85.1%)

The favourable variance is driven by three factors:

1. **L-11 (Buffer)** was unallocated at programme start and has now been released to forecasted overruns on L-01 (AcmeCloud commit) and L-04 (AcmeCloud ProServ) without exceeding the original line totals.
2. **M3 / M4 monthly burn** has been consistently 5–6% under plan as the M-LZ slip absorbed cost-side capacity that was already paid for.
3. **R-18** (AcmeDB licence overrun risk) has not materialised; architecture review confirmed the APAC region runs as a read-only replica without a separate licence.

## 5. Variance against SC-4 (±5%)

The programme is within the ±5% success criterion (SC-4 in `01_project_charter.md`) — favourable. Eberhard reports the same number weekly to the Sponsor.

## 6. Open Budget Decisions

- **D-BUD-2026-08-1:** whether to release part of the contingency reserve early to fund a third cut-over rehearsal (if R-12 mitigation requires it). Estimated cost: $24,000. Decision deadline: 2026-09-15. Pending.

## 7. Decision Log Linkage

- DEC-2026-04-02-01: budget envelope raise from $2,200,000 to $2,400,000 (drove L-03 to $480,000).
- DEC-2026-06-14-02: release L-11 buffer to L-01 / L-04 forecast variance.

Both decisions are documented in full in `14_decision_log.md`.

---

**References:** `01_project_charter.md`, `05_risk_register.md` (R-18), `14_decision_log.md`, `13_status_report_m4.md`.
