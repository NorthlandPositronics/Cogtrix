# Project Status Report — Project Nimbus — Month 1 (April 2026)

**Document ID:** NIMB-STATUS-M1
**Reporting Date:** 2026-05-01
**Reporting Period:** 2026-04-06 → 2026-04-30
**Overall Status:** Yellow
**Summary:** Programme started on time. M-LZ projection slipped by 1 week due to AcmeCloud TAM rotation. Recovery plan in place; no end-date impact.

---

## Status Summary

| Area | Status | Notes |
|---|---|---|
| Scope | Green | Unchanged. CHG-NIMB-002 (OIDC federation) approved within PM authority. |
| Timeline | Yellow | M-LZ slip projected (+1 week). M-NET unaffected. |
| Budget | Green | M1 spend $298,400 against $310,000 plan (-3.7%). |
| Resources | Green | All 18 FTEs onboarded and operational. |
| Risks | Yellow | R-12 (replication lag) escalated to monitoring on 2026-04-22 after first benchmark spike. |
| Dependencies | Green | No upstream dependencies blocking. |

## Key Progress

- Programme kick-off completed 2026-04-06 (NIMB-049).
- RACI matrix approved by Sponsor + CTO on 2026-04-15 (NIMB-050).
- AcmeCloud organisation set-up complete across all three regions (NIMB-001).
- Baseline IAM configuration complete (NIMB-002).
- KMS keys + cross-region replication operational (NIMB-004).
- Risk register baseline approved 2026-04-18 (NIMB-051). 9 risks recorded at baseline; 3 added during M1 (R-12, R-13, R-14).
- Stakeholder communication plan published (NIMB-052, document `18_communication_plan.md`).

## Current Blockers

- None blocking, but the AcmeCloud TAM rotation (Yusuf Almasi onboarded 2026-04-12) cost the Platform Squad approximately 5 working days of velocity, which is the proximate cause of the M-LZ slip projection.

## Upcoming Milestones (Next 60 Days)

| Milestone | Owner | Due Date | Status |
|---|---|---|---|
| M-LZ — Landing-zone complete | Aldous Pemberton-Riggs | 2026-05-29 | At risk (Yellow) |
| M-NET — Inter-region networking complete | Vukašin Andrássy | 2026-06-19 | On track |

## Decisions Needed

| Decision | Owner | Needed By | Impact |
|---|---|---|---|
| Approve M-LZ +1 week slip absorption plan | Sponsor | 2026-05-15 | Schedule recovery |

## Recommended Actions

- PM to publish the M-LZ recovery plan by 2026-05-08, with the Sponsor's sign-off by 2026-05-15.
- Data Squad to begin AcmeDB benchmarking against on-prem write patterns to surface R-12 contours earlier.
- Platform Squad to re-baseline its M-LZ work to absorb the +1 week within squad-level float.

## Risks Update

- R-12 (replication lag): newly opened. First benchmark on 2026-04-22 showed an 8-second spike. Monitoring weekly.
- R-13 (APAC capacity): newly opened. Capacity confirmation pending from AcmeCloud TAM.
- R-14 (service-account inventory): newly opened. Discovery sweep deferred to M2.

## Variance Against Charter

- Schedule: M-LZ tracking +1 week vs charter baseline.
- Budget: tracking favourable (-3.7% MTD).
- Scope: unchanged.

---

**References:** `01_project_charter.md`, `04_schedule_milestones.md`, `05_risk_register.md`, `07_budget.md`.
