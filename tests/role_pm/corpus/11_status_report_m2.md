# Project Status Report — Project Nimbus — Month 2 (May 2026)

**Document ID:** NIMB-STATUS-M2
**Reporting Date:** 2026-06-02
**Reporting Period:** 2026-05-01 → 2026-05-31
**Overall Status:** Green
**Summary:** M-LZ delivered on the recovered 2026-05-29 date. Inter-region networking on track. R-12 contoured but not closed. M3 will focus on data-tier preparation and the first AcmeDB cross-region replica.

---

## Status Summary

| Area | Status | Notes |
|---|---|---|
| Scope | Green | No new change requests during M2. |
| Timeline | Green | M-LZ delivered 2026-05-29 as recovered plan committed. |
| Budget | Green | M2 spend $297,200 against $315,000 plan (-5.7%). Cumulative -4.7%. |
| Resources | Green | 18 FTEs, fully engaged. |
| Risks | Yellow | R-12 still open; R-15 (CDC saturation) newly opened. |
| Dependencies | Green | No upstream dependencies blocking. |

## Key Progress

- M-LZ delivered on 2026-05-29 (NIMB-001 through NIMB-005, NIMB-008 complete).
- OIDC federation operational (NIMB-006).
- Bastion access via Session Manager rolled out 2026-06-08 (NIMB-017 — note: completed slightly into M3 but reported here for narrative continuity).
- Transit Gateway hub up in eu-central-1 (NIMB-011).
- Transit Gateway peering established to us-east-1 (partial NIMB-012 — ap-southeast-1 peering scheduled for M3).
- R-12 mitigation actions agreed with AcmeCloud TAM: parallelism tuning + extra read-replicas + per-shard lag instrumentation.
- M2 status report (this document) published 2026-06-02 (NIMB-054).

## Current Blockers

- None.

## Upcoming Milestones (Next 60 Days)

| Milestone | Owner | Due Date | Status |
|---|---|---|---|
| M-NET — Inter-region networking complete | Vukašin Andrássy | 2026-06-19 | On track |
| M-DB-R — Database cut-over rehearsals complete | Beatriz Cazadora-Olesen | 2026-09-11 | At risk (Yellow — R-12) |

## Decisions Needed

| Decision | Owner | Needed By | Impact |
|---|---|---|---|
| Approve R-12 mitigation budget call-down (if needed) | Sponsor | 2026-09-15 (post-rehearsal 2) | Schedule + budget |

## Recommended Actions

- Data Squad to begin AcmeDB benchmarking under simulated 1.5× peak load by end of M3 (NIMB-024 scheduled 2026-08-25).
- PM to flag the Helmsdale acceptance window (R-19) at the Q1 steering meeting (2026-06-27).

## Risks Update

- R-12: open, mitigation in flight, weekly tracking.
- R-13 (APAC capacity): unchanged; firm-commitment date confirmed verbally for 2026-09-01.
- R-14: discovery sweep started 2026-05-19; 142 service accounts catalogued; 31 unowned.
- R-15 (CDC saturation): newly opened 2026-05-26 after preliminary sizing review.

## Variance Against Charter

- Schedule: M-LZ delivered on recovered plan. No further slips.
- Budget: tracking favourable (-4.7% MTD cumulative).
- Scope: unchanged.

---

**References:** `04_schedule_milestones.md`, `05_risk_register.md`, `07_budget.md`, `15_meeting_notes_steering_q1.md`.
