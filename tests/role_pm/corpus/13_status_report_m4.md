# Project Status Report — Project Nimbus — Month 4 (July 2026)

**Document ID:** NIMB-STATUS-M4
**Reporting Date:** 2026-08-04
**Reporting Period:** 2026-07-01 → 2026-07-31
**Overall Status:** Yellow
**Summary:** R-12 (replication lag) remains the critical-path risk. Rehearsal 1 ran 51 minutes vs the 38-minute budget — within tolerance but no slack. Q2 steering held 2026-07-09 (minutes: `16_meeting_notes_steering_q2.md`). Helmsdale acceptance window is the secondary watch.

---

## Status Summary

| Area | Status | Notes |
|---|---|---|
| Scope | Green | CHG-NIMB-003 (parallelisation) approved. CHG-NIMB-004 (defer customer notification to T-21) approved. |
| Timeline | Yellow | M-DB-R on track but no slack post-rehearsal 1. M-CUT / M-CLEAN / M-DERACK all unchanged. |
| Budget | Green | M4 spend $251,100 against $267,000 plan (-6.0%). Cumulative -5.4%. End-of-programme forecast at 85.1% envelope draw. |
| Resources | Green | 18 FTEs. Engineering reservation for D-09 commits 2026-09-01. |
| Risks | Yellow | R-12 still open. R-19 escalated to Sponsor at Q2 steering. R-13 firm-commitment date holds. |
| Dependencies | Green | No upstream blockers. |

## Key Progress

- Q2 steering meeting held 2026-07-09. Minutes in `16_meeting_notes_steering_q2.md`.
- Rehearsal 1 of database cut-over executed 2026-08-08 (NIMB-025). Duration: 51 minutes against the 38-minute production budget. Gap analysis attributes 7 minutes to a known checksum-verification step that can be parallelised and 6 minutes to per-shard lag during the freeze window (R-12 contour).
- Cross-region replica us-east-1 stable. p95 replication lag now 4.2s (was 5s at end of M3).
- Object-store replication eu-central-1 ↔ us-east-1 complete (NIMB-030). ap-southeast-1 still in flight (NIMB-031).
- CompactSync-API canary deploy live in eu-central-1 (NIMB-036). us-east-1 + ap-southeast-1 deploys started (NIMB-037).
- D-09 architectural design signed off by CTO 2026-07-30. Engineering kicks off 2026-09-01 per the CTO's reservation.
- Service-account discovery sweep: 18 unowned accounts at start of M4, 7 still unclaimed at end of M4. On track for 2026-09-12 auto-disable cut-off.

## Current Blockers

- None blocking, but rehearsal 1's 51-minute outcome eliminates schedule slack on the critical path. Rehearsal 2 (2026-08-21) must come in ≤ 38 minutes for confidence ahead of the production cut-over (2026-10-16). If rehearsal 2 also exceeds 38 minutes, PM will trigger CHG-NIMB-005 (third rehearsal, $24,000 from contingency).

## Upcoming Milestones (Next 60 Days)

| Milestone | Owner | Due Date | Status |
|---|---|---|---|
| Rehearsal 2 of database cut-over | Beatriz Cazadora-Olesen | 2026-08-21 | On track |
| Rehearsal 3 of database cut-over | Beatriz Cazadora-Olesen | 2026-09-04 | On track |
| M-DB-R (rehearsals complete) | Beatriz Cazadora-Olesen | 2026-09-11 | Yellow |
| CompactSync-API canary at 5% traffic (M-APP-C) | Hyeon-Jin Park | 2026-10-02 | On track |

## Decisions Needed

| Decision | Owner | Needed By | Impact |
|---|---|---|---|
| Approve CHG-NIMB-005 (third rehearsal) — conditional | Tomislav Hessford | 2026-09-15 | $24,000 contingency draw |

## Recommended Actions

- Data Squad to investigate parallelising the checksum-verification step ahead of rehearsal 2 (expected to recover 7 minutes).
- Data Squad to push p95 lag from 4.2s to ≤ 3s by 2026-09-04 (rehearsal 2 deadline).
- Customer Success to begin Helmsdale validation-window dry-run conversations from 2026-09-15.
- PM to publish Q2 steering minutes (done — `16_meeting_notes_steering_q2.md`).

## Risks Update

- R-12 (replication lag): p95 4.2s → target 3.0s. Mitigation actions in flight.
- R-13 (APAC capacity): written confirmation received 2026-07-31. Risk downgraded to Low-High at the M5 review.
- R-14 (service-account inventory): 7 unowned accounts remaining; on track.
- R-15 (CDC saturation): test 2026-08-25 still pending.
- R-17 (observability gap): shadow-alerting bridge prototype complete; full rollout scheduled for 2026-10-01.
- R-19 (Helmsdale): escalated. Contingency approved. Customer Success leading dry-run.

## Variance Against Charter

- Schedule: critical-path milestone (M-DB-R) tracking +1 week with no further slack. Programme end-date unchanged.
- Budget: -5.4% cumulative. End-of-programme forecast: $2,041,500 of $2,400,000 envelope (85.1% draw including 16.7% reserve draw).
- Scope: unchanged.

## Programme-Level Counts (as of 2026-07-31)

- WBS items: 58 total. Done: 23 (40%). In-progress: 11 (19%). Pending: 24 (41%). Blocked: 0. Descoped: 0.
- Risks: 12 open. High-High: 2 (R-12, R-19). Medium-High: 2. Medium-Medium: 4. Low-Medium: 3 (R-20 mitigated, R-21 mitigated, R-22 open). Low-Low: 1.
- Change requests: 4 approved, 1 pending, 1 rejected, 1 withdrawn.

---

**References:** `01_project_charter.md`, `03_work_breakdown_structure.md`, `04_schedule_milestones.md`, `05_risk_register.md`, `07_budget.md`, `09_change_log.md`, `16_meeting_notes_steering_q2.md`.
