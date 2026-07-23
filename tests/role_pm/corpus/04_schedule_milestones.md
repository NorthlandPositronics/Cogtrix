# Schedule and Milestones — Project Nimbus

**Document ID:** NIMB-SCHED-001
**Version:** 1.3
**Last updated:** 2026-08-12 (Month 4 — reflects the schedule recovery actions agreed at the Q2 steering meeting, `16_meeting_notes_steering_q2.md`)

This document is the canonical schedule for Nimbus. Dates are real (working) dates. Dependencies reference WBS IDs (`NIMB-NNN`) from `03_work_breakdown_structure.md`.

---

## 1. Programme-Level Dates

| Milestone | Original Target | Current Target | Variance |
|---|---|---|---|
| Programme start | 2026-04-06 | 2026-04-06 | — |
| Landing-zone complete (M-LZ) | 2026-05-22 | 2026-05-29 | +1 week |
| Inter-region networking complete (M-NET) | 2026-06-12 | 2026-06-19 | +1 week |
| Database cut-over rehearsal complete (M-DB-R) | 2026-09-04 | 2026-09-11 | +1 week |
| Application canary 5% live (M-APP-C) | 2026-10-02 | 2026-10-02 | — |
| Database cut-over (production) (M-DB-X) | 2026-10-16 | 2026-10-16 | — |
| Full traffic on AcmeCloud (M-CUT) | 2026-11-20 | 2026-11-20 | — |
| 30-day clean-window complete (M-CLEAN) | 2026-12-18 | 2026-12-18 | — |
| Frankfurt facility de-rack (M-DERACK) | 2026-12-18 | 2026-12-18 | — |

The two +1-week slips on M-LZ and M-NET are absorbed inside the schedule; neither extends programme end. The Q2 steering meeting (2026-07-09) approved the recovery plan that uses the parallelisation of NIMB-021 (cross-region replica AP) with NIMB-031 (object-store replication AP) to claw back the time.

## 2. Per-Milestone Detail

### M-LZ — Landing-Zone Complete

- **Target:** 2026-05-29 (was 2026-05-22 — +1 week slip absorbed)
- **Owning squad:** Platform
- **Status (as of M4):** done
- **WBS items rolled up:** NIMB-001 through NIMB-008
- **Acceptance:** see `02_scope_statement.md` D-01.
- **Notes:** the +1-week slip was caused by an unexpected Acme TAM rotation (Yusuf Almasi replaced the previous TAM mid-March), surfaced in M1's status report (`10_status_report_m1.md`).

### M-NET — Inter-Region Networking Complete

- **Target:** 2026-06-19 (was 2026-06-12 — +1 week slip absorbed)
- **Owning squad:** Networking
- **Status:** done
- **WBS:** NIMB-011 through NIMB-014, NIMB-017
- **Dependencies:** M-LZ
- **Acceptance:** see D-02.
- **Notes:** failover drill (NIMB-018) is scheduled separately for 2026-10-30.

### M-DB-R — Database Cut-Over Rehearsals Complete

- **Target:** 2026-09-11 (was 2026-09-04 — +1 week)
- **Owning squad:** Data
- **Status:** in-progress (rehearsal 1 of 3 complete as of 2026-08-08; rehearsals 2 + 3 scheduled for 2026-08-21 and 2026-09-04)
- **WBS:** NIMB-019 through NIMB-027
- **Dependencies:** M-LZ, M-NET
- **Acceptance:** all three rehearsals pass with cut-over duration ≤ 38 minutes (the production budget; rehearsal 1 came in at 51 minutes — see `13_status_report_m4.md` for the gap analysis).
- **Critical-path note:** this milestone IS the critical path for the programme. Any further slip from M4 onwards has 1:1 impact on the end date.

### M-APP-C — Application Canary 5% Live

- **Target:** 2026-10-02
- **Owning squad:** Migration
- **Status:** pending (NIMB-038)
- **WBS:** NIMB-035 through NIMB-038
- **Dependencies:** M-LZ, M-NET, M-DB-R (the canary runs against the rehearsed primary; production cut-over is M-DB-X)

### M-DB-X — Database Cut-Over (Production)

- **Target:** 2026-10-16
- **Owning squad:** Data
- **Status:** pending
- **WBS:** NIMB-028
- **Dependencies:** M-DB-R + at least one successful M-APP-C canary week.
- **Cut-over window:** 22:00 UTC Friday → 02:00 UTC Saturday (4-hour budgeted maintenance window negotiated with the top 20 enterprise tenants).

### M-CUT — Full Traffic on AcmeCloud

- **Target:** 2026-11-20
- **Owning squad:** Migration + Networking (joint)
- **WBS:** NIMB-016, NIMB-039, NIMB-041, NIMB-042, NIMB-044, NIMB-045
- **Dependencies:** M-DB-X complete; canary fully validated.

### M-CLEAN — 30-Day Clean Window Complete

- **Target:** 2026-12-18
- **Acceptance:** Frankfurt traffic at 0% for 30 consecutive days (SC-1).
- **Dependencies:** M-CUT.

### M-DERACK — Frankfurt Facility De-rack

- **Target:** 2026-12-18
- **WBS:** NIMB-058
- **Dependencies:** M-CLEAN.

## 3. Critical Path

The critical path through Nimbus runs:

```
M-LZ → M-NET → M-DB-R → M-DB-X → M-CUT → M-CLEAN → M-DERACK
```

The two slipped milestones (M-LZ +1w, M-NET +1w) have already been absorbed. The remaining float in the schedule sits between M-DB-X (2026-10-16) and M-CUT (2026-11-20) — 5 weeks of buffer for canary observation and any late surprises.

## 4. Risks to the Schedule

The risks most likely to threaten the critical path are documented in `05_risk_register.md`. The two High-High risks (R-12, R-19) both have schedule implications and are the standing agenda items for the Q3 steering meeting (2026-10-08).

---

**References:** `01_project_charter.md`, `03_work_breakdown_structure.md`, `05_risk_register.md`, `13_status_report_m4.md`, `16_meeting_notes_steering_q2.md`.
