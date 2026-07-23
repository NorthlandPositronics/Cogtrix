# Risk Register — Project Nimbus

**Document ID:** NIMB-RISK-001
**Version:** 2.1
**Last updated:** 2026-08-12 (Month 4)

12 risks are currently tracked. IDs run from R-12 to R-23 (R-1 through R-11 belonged to the prior Frankfurt-resilience programme and are closed; the numbering is continuous across the COO's risk portfolio).

Each risk carries:

- **Probability** and **Impact** on a Low / Medium / High scale.
- An **Owner** (the individual accountable, not the squad).
- A current **Status:** `open`, `monitoring`, `mitigated`, `escalated`, `closed`.
- A **Mitigation** and, where applicable, a **Contingency**.

---

## High-High Risks (Escalated)

### R-12 — AcmeDB cross-region replication lag under peak load

- **Probability:** High
- **Impact:** High
- **Owner:** Beatriz Cazadora-Olesen (Data Squad Lead)
- **Status:** escalated (raised to Sponsor + CTO at the Q2 steering, 2026-07-09)
- **Description:** Pre-production benchmarks show AcmeDB cross-region replication lag occasionally spiking to 11 seconds
  during peak write windows (versus the 2-second SLO baked into deliverable D-04 acceptance). Sustained high lag during
  the cut-over window would force either (a) extending the maintenance window past the negotiated 4 hours or
  (b) accepting up to 11 seconds of data loss in a failover scenario, neither of which is acceptable to the EU pilot
  tenant (Helmsdale Logistics).
- **Mitigation:** AcmeCloud TAM (Yusuf Almasi) engaged. Three actions in flight: (i) tuning the replication parallelism setting from 4 to 8 on the eu-central-1 primary, (ii) adding two AcmeDB read-replica nodes in us-east-1 to spread the load, (iii) instrumenting per-shard lag metrics so the Data Squad can spot regressions in real time.
- **Contingency:** If the tuning + extra replicas do not reduce p95 lag below 3 seconds by 2026-09-04 (rehearsal 2 deadline), the cut-over date will be re-baselined and escalated to the Sponsor. Decision deadline: 2026-09-15.

### R-19 — Helmsdale Logistics data-residency acceptance window

- **Probability:** High
- **Impact:** High
- **Owner:** Linnaea Korhonen (Head of Customer Success)
- **Status:** escalated (raised to Sponsor at the Q2 steering)
- **Description:** Helmsdale Logistics (HL-2025-NOV) has a contractual right to validate per-tenant data-residency
  end-to-end within a 21-day acceptance window once we make the feature available. The current schedule has D-09
  (NIMB-046, NIMB-047, NIMB-048) finishing on 2026-11-12. That puts the Helmsdale 21-day validation window inside the
  holiday slow-down, when their compliance team is operating at half capacity. If they cannot complete validation by
  their 2026-12-03 deadline they may re-open contract negotiations.
- **Mitigation:** Customer Success has secured a written commitment from Helmsdale's COO to accelerate the validation to a 14-day window starting 2026-11-09, conditional on us delivering D-09 one week ahead of the published schedule. Migration Squad is investigating whether NIMB-046 + NIMB-047 can be parallelised to recover the week.
- **Contingency:** If we cannot deliver D-09 by 2026-11-05, Customer Success will offer Helmsdale a written commitment to honour the original contract pricing through Q1 FY27 even if the formal acceptance slips. Sponsor approval for this commitment was granted at the Q2 steering (see `16_meeting_notes_steering_q2.md`).

---

## High-Medium Risks (Monitoring)

### R-13 — AcmeCloud capacity reservation in ap-southeast-1

- **Probability:** Medium
- **Impact:** High
- **Owner:** Tomislav Hessford (Sponsor — delegated to Yusuf Almasi for operational tracking)
- **Status:** monitoring
- **Description:** SLA-2026-04-A3 reserves capacity for eu-central-1 and us-east-1 but only commits "best-effort" for ap-southeast-1 until 2026-09-01. If AcmeCloud is supply-constrained on the date we need to provision NIMB-021 + NIMB-031 in ap-southeast-1, we may face a 2–3 week delay.
- **Mitigation:** TAM has confirmed verbally that the reservation will convert to a hard commitment from 2026-09-01. We have asked for written confirmation by 2026-08-31.
- **Contingency:** If reservation is not confirmed, ap-southeast-1 deployment slips and we deliver SC-3 (per-tenant residency) for EU + US only at programme close, with APAC following in Q1 FY27. Sponsor pre-approved this scope flex at the Q2 steering.

### R-14 — Service-account inventory completeness

- **Probability:** Medium
- **Impact:** High
- **Owner:** Aldous Pemberton-Riggs (Platform Squad)
- **Status:** monitoring
- **Description:** NIMB-007 (service-account reconciliation) is in-progress. The current inventory has 142 accounts on the on-prem cluster; 31 of these still lack a clearly identified owning service. Cutting over with unowned accounts risks either (a) silently breaking integrations or (b) over-permissioning the new estate.
- **Mitigation:** Platform Squad is running a 4-week discovery sweep (started 2026-07-15). The sweep auto-disables any account that has not been claimed by 2026-09-12.
- **Contingency:** Unclaimed accounts at the cut-off are disabled, with rollback plan documented per account.

---

## Medium-Medium Risks (Open)

### R-15 — CDC pipeline saturation under peak write load

- **Probability:** Medium
- **Impact:** Medium
- **Owner:** Beatriz Cazadora-Olesen
- **Status:** open
- **Description:** NIMB-023 (CDC pipeline on-prem → AcmeDB) is sized for the current average write rate (4,200 writes/sec). Peak production write rate is 6,800 writes/sec. NIMB-024 (load validation at 1.5× production) is pending; until then we cannot confirm CDC won't fall behind during peak.
- **Mitigation:** NIMB-024 is scheduled for 2026-08-25. Sizing buffer will be added if the test fails.

### R-16 — Engineering staff bandwidth around D-09 (region pinning)

- **Probability:** Medium
- **Impact:** Medium
- **Owner:** Avantika Sundararaman (CTO)
- **Status:** open
- **Description:** D-09 is the one in-scope application-level feature, and it requires deep changes to the API request-routing layer. Two of the three engineers with the required familiarity are also assigned to the Q3 product roadmap (unrelated to Nimbus). If Q3 product work runs over, D-09 risks slipping.
- **Mitigation:** CTO has signed off that Nimbus has first call on these engineers' time between 2026-09-01 and 2026-11-12. Tracked weekly at the Nimbus PM stand-up.

### R-17 — Observability gap during cut-over

- **Probability:** Medium
- **Impact:** Medium
- **Owner:** Bartholomew Okafor-Sing (Head of SRE)
- **Status:** open
- **Description:** Decommissioning the legacy VictoriaMetrics estate (NIMB-008/NIMB-009 transition) before the cut-over leaves a 36-hour window where some legacy alerts will not have a fresh data source. If a customer incident happens in that window, we may not see it as quickly as we would today.
- **Mitigation:** A shadow-alerting bridge is being built (Platform Squad) that lets the legacy alert definitions consume Mimir data via the OpenMetrics endpoint until the cut-over.

### R-18 — Cost overrun on AcmeDB Enterprise licences

- **Probability:** Medium
- **Impact:** Medium
- **Owner:** Tomislav Hessford
- **Status:** open
- **Description:** The dual-region database licences (DEC-2026-04-02-01 in `14_decision_log.md`) drove the budget envelope from $2.2M to $2.4M. If a third region (APAC) requires its own licence rather than reading from an existing one, we add $180,000 to the spend.
- **Mitigation:** Architecture review confirms the APAC region can run as a read-only replica without a separate licence, but the cluster sizing has to be re-validated post-cut-over. Finance has agreed to a $200,000 contingency call-down if the read-replica model proves insufficient.

---

## Low-Medium and Low-Low Risks (Monitoring)

### R-20 — Bastion access regression during Session Manager rollout

- **Probability:** Low
- **Impact:** Medium
- **Owner:** Aldous Pemberton-Riggs
- **Status:** mitigated (rolled out 2026-06-08, no regressions reported in M2 status)

### R-21 — Customer-portal CDN cache invalidation latency

- **Probability:** Low
- **Impact:** Medium
- **Owner:** Hyeon-Jin Park (Migration Squad)
- **Status:** mitigated (load test 2026-07-22 showed p99 cache invalidation at 4.3 seconds, well under the 30-second budget)

### R-22 — Frankfurt landlord lease termination dispute

- **Probability:** Low
- **Impact:** Medium
- **Owner:** Tomislav Hessford
- **Status:** open
- **Description:** The lease termination clause requires 60 days' notice from 2027-01-01, which is post-programme. Risk is only if M-DERACK slips past 2026-12-18 and we miss the notice window. Mitigation: PM tracks M-DERACK weekly; if it shifts more than 1 week, Sponsor + Legal are notified immediately.

### R-23 — Documentation lag for the new estate

- **Probability:** Low
- **Impact:** Low
- **Owner:** PM (no individual; rolling action)
- **Status:** open
- **Description:** Runbooks and architecture diagrams for the new estate are being authored alongside the migration. There is a real risk that on cut-over day the SRE on-call will not have current documentation for some workloads. Mitigation: documentation is a hard gate on each squad sign-off (see `02_scope_statement.md` section 6).

---

## Risk Matrix (Summary)

| | Low Impact | Medium Impact | High Impact |
|---|---|---|---|
| **High Probability** | — | — | R-12, R-19 |
| **Medium Probability** | — | R-15, R-16, R-17, R-18 | R-13, R-14 |
| **Low Probability** | R-23 | R-20, R-21, R-22 | — |

**Counts:** High-High: 2 (R-12, R-19). Medium-High: 2 (R-13, R-14). Medium-Medium: 4 (R-15–R-18). Low-Medium: 3 (R-20–R-22). Low-Low: 1 (R-23).

---

**References:** `01_project_charter.md` (escalation thresholds), `04_schedule_milestones.md` (schedule impact), `13_status_report_m4.md`, `16_meeting_notes_steering_q2.md`, `14_decision_log.md`.
