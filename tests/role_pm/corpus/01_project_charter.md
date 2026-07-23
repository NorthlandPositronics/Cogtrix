# Project Charter — Project Nimbus

**Document ID:** NIMB-CHARTER-001
**Version:** 1.2
**Approved:** 2026-04-03
**Sponsor:** Tomislav Hessford, Chief Operating Officer
**Project Manager:** (AI PM Agent — see communication plan)

---

## 1. Project Name

Project Nimbus — Multi-Region Cloud Migration Program

## 2. Business Case

CompactSync, our flagship SaaS product, runs on a single-region on-premise deployment in the Frankfurt data centre (CompactSync-Prod-FRA1). The deployment has reached three structural limits:

- Capacity: peak CPU utilisation routinely exceeds 87% during the 13:00–16:00 UTC business-hours window; no room for the 18% YoY traffic growth the product team forecasts for FY27.
- Resilience: a single-region failure has zero customer-visible redundancy. The 2025-11-08 power incident at the Frankfurt facility caused a 4-hour 12-minute total outage. Customer-impact remediations cost the company $312,000 in SLA credits and 3 enterprise renewals.
- Compliance: two prospective enterprise customers (one US-based, one Singapore-based) have made multi-region data residency a contract precondition. We forfeited $2.1M ARR in 2025 because we could not satisfy this without a multi-region architecture.

The cloud migration is the agreed strategic response. Nimbus moves CompactSync to AcmeCloud across three regions (eu-central-1, us-east-1, ap-southeast-1) with active-active read traffic and active-passive write traffic, then decommissions the Frankfurt on-prem footprint.

## 3. Strategic Goals

The program delivers three strategic outcomes:

- **G1 — Capacity headroom:** support 3× current peak traffic without architectural rework.
- **G2 — Multi-region resilience:** sustain a single full-region failure with < 90-second customer-visible disruption.
- **G3 — Data residency compliance:** enable per-tenant region pinning for EU, US, and APAC customers, unblocking the deferred enterprise pipeline.

## 4. Scope (Summary)

Detailed scope is in `02_scope_statement.md`. At the charter level:

- **In scope:** CompactSync application services, the customer-facing API tier, the primary PostgreSQL data store, the object store (currently MinIO on-prem), the analytics pipeline, observability stack, identity service, and the customer-portal frontend.
- **Out of scope:** the internal Atlassian estate (Jira, Confluence, Bitbucket); the marketing-site WordPress instance; the development sandboxes (separate retirement programme); legacy reporting tooling slated for end-of-life Q1 FY27.

## 5. Success Criteria

Project Nimbus is considered successful when ALL of the following are demonstrated:

- **SC-1:** All in-scope workloads running on AcmeCloud across the three target regions, with on-prem Frankfurt traffic at 0% for a continuous 30-day window.
- **SC-2:** Multi-region failover drill completed with measured customer-visible disruption ≤ 90 seconds.
- **SC-3:** Per-tenant region pinning operational and validated by the EU pilot tenant (Helmsdale Logistics, contract ref HL-2025-NOV).
- **SC-4:** Total programme spend within ±5% of the approved $2,400,000 budget envelope.
- **SC-5:** Sign-off from the COO (Sponsor), CTO, Head of Security, Head of SRE, and Head of Customer Success.

## 6. High-Level Timeline

- **Start:** 2026-04-06 (Monday after charter approval)
- **End:** 2026-12-18
- **Duration:** 9 months (37 weeks of execution + 2 weeks of programme close)

Detailed milestones are in `04_schedule_milestones.md`.

## 7. Budget

Approved envelope: **$2,400,000**. Breakdown is in `07_budget.md`. The envelope covers AcmeCloud committed-use spend for the migration window, professional-services support from AcmeCloud, contractor support for the data-tier cut-over, observability tooling licences for the new estate, and a 12% contingency reserve held by the Sponsor.

## 8. Programme Structure

Nimbus is organised into four squads (detailed RACI in `08_raci_matrix.md`):

- **Platform Squad** — landing-zone, IAM, networking foundation, observability.
- **Migration Squad** — application workloads, customer-portal frontend, customer-API tier.
- **Data Squad** — PostgreSQL clusters, object store, analytics pipeline, residency controls.
- **Networking Squad** — inter-region links, DNS / traffic management, cut-over networking.

Total programme staffing: 18 FTEs across the four squads (4 / 6 / 5 / 3).

## 9. Key Stakeholders

The full stakeholder register is in `06_stakeholder_register.md`. Charter-level stakeholders:

- **Sponsor:** Tomislav Hessford (COO)
- **Senior Decision Authority:** Avantika Sundararaman (CTO)
- **Risk & Compliance:** Pernille Vrieze (Head of Security)
- **Operational Sign-off:** Bartholomew Okafor-Sing (Head of SRE)
- **Customer Success Sign-off:** Linnaea Korhonen (Head of Customer Success)
- **Vendor Relationship:** Yusuf Almasi (AcmeCloud TAM)

## 10. Authority and Constraints

The PM has standing authority to:

- Re-prioritise within-squad work to recover schedule slack.
- Approve change requests with cumulative budget impact under $20,000.
- Convene cross-squad sync meetings without prior approval.

The PM must escalate (to the Sponsor + CTO) any of:

- Scope changes that extend the timeline by > 2 weeks.
- Budget variance exceeding ±5% of the envelope.
- Risks rated High-High in the risk register (currently R-12 and R-19).
- SLA breaches by AcmeCloud.

## 11. Assumptions

- AcmeCloud capacity in eu-central-1 is reserved under SLA-2026-04-A3 from 2026-05-01.
- The Frankfurt landlord has confirmed lease termination is permitted from 2027-01-01 with 60 days' notice.
- No major product feature work will reduce engineering availability below the committed 18 FTE level during the migration window.
- The data residency feature in the application layer is a hard prerequisite for SC-3 and is delivered by the Migration Squad as deliverable D-09.

## 12. Out-of-Scope Reminders

The following are explicitly NOT Project Nimbus's responsibility, even though they are sometimes confused with it:

- Re-architecting CompactSync from a monolith to microservices (separate initiative, not approved).
- Migrating Atlassian — see exclusion in section 4.
- The annual SOC2 Type II audit (runs in parallel; security team owns).
- Customer pricing changes related to the new regions (Finance + Sales own).

## 13. Approval

This charter was approved by the Sponsor on 2026-04-03. Version 1.2 incorporates the 2026-04-02 amendment that raised the budget envelope from $2,200,000 to $2,400,000 to fund the dual-region database licences (AcmeDB Enterprise — see decision log entry DEC-2026-04-02-01 in `14_decision_log.md`).

---

**References:** `02_scope_statement.md`, `04_schedule_milestones.md`, `05_risk_register.md`, `06_stakeholder_register.md`, `07_budget.md`, `08_raci_matrix.md`, `14_decision_log.md`.
