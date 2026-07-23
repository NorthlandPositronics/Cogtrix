# Work Breakdown Structure (WBS) — Project Nimbus

**Document ID:** NIMB-WBS-001
**Version:** 1.1
**Last updated:** 2026-08-12 (Month 4)

This document enumerates the 58 work items in the Nimbus programme. Each item has a stable ID (`NIMB-001` through `NIMB-058`), an owning squad, a current status, and a deliverable map. Cross-reference with `04_schedule_milestones.md` for dates and `08_raci_matrix.md` for responsibility.

---

## Legend

- **Status:** `done`, `in-progress`, `blocked`, `pending`, `descoped`.
- **Squad:** `Platform`, `Migration`, `Data`, `Networking`.
- **Deliv:** the deliverable from `02_scope_statement.md` this WBS item rolls up to.

---

## 1. Platform Squad — Landing Zone & Foundations (10 items)

| ID | Task | Status | Deliv |
|---|---|---|---|
| NIMB-001 | AcmeCloud organisation set-up (3 accounts: prod-eu, prod-us, prod-ap) | done | D-01 |
| NIMB-002 | Baseline IAM (root MFA, break-glass accounts, audit logging) | done | D-01 |
| NIMB-003 | VPC + subnet layout per region | done | D-01 |
| NIMB-004 | KMS keys + cross-region key replication | done | D-01 |
| NIMB-005 | Centralised logging (CloudTrail + log-archive account) | done | D-01 |
| NIMB-006 | OIDC federation from Okta to AcmeCloud IAM | done | D-03 |
| NIMB-007 | Service-account inventory and reconciliation | in-progress | D-03 |
| NIMB-008 | Baseline observability stack (Mimir cluster bootstrapped) | done | D-08 |
| NIMB-009 | Grafana managed instance + dashboard import | in-progress | D-08 |
| NIMB-010 | AlertManager wiring to PagerDuty + Slack | pending | D-08 |

## 2. Networking Squad — Inter-Region Connectivity (8 items)

| ID | Task | Status | Deliv |
|---|---|---|---|
| NIMB-011 | Transit Gateway (eu-central-1 hub) | done | D-02 |
| NIMB-012 | Transit Gateway peering (us-east-1, ap-southeast-1) | done | D-02 |
| NIMB-013 | Route-table baseline + propagation | done | D-02 |
| NIMB-014 | Inter-region latency benchmarking | done | D-02 |
| NIMB-015 | DNS migration plan (Route managed zones) | in-progress | D-02 |
| NIMB-016 | Customer-facing DNS cut-over | pending | D-02, D-10 |
| NIMB-017 | Bastion access via AcmeCloud Session Manager | done | D-01 |
| NIMB-018 | Failover drill (region-loss simulation) | pending | D-02 |

## 3. Data Squad — Database & Object Store (16 items)

| ID | Task | Status | Deliv |
|---|---|---|---|
| NIMB-019 | AcmeDB Enterprise cluster provisioning (eu-central-1 primary) | done | D-04 |
| NIMB-020 | Cross-region replica (us-east-1) | done | D-04 |
| NIMB-021 | Cross-region replica (ap-southeast-1) | in-progress | D-04 |
| NIMB-022 | Schema migration validation against PostgreSQL 15.4 source | done | D-04 |
| NIMB-023 | CDC pipeline (on-prem → AcmeDB) | in-progress | D-04 |
| NIMB-024 | CDC pipeline validation under load (1.5× production write rate) | pending | D-04 |
| NIMB-025 | Database cut-over rehearsal (rehearsal 1 of 3) | pending | D-04 |
| NIMB-026 | Database cut-over rehearsal (rehearsal 2 of 3) | pending | D-04 |
| NIMB-027 | Database cut-over rehearsal (rehearsal 3 of 3) | pending | D-04 |
| NIMB-028 | Database cut-over (production) | pending | D-04 |
| NIMB-029 | Object-store inventory + sizing (final: 14.06 TB) | done | D-05 |
| NIMB-030 | Object-store replication (eu-central-1 ↔ us-east-1) | done | D-05 |
| NIMB-031 | Object-store replication (eu-central-1 ↔ ap-southeast-1) | in-progress | D-05 |
| NIMB-032 | Per-bucket residency labels applied | pending | D-05 |
| NIMB-033 | Analytics pipeline (dbt) re-pointed at AcmeDB read replica | pending | D-07 |
| NIMB-034 | Analytics freshness SLO validation (14 consecutive days) | pending | D-07 |

## 4. Migration Squad — Applications (14 items)

| ID | Task | Status | Deliv |
|---|---|---|---|
| NIMB-035 | CompactSync-API containerisation review | done | D-06 |
| NIMB-036 | CompactSync-API deploy to AcmeCloud (eu-central-1, canary) | done | D-06 |
| NIMB-037 | CompactSync-API deploy to us-east-1 + ap-southeast-1 | in-progress | D-06 |
| NIMB-038 | CompactSync-API canary traffic at 5% (7-day window) | pending | D-06 |
| NIMB-039 | CompactSync-API full traffic cut-over | pending | D-06 |
| NIMB-040 | CompactSync-Portal CDN edge configuration | done | D-06 |
| NIMB-041 | CompactSync-Portal deploy to all three regions | in-progress | D-06 |
| NIMB-042 | CompactSync-Sync worker fleet rebuild on AcmeCloud | in-progress | D-06 |
| NIMB-043 | CompactSync-Sync soak test (72 hours at production load) | pending | D-06 |
| NIMB-044 | CompactSync-Identity migration | in-progress | D-06 |
| NIMB-045 | CompactSync-Search (Meilisearch) deploy and re-index | pending | D-07 |
| NIMB-046 | Per-tenant `region_hint` API surface (D-09 feature) | in-progress | D-09 |
| NIMB-047 | Per-tenant region pinning — data path enforcement | pending | D-09 |
| NIMB-048 | EU pilot tenant validation (Helmsdale Logistics, HL-2025-NOV) | pending | D-09 |

## 5. Programme-Level / Cross-Squad (10 items)

| ID | Task | Status | Deliv |
|---|---|---|---|
| NIMB-049 | Programme kick-off and squad onboarding | done | — |
| NIMB-050 | RACI matrix sign-off | done | — |
| NIMB-051 | Risk register baseline | done | — |
| NIMB-052 | Stakeholder communication plan | done | — |
| NIMB-053 | M1 status report (April) | done | — |
| NIMB-054 | M2 status report (May) | done | — |
| NIMB-055 | M3 status report (June + Q1 steering) | done | — |
| NIMB-056 | M4 status report (July + Q2 steering) | done | — |
| NIMB-057 | Programme retrospective and lessons learned | pending | — |
| NIMB-058 | Frankfurt facility de-rack and lease termination notice | pending | D-10 |

---

## Summary as of 2026-08-12 (Month 4)

- **Total items:** 58
- **Done:** 23 (40%)
- **In-progress:** 11 (19%)
- **Pending:** 24 (41%)
- **Blocked:** 0
- **Descoped:** 0

Status counts are consistent with the M4 status report (`13_status_report_m4.md`).

---

**References:** `02_scope_statement.md` (deliverable definitions), `04_schedule_milestones.md` (dates), `05_risk_register.md` (item-level risks), `08_raci_matrix.md` (squad ownership).
