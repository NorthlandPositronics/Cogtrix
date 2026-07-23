# Scope Statement — Project Nimbus

**Document ID:** NIMB-SCOPE-001
**Version:** 1.0
**Approved:** 2026-04-08

This document expands the high-level scope summary in `01_project_charter.md` into auditable in-scope / out-of-scope / deliverable lists.

---

## 1. In-Scope Workloads

The following CompactSync workloads are migrated to AcmeCloud as part of Project Nimbus:

- **CompactSync-API** (Go, 26 endpoints) — customer-facing REST API tier.
- **CompactSync-Portal** (TypeScript / Next.js) — customer-portal frontend, served via AcmeCloud CDN edge.
- **CompactSync-Sync** (Go) — the document-sync worker fleet (currently 12 nodes on-prem).
- **CompactSync-Identity** (Rust) — OIDC identity service.
- **CompactSync-Analytics** (Python + dbt) — nightly analytics pipeline, currently writing to the on-prem PostgreSQL replica.
- **CompactSync-Observability** (OpenTelemetry collector + Grafana + Mimir) — replaced with AcmeCloud-managed equivalents.
- **CompactSync-Search** (Meilisearch v1.9) — customer search index.

## 2. In-Scope Data Stores

- **Primary PostgreSQL cluster** (currently single-region, version 15.4) — migrated to AcmeDB Enterprise multi-region with cross-region read replicas.
- **Object store** (currently MinIO, ~14 TB) — migrated to AcmeCloud Object Storage with cross-region replication.
- **Time-series metrics store** (currently VictoriaMetrics) — replaced with AcmeCloud-managed Mimir.

## 3. In-Scope Networking and Identity

- New landing-zone VPC topology across eu-central-1, us-east-1, ap-southeast-1.
- VPC peering and Transit Gateway across the three regions.
- Customer-facing DNS migration to AcmeCloud Route managed zones.
- Bastion access via AcmeCloud Session Manager (legacy SSH bastion is decommissioned).
- Service-account IAM consolidated onto AcmeCloud IAM with OIDC federation back to the corporate Okta tenant.

## 4. Out of Scope

Project Nimbus explicitly EXCLUDES the following:

- **OS-1:** Internal Atlassian estate (Jira, Confluence, Bitbucket) — remains on the on-prem Frankfurt facility for the duration of Nimbus. A separate programme (TBD) will retire it.
- **OS-2:** Marketing-site WordPress instance — remains on the existing managed-WordPress provider.
- **OS-3:** Development sandboxes (Compact-Dev-1 through Compact-Dev-7) — separate retirement programme, owned by Engineering Operations.
- **OS-4:** Legacy reporting tooling (Pentaho dashboards) — slated for end-of-life in Q1 FY27 regardless of Nimbus.
- **OS-5:** Re-architecting CompactSync from monolith to microservices — explicitly NOT a Nimbus deliverable. The migration is a lift-and-modernise (not lift-and-shift, not full rewrite).
- **OS-6:** Customer pricing changes related to multi-region availability — Finance and Sales own.
- **OS-7:** Annual SOC2 Type II audit — runs in parallel under the security team's ownership.
- **OS-8:** Application-level feature work unrelated to migration. The data-residency feature (deliverable D-09) is the ONE in-scope feature; everything else stays on the standard product roadmap.

## 5. Deliverables

| ID | Deliverable | Owner Squad | Acceptance Criteria |
|---|---|---|---|
| D-01 | Landing-zone foundation (3 regions) | Platform | All four AcmeCloud accounts provisioned; baseline IAM + networking + observability validated; security review signed off. |
| D-02 | Inter-region networking | Networking | Transit Gateway operational; < 35 ms p50 inter-region latency; failover drill passed. |
| D-03 | Identity and OIDC federation | Platform | OIDC federation from Okta to AcmeCloud IAM operational; service-account inventory reconciled. |
| D-04 | Database migration (PostgreSQL → AcmeDB Enterprise) | Data | Cross-region cluster operational; CDC replication validated; cut-over rehearsed end-to-end. |
| D-05 | Object-store migration (MinIO → AcmeCloud Object Storage) | Data | All 14 TB replicated; checksum validation passed; per-bucket residency labels applied. |
| D-06 | Application migration (API + Portal + Sync) | Migration | All three services running on AcmeCloud across the three regions; canary traffic 5% for 7 days then full cut-over. |
| D-07 | Search and analytics workloads | Migration | Meilisearch + dbt analytics running on AcmeCloud; freshness SLO met for 14 consecutive days. |
| D-08 | Observability cut-over | Platform | Mimir-backed dashboards live; legacy VictoriaMetrics decommissioned; alerting routed via AcmeCloud-managed AlertManager. |
| D-09 | Per-tenant region pinning feature | Migration | API surface accepts `region_hint` per tenant; data path honours the hint; EU pilot tenant (Helmsdale Logistics) validated end-to-end. |
| D-10 | Cut-over and decommission | All squads | On-prem traffic at 0% for 30 consecutive days; Frankfurt facility de-racked. |

Each deliverable maps to a section of the WBS in `03_work_breakdown_structure.md` and to specific milestones in `04_schedule_milestones.md`.

## 6. Acceptance Process

Each deliverable goes through three checkpoints:

1. **Squad sign-off:** the owning squad demonstrates the deliverable against its acceptance criteria.
2. **Cross-squad review:** dependent squads validate that integration points work end-to-end.
3. **Sponsor sign-off:** the COO Sponsor approves on the basis of the squad and cross-squad reviews. SC-1 through SC-5 (charter section 5) are evaluated at programme close, not per-deliverable.

## 7. Scope-Change Process

Any in-flight change to scope (additions, removals, or substantive alterations to deliverable acceptance criteria) goes through the change-control process documented in `09_change_log.md`. The PM has standing authority for cumulative budget impact below $20,000 per the charter; everything else escalates to the Sponsor + CTO.

## 8. Constraints

The scope is constrained by:

- The $2,400,000 budget envelope (`07_budget.md`).
- The 18-FTE staffing commitment (`08_raci_matrix.md`).
- The 2026-12-18 end date (`04_schedule_milestones.md`).
- The AcmeCloud capacity reservation under SLA-2026-04-A3 (`17_vendor_acme_contract_summary.md`).

A scope change that strains any one of these requires Sponsor approval.

---

**References:** `01_project_charter.md`, `03_work_breakdown_structure.md`, `04_schedule_milestones.md`, `07_budget.md`, `08_raci_matrix.md`, `09_change_log.md`, `17_vendor_acme_contract_summary.md`.
