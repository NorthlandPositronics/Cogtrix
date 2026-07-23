# PMBOK Reference — Knowledge Areas

**Document ID:** NIMB-REF-PMBOK-KA
**Version:** 1.0
**Source:** condensed extract of the PMBOK Guide 6th-edition Knowledge Areas, retained as reference vocabulary for the AI PM Agent. (PMBOK 7th edition has largely re-organised these into "performance domains"; the 6th-edition knowledge-area framing is still in common professional use, so the agent should recognise both vocabularies.)

This is reference material, NOT a Nimbus-specific document.

---

## Ten Knowledge Areas

### 1. Integration Management

**Concern:** unifying the components of project management into a coherent whole.

**Key activities:** develop charter, develop PM plan, direct and manage work, manage project knowledge, monitor and control work, perform integrated change control, close project or phase.

**Nimbus mapping:** the PM role itself; `01_project_charter.md`, this entire document set.

### 2. Scope Management

**Concern:** ensuring the project includes all the work required, and only the work required.

**Key activities:** plan scope management, collect requirements, define scope, create WBS, validate scope, control scope.

**Nimbus mapping:** `02_scope_statement.md`, `03_work_breakdown_structure.md`, the change-control process in `09_change_log.md`.

### 3. Schedule Management

**Concern:** managing timely completion of the project.

**Key activities:** plan schedule management, define activities, sequence activities, estimate durations, develop and control schedule.

**Nimbus mapping:** `04_schedule_milestones.md`.

### 4. Cost Management

**Concern:** planning, estimating, budgeting, financing, funding, managing, and controlling costs.

**Key activities:** plan cost management, estimate costs, determine budget, control costs.

**Nimbus mapping:** `07_budget.md`.

### 5. Quality Management

**Concern:** project and product quality.

**Key activities:** plan quality management, manage quality, control quality.

**Nimbus mapping:** acceptance criteria in `02_scope_statement.md`; rehearsal pass/fail gates in `04_schedule_milestones.md`.

### 6. Resource Management

**Concern:** identifying, acquiring, and managing the resources needed for successful project completion.

**Key activities:** plan resource management, estimate activity resources, acquire resources, develop and manage team, control resources.

**Nimbus mapping:** the 18-FTE allocation (charter), `08_raci_matrix.md`, the engineering reservation for D-09 in `16_meeting_notes_steering_q2.md`.

### 7. Communications Management

**Concern:** timely and appropriate planning, collection, creation, distribution, storage, retrieval, management, control, monitoring, and ultimate disposition of project information.

**Key activities:** plan communications, manage communications, monitor communications.

**Nimbus mapping:** `18_communication_plan.md`, the monthly status reports.

### 8. Risk Management

**Concern:** conducting risk-management planning, identification, analysis, response planning, and controlling risk on a project.

**Key activities:** plan risk management, identify risks, perform qualitative + quantitative risk analysis, plan + implement risk responses, monitor risks.

**Nimbus mapping:** `05_risk_register.md`, escalation thresholds in `01_project_charter.md` section 10.

### 9. Procurement Management

**Concern:** purchasing or acquiring products, services, or results from outside the project team.

**Key activities:** plan procurement management, conduct procurements, control procurements.

**Nimbus mapping:** `17_vendor_acme_contract_summary.md`.

### 10. Stakeholder Management

**Concern:** identifying people, groups, or organisations impacted by the project; analysing stakeholder expectations and their impact; developing appropriate strategies for engaging stakeholders.

**Key activities:** identify stakeholders, plan stakeholder engagement, manage stakeholder engagement, monitor stakeholder engagement.

**Nimbus mapping:** `06_stakeholder_register.md`, `18_communication_plan.md`.

---

## Cross-Reference: Knowledge Areas vs Process Groups

A complete Knowledge Area touches multiple Process Groups. The matrix below is the standard 6th-edition cross-reference, condensed:

| Knowledge Area | Initiating | Planning | Executing | Monitoring & Controlling | Closing |
|---|---|---|---|---|---|
| Integration | ✓ | ✓ | ✓ | ✓ | ✓ |
| Scope | | ✓ | | ✓ | |
| Schedule | | ✓ | | ✓ | |
| Cost | | ✓ | | ✓ | |
| Quality | | ✓ | ✓ | ✓ | |
| Resource | | ✓ | ✓ | ✓ | |
| Communications | | ✓ | ✓ | ✓ | |
| Risk | | ✓ | ✓ | ✓ | |
| Procurement | | ✓ | ✓ | ✓ | |
| Stakeholder | ✓ | ✓ | ✓ | ✓ | |

---

**References (general):** PMBOK Guide 6th Edition (knowledge-areas chapters), 7th Edition (performance-domains chapter for the modern framing).
**References (Nimbus):** as cited per knowledge area above.
