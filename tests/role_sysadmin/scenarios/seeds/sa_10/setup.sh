#!/usr/bin/env bash
# sa_10 seed — runs ON the target as root BEFORE the agent (Target._apply_seed).
#
# Plants several large incident POST-MORTEMS (natural-language prose, not
# structured logs) that the agent must READ IN FULL to synthesise cross-incident
# themes. Prose is used deliberately: a structured log lets the agent shortcut
# with grep/awk/wc (small tool output, no context growth); a "what are the common
# contributing factors across these incidents" question can only be answered by
# reading each narrative into the model's context — which is exactly what makes
# this the context-heavy workload for the context_max_tokens compression sweep
# (#2360 / #2365). Each file is sized just under the 30 000-char per-ToolMessage
# history cap, so a single-file read lands a large, compression-eligible block.
set -e

mkdir -p /var/incidents

awk 'BEGIN {
  # Per-incident distinct root cause (the agent must read each to know it).
  split("payment-api database-cluster auth-service search-index notification-worker cdn-edge billing-batch inventory-sync", svc, " ");
  split("a saturated connection pool|an expired TLS certificate|a silent config drift|an unbounded retry storm|a full disk on the primary|a leaked file descriptor|a clock skew across nodes|a poisoned cache entry", cause, "|");
  # Narrative fragments that IMPLY the recurring contributing factors WITHOUT
  # naming them as grep-able keywords — the model has to infer the theme.
  split("no dashboard tracked the relevant metric so the trend went unseen for hours|the alert for this condition had been muted weeks earlier during unrelated noise|the change that introduced the fault was applied by hand and never reviewed by a second engineer|the runbook referenced a procedure that no longer matched the current topology|there was no rehearsed way to roll the change back once it was live|the component was a single instance with no standby to fail over to|capacity had not been re-evaluated since the last traffic increase|the deploy skipped the staging environment because of time pressure", frag, "|");
  for (n = 1; n <= 8; n++) {
    f = "/var/incidents/incident_" n ".md";
    printf "# Post-mortem: %s outage (INC-20260%d)\n\n", svc[n], n > f;
    printf "## Summary\nOn the day of the incident the %s degraded and then failed for a sustained period. The proximate trigger was %s. What follows is the full narrative reconstructed from the on-call notes and chat history; the review team asks that you read it in its entirety, because the factors that let a small trigger become a long outage are spread across the timeline rather than stated in one place.\n\n", svc[n], cause[n] >> f;
    printf "## Timeline\n" >> f;
    # ~130 prose timeline lines, each weaving in a contributing-factor fragment
    # with per-line variation so the document is genuinely non-repetitive.
    for (i = 1; i <= 130; i++) {
      hh = (i * 3 + n) % 24; mm = (i * 7) % 60;
      fr = frag[((i + n) % 8) + 1];
      printf "At %02d:%02d, %d minutes into the event, the responding engineer worked the %s path and later recorded that %s; this cost roughly %d minutes before the next meaningful step, and the note-taker flagged it as a factor worth revisiting in review.\n", hh, mm, i, svc[n], fr, (i % 17) + 3 >> f;
    }
    printf "\n## Root-cause analysis\nThe technical root cause was %s. But the review is less interested in the trigger than in why recovery took as long as it did. Read the timeline above closely: the same handful of organisational gaps recur under different guises, and naming them precisely (not the symptoms) is the point of this exercise.\n\n", cause[n] >> f;
    printf "## Remediation\nShort-term mitigations were applied by the end of the day. The durable fixes are owned by the service team and tracked separately; they are intentionally NOT enumerated here so that a reader who has not read the narrative cannot reconstruct the contributing factors from this section alone.\n", n >> f;
  }
}'

chmod 644 /var/incidents/*.md
echo "sa_10 seed: $(ls /var/incidents/*.md | wc -l) post-mortems, total $(du -sh /var/incidents | cut -f1), largest $(wc -c < /var/incidents/incident_1.md) chars" >&2
exit 0
