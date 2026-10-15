-- Audit trail for the HITL PR-review agent (SQLite version).
--
-- LangGraph's SqliteSaver already persists checkpoints (and lets us resume
-- or `time-travel` to any earlier state) in its own tables. Those tables are
-- great for *resuming* a paused graph but they're a blob-of-state per step,
-- not a queryable structured log. This table is the structured log.
--
-- One row per meaningful decision/event in a review session. Fields are
-- first-class columns so auditors can query directly:
--
--   SELECT AVG(confidence) FROM audit_events WHERE decision = 'approve';
--   SELECT * FROM audit_events WHERE risk_level = 'high' AND decision = 'auto';

CREATE TABLE IF NOT EXISTS audit_events (
    id                INTEGER  PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT     NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),

    -- Session context
    thread_id         TEXT     NOT NULL,
    pr_url            TEXT     NOT NULL,

    -- AuditEntry fields (common/schemas.py)
    agent_id          TEXT     NOT NULL,
    action            TEXT     NOT NULL,
    confidence        REAL     NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    risk_level        TEXT     NOT NULL CHECK (risk_level IN ('low', 'med', 'high')),
    reviewer_id       TEXT,
    decision          TEXT     NOT NULL,
    reason            TEXT,
    execution_time_ms INTEGER  NOT NULL CHECK (execution_time_ms >= 0),
    event_key         TEXT
);

CREATE INDEX IF NOT EXISTS audit_events_thread_idx   ON audit_events (thread_id, id);
CREATE INDEX IF NOT EXISTS audit_events_pr_idx       ON audit_events (pr_url, timestamp DESC);
CREATE INDEX IF NOT EXISTS audit_events_decision_idx ON audit_events (decision);
CREATE INDEX IF NOT EXISTS audit_events_risk_idx     ON audit_events (risk_level);
CREATE UNIQUE INDEX IF NOT EXISTS audit_events_dedupe_idx
    ON audit_events (thread_id, event_key) WHERE event_key IS NOT NULL;
