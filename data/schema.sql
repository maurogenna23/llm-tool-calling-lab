-- arnie-lab schema. Applied on every start; safe to re-run.

CREATE TABLE IF NOT EXISTS dishes (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL UNIQUE,
    category    TEXT    NOT NULL,            -- entrada | principal | postre | bebida
    price_cents INTEGER NOT NULL CHECK (price_cents > 0),
    description TEXT    NOT NULL,
    tags        TEXT    NOT NULL DEFAULT '', -- comma separated: vegano,vegetariano,sin-gluten,picante
    available   INTEGER NOT NULL DEFAULT 1 CHECK (available IN (0, 1))
);

CREATE TABLE IF NOT EXISTS dining_tables (
    id    INTEGER PRIMARY KEY,
    seats INTEGER NOT NULL CHECK (seats > 0),
    zone  TEXT    NOT NULL                   -- salon | terraza | barra
);

CREATE TABLE IF NOT EXISTS reservations (
    id            INTEGER PRIMARY KEY,
    code          TEXT    NOT NULL UNIQUE,   -- R-7K2M
    customer_name TEXT    NOT NULL,
    party_size    INTEGER NOT NULL CHECK (party_size > 0),
    starts_at     TEXT    NOT NULL,          -- ISO 8601, local time, minute precision
    table_id      INTEGER NOT NULL REFERENCES dining_tables(id),
    status        TEXT    NOT NULL DEFAULT 'CONFIRMED'
                  CHECK (status IN ('CONFIRMED', 'CANCELLED')),
    created_at    TEXT    NOT NULL
);

-- The availability query filters by status and scans a time window.
CREATE INDEX IF NOT EXISTS idx_reservations_slot ON reservations (status, starts_at);

CREATE TABLE IF NOT EXISTS opening_hours (
    weekday   INTEGER PRIMARY KEY CHECK (weekday BETWEEN 0 AND 6),  -- 0 = Monday
    opens_at  TEXT,                          -- NULL on both columns means closed
    closes_at TEXT
);
