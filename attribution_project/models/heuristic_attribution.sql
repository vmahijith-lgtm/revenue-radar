{{ config(materialized='view') }}
-- Multi-Touch Heuristic Attribution
-- Correctly attributes revenue across ALL touches of converting users.
--
-- Bug fix: the original WHERE conversion_value > 0 on touch rows
-- discarded every non-converting touch, making First Touch / U-Shaped
-- / Time-Decay all behave like Last Touch.
--
-- Correct approach:
--   1. Find users who converted and sum their total revenue.
--   2. Join ALL their touches (not just the converting touch).
--   3. Compute positional windows over the full journey.
--   4. Apply attribution weights to every touch, not just the last one.

WITH converting_users AS (
    -- Step 1: users who actually converted, with their total revenue
    SELECT
        user_id,
        SUM(conversion_value) AS total_conversion_value
    FROM {{ source('main', 'raw_clicks') }}
    WHERE conversion = 1 AND conversion_value > 0
    GROUP BY user_id
),

sessions AS (
    -- Step 2: ALL touches for converting users, carrying user-level revenue
    SELECT
        r.channel,
        r.user_id,
        cu.total_conversion_value                                          AS conversion_value,
        r.timestamp,
        ROW_NUMBER() OVER(PARTITION BY r.user_id ORDER BY r.timestamp ASC)  AS seq_asc,
        ROW_NUMBER() OVER(PARTITION BY r.user_id ORDER BY r.timestamp DESC) AS seq_desc,
        COUNT(*)     OVER(PARTITION BY r.user_id)                           AS total_touches
    FROM {{ source('main', 'raw_clicks') }} r
    INNER JOIN converting_users cu USING (user_id)
    WHERE r.channel IS NOT NULL
),

sessions_with_weights AS (
    -- Step 3: add time-decay weights (exponential, most recent = highest)
    SELECT
        *,
        POW(0.5, seq_desc - 1)                                            AS time_weight,
        SUM(POW(0.5, seq_desc - 1)) OVER (PARTITION BY user_id)           AS total_weight_per_user
    FROM sessions
)

-- Step 4: aggregate attribution per channel
SELECT
    channel,

    -- First Touch: 100% to the channel that started the journey
    SUM(
        CASE WHEN seq_asc = 1 THEN conversion_value ELSE 0 END
    ) AS val_first_touch,

    -- Last Touch: 100% to the channel immediately before conversion
    SUM(
        CASE WHEN seq_desc = 1 THEN conversion_value ELSE 0 END
    ) AS val_last_touch,

    -- U-Shaped (40/40/20): first and last touches share 80%, middle shares 20%
    SUM(
        conversion_value *
        CASE
            WHEN total_touches = 1 THEN 1.0
            WHEN total_touches = 2 THEN 0.5
            WHEN seq_asc  = 1      THEN {{ var('u_shape')['first'] }}
            WHEN seq_desc = 1      THEN {{ var('u_shape')['last'] }}
            ELSE CAST({{ var('u_shape')['middle'] }} AS DOUBLE) / (total_touches - 2)
        END
    ) AS val_u_shaped,

    -- Time Decay: exponential decay, touches closer to conversion get more credit
    -- weight_i = 0.5^(seq_desc-1), normalised across all touches for that user
    SUM(
        conversion_value * time_weight / total_weight_per_user
    ) AS val_time_decay

FROM sessions_with_weights
GROUP BY channel
ORDER BY channel