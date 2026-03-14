{{ config(materialized='table') }}

-- Combined comparison model including Time Decay attribution

WITH heuristics AS (
    SELECT
        channel,
        val_first_touch,
        val_last_touch,
        val_u_shaped,
        val_time_decay
    FROM {{ ref('heuristic_attribution') }}
),

markov AS (
    SELECT
        channel,
        attributed_value AS val_markov
    FROM {{ ref('markov_attribution') }}
    WHERE status = 'success'
)

SELECT
    h.channel,
    h.val_first_touch,
    h.val_last_touch,
    h.val_u_shaped,
    h.val_time_decay,
    COALESCE(m.val_markov, 0) AS val_markov
FROM heuristics h
LEFT JOIN markov m USING (channel)
ORDER BY h.channel
