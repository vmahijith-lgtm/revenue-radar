{{ config(materialized='view') }}
-- Calculates First Touch, Last Touch, U-Shaped, and Time-Decay
WITH sessions AS (
    SELECT 
        channel,
        user_id,
        conversion_value,
        timestamp,
        -- Window Functions to find position
        ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY timestamp ASC)  AS seq_asc,
        ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY timestamp DESC) AS seq_desc,
        COUNT(*)     OVER(PARTITION BY user_id)                         AS total_touches
    FROM {{ source('main', 'raw_clicks') }}
    WHERE channel IS NOT NULL -- Simple cleaning
),
sessions_with_weights AS (
    SELECT 
        *,
        POW(0.5, seq_desc - 1) AS time_weight,
        SUM(POW(0.5, seq_desc - 1)) OVER (PARTITION BY user_id) AS total_weight_per_user
    FROM sessions
    WHERE conversion_value > 0
)
SELECT 
    channel,
    -- First Touch Logic
    SUM(
        CASE 
            WHEN seq_asc = 1 THEN conversion_value 
            ELSE 0 
        END
    ) AS val_first_touch,
    -- Last Touch Logic
    SUM(
        CASE 
            WHEN seq_desc = 1 THEN conversion_value 
            ELSE 0 
        END
    ) AS val_last_touch,
    -- U-Shaped Logic (40/40/20)
    SUM(
        conversion_value * 
        CASE 
            WHEN total_touches = 1 THEN 1.0
            WHEN total_touches = 2 THEN 0.5
            WHEN seq_asc = 1 THEN {{ var('u_shape')['first'] }}
            WHEN seq_desc = 1 THEN {{ var('u_shape')['last'] }}
            ELSE CAST({{ var('u_shape')['middle'] }} AS DOUBLE) / (total_touches - 2)
        END
    ) AS val_u_shaped,
    -- Time-Decay Attribution
    SUM(
        conversion_value * time_weight / total_weight_per_user
    ) AS val_time_decay
FROM sessions_with_weights
GROUP BY channel
ORDER BY channel