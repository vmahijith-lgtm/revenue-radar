{{ config(materialized='table') }}

-- ROI Attribution Model
-- Calculates ROI% for each channel and each attribution model.
-- ROI% = (Revenue - Spend) / Spend * 100

WITH rev AS (
    SELECT
        channel,
        val_first_touch,
        val_last_touch,
        val_u_shaped,
        val_time_decay,
        val_markov
    FROM {{ ref('final_attribution') }}
),

cost AS (
    SELECT
        channel,
        spend
    FROM {{ ref('channel_spend') }}
)

SELECT
    rev.channel,
    COALESCE(cost.spend, 0) AS spend,

    -- Attributed revenue by model
    rev.val_first_touch,
    rev.val_last_touch,
    rev.val_u_shaped,
    rev.val_time_decay,
    rev.val_markov,

    -- ROI% as ((Revenue - Spend) / Spend) * 100
    -- NULLIF(cost.spend, 0) avoids division by zero (e.g. Direct).
    ROUND(
        (rev.val_first_touch - cost.spend) / NULLIF(cost.spend, 0) * 100,
        2
    ) AS roi_first_touch,

    ROUND(
        (rev.val_last_touch - cost.spend) / NULLIF(cost.spend, 0) * 100,
        2
    ) AS roi_last_touch,

    ROUND(
        (rev.val_u_shaped - cost.spend) / NULLIF(cost.spend, 0) * 100,
        2
    ) AS roi_u_shaped,

    ROUND(
        (rev.val_time_decay - cost.spend) / NULLIF(cost.spend, 0) * 100,
        2
    ) AS roi_time_decay,

    ROUND(
        (rev.val_markov - cost.spend) / NULLIF(cost.spend, 0) * 100,
        2
    ) AS roi_markov

FROM rev
LEFT JOIN cost USING (channel)
ORDER BY rev.channel
