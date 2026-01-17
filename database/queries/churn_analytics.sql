-- ============================================================================
-- Customer Churn Prediction System - SQL Analytics Module
-- ============================================================================
-- Purpose: Demonstrate SQL proficiency for job portfolio
-- Skills: Aggregations, JOINs, CASE statements, Window Functions, CTEs
-- Database: PostgreSQL 16 (port 5433)
-- ============================================================================

-- ============================================================================
-- SECTION 1: BASIC ANALYTICS (Queries 1-4)
-- Skills: COUNT, GROUP BY, Simple Aggregations, Filtering
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Query 1: Overall Churn Summary
-- Purpose: High-level churn statistics for executive reporting
-- Skill: Basic aggregation with CASE for conditional counting
-- ----------------------------------------------------------------------------
SELECT 
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) AS churned_customers,
    SUM(CASE WHEN churn = FALSE THEN 1 ELSE 0 END) AS retained_customers,
    ROUND(100.0 * SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(AVG(total_charges), 2) AS avg_total_charges
FROM customers;


-- ----------------------------------------------------------------------------
-- Query 2: Churn Rate by Contract Type
-- Purpose: Identify which contract types have highest churn risk
-- Skill: JOIN, GROUP BY, ORDER BY
-- ----------------------------------------------------------------------------
SELECT 
    ct.contract_name,
    COUNT(c.customer_id) AS total_customers,
    SUM(CASE WHEN c.churn = TRUE THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN c.churn = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers c
JOIN customer_contracts cc ON c.customer_id = cc.customer_id
JOIN contract_types ct ON cc.contract_type_id = ct.contract_type_id
GROUP BY ct.contract_name
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- Query 3: Churn Analysis by Tenure Buckets
-- Purpose: Understand churn patterns across customer lifetime
-- Skill: CASE for bucketing, Aggregation
-- ----------------------------------------------------------------------------
SELECT 
    CASE 
        WHEN tenure_months <= 6 THEN '0-6 months'
        WHEN tenure_months <= 12 THEN '7-12 months'
        WHEN tenure_months <= 24 THEN '13-24 months'
        WHEN tenure_months <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_bucket,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM customers
GROUP BY 
    CASE 
        WHEN tenure_months <= 6 THEN '0-6 months'
        WHEN tenure_months <= 12 THEN '7-12 months'
        WHEN tenure_months <= 24 THEN '13-24 months'
        WHEN tenure_months <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END
ORDER BY 
    MIN(tenure_months);


-- ----------------------------------------------------------------------------
-- Query 4: Churn by Payment Method
-- Purpose: Identify payment methods associated with higher churn
-- Skill: JOIN, Aggregation, Filtering
-- ----------------------------------------------------------------------------
SELECT 
    cc.payment_method,
    COUNT(c.customer_id) AS total_customers,
    SUM(CASE WHEN c.churn = TRUE THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN c.churn = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(c.monthly_charges), 2) AS avg_monthly_charges,
    ROUND(SUM(c.total_charges), 2) AS total_revenue
FROM customers c
JOIN customer_contracts cc ON c.customer_id = cc.customer_id
GROUP BY cc.payment_method
ORDER BY churn_rate_pct DESC;


-- ============================================================================
-- SECTION 2: INTERMEDIATE ANALYTICS (Queries 5-8)
-- Skills: Multiple JOINs, Subqueries, Complex CASE logic
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Query 5: Revenue Impact Analysis
-- Purpose: Quantify monthly revenue lost to churn
-- Skill: Aggregation with business metrics calculation
-- ----------------------------------------------------------------------------
SELECT 
    'Current Monthly Revenue' AS metric,
    ROUND(SUM(CASE WHEN churn = FALSE THEN monthly_charges ELSE 0 END), 2) AS value
FROM customers
UNION ALL
SELECT 
    'Lost Monthly Revenue (Churned)',
    ROUND(SUM(CASE WHEN churn = TRUE THEN monthly_charges ELSE 0 END), 2)
FROM customers
UNION ALL
SELECT 
    'Potential Monthly Revenue (If No Churn)',
    ROUND(SUM(monthly_charges), 2)
FROM customers
UNION ALL
SELECT 
    'Revenue Loss Percentage',
    ROUND(100.0 * SUM(CASE WHEN churn = TRUE THEN monthly_charges ELSE 0 END) / SUM(monthly_charges), 2)
FROM customers;


-- ----------------------------------------------------------------------------
-- Query 6: Service Adoption vs Churn Analysis
-- Purpose: Understand which services correlate with churn
-- Skill: Multiple JOINs, Conditional aggregation
-- ----------------------------------------------------------------------------
SELECT 
    st.service_name,
    COUNT(DISTINCT CASE WHEN cs.is_subscribed = TRUE THEN c.customer_id END) AS customers_with_service,
    COUNT(DISTINCT CASE WHEN cs.is_subscribed = TRUE AND c.churn = TRUE THEN c.customer_id END) AS churned_with_service,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN cs.is_subscribed = TRUE AND c.churn = TRUE THEN c.customer_id END) / 
          NULLIF(COUNT(DISTINCT CASE WHEN cs.is_subscribed = TRUE THEN c.customer_id END), 0), 2) AS churn_rate_with_service,
    COUNT(DISTINCT CASE WHEN cs.is_subscribed = FALSE OR cs.is_subscribed IS NULL THEN c.customer_id END) AS customers_without_service,
    COUNT(DISTINCT CASE WHEN (cs.is_subscribed = FALSE OR cs.is_subscribed IS NULL) AND c.churn = TRUE THEN c.customer_id END) AS churned_without_service,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN (cs.is_subscribed = FALSE OR cs.is_subscribed IS NULL) AND c.churn = TRUE THEN c.customer_id END) / 
          NULLIF(COUNT(DISTINCT CASE WHEN cs.is_subscribed = FALSE OR cs.is_subscribed IS NULL THEN c.customer_id END), 0), 2) AS churn_rate_without_service
FROM customers c
CROSS JOIN service_types st
LEFT JOIN customer_services cs ON c.customer_id = cs.customer_id AND st.service_type_id = cs.service_type_id
GROUP BY st.service_name
ORDER BY st.service_name;


-- ----------------------------------------------------------------------------
-- Query 7: Customer Risk Segmentation (Multi-factor)
-- Purpose: Segment customers into risk categories based on multiple factors
-- Skill: Complex CASE with multiple conditions, Subquery
-- ----------------------------------------------------------------------------
SELECT 
    risk_segment,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) AS actually_churned,
    ROUND(100.0 * SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS actual_churn_rate,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(SUM(monthly_charges), 2) AS total_monthly_revenue
FROM (
    SELECT 
        c.*,
        CASE 
            WHEN c.tenure_months <= 6 AND ct.contract_name = 'Month-to-month' AND c.monthly_charges > 70 THEN 'Critical Risk'
            WHEN c.tenure_months <= 12 AND ct.contract_name = 'Month-to-month' THEN 'High Risk'
            WHEN ct.contract_name = 'Month-to-month' OR c.tenure_months <= 12 THEN 'Medium Risk'
            ELSE 'Low Risk'
        END AS risk_segment
    FROM customers c
    JOIN customer_contracts cc ON c.customer_id = cc.customer_id
    JOIN contract_types ct ON cc.contract_type_id = ct.contract_type_id
) segmented
GROUP BY risk_segment
ORDER BY 
    CASE risk_segment 
        WHEN 'Critical Risk' THEN 1 
        WHEN 'High Risk' THEN 2 
        WHEN 'Medium Risk' THEN 3 
        ELSE 4 
    END;


-- ----------------------------------------------------------------------------
-- Query 8: Demographic Churn Analysis
-- Purpose: Analyze churn patterns by demographic combinations
-- Skill: Multiple column grouping, Aggregation
-- ----------------------------------------------------------------------------
SELECT 
    gender,
    senior_citizen,
    CASE 
        WHEN partner = TRUE AND dependents = TRUE THEN 'Family (Partner + Dependents)'
        WHEN partner = TRUE THEN 'With Partner Only'
        WHEN dependents = TRUE THEN 'With Dependents Only'
        ELSE 'Single'
    END AS family_status,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY 
    gender,
    senior_citizen,
    CASE 
        WHEN partner = TRUE AND dependents = TRUE THEN 'Family (Partner + Dependents)'
        WHEN partner = TRUE THEN 'With Partner Only'
        WHEN dependents = TRUE THEN 'With Dependents Only'
        ELSE 'Single'
    END
ORDER BY churn_rate_pct DESC;


-- ============================================================================
-- SECTION 3: ADVANCED ANALYTICS (Queries 9-12)
-- Skills: Window Functions, CTEs, Complex Business Logic
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Query 9: Customer Lifetime Value (CLV) Analysis with Ranking
-- Purpose: Identify high-value customers at risk
-- Skill: Window Functions (RANK, NTILE), CTE
-- ----------------------------------------------------------------------------
WITH customer_clv AS (
    SELECT 
        c.customer_id,
        c.tenure_months,
        c.monthly_charges,
        c.total_charges,
        c.churn,
        -- Simple CLV estimation: monthly charges * expected remaining tenure
        CASE 
            WHEN c.churn = TRUE THEN c.total_charges
            ELSE c.total_charges + (c.monthly_charges * 12) -- Assume 12 more months
        END AS estimated_clv,
        ct.contract_name
    FROM customers c
    JOIN customer_contracts cc ON c.customer_id = cc.customer_id
    JOIN contract_types ct ON cc.contract_type_id = ct.contract_type_id
),
clv_ranked AS (
    SELECT 
        *,
        NTILE(4) OVER (ORDER BY estimated_clv DESC) AS clv_quartile,
        RANK() OVER (ORDER BY estimated_clv DESC) AS clv_rank
    FROM customer_clv
)
SELECT 
    clv_quartile,
    CASE clv_quartile 
        WHEN 1 THEN 'Top 25% (Premium)'
        WHEN 2 THEN '25-50% (High Value)'
        WHEN 3 THEN '50-75% (Medium Value)'
        ELSE 'Bottom 25% (Low Value)'
    END AS value_segment,
    COUNT(*) AS customers,
    SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(estimated_clv), 2) AS avg_clv,
    ROUND(SUM(estimated_clv), 2) AS total_clv_at_risk
FROM clv_ranked
GROUP BY clv_quartile
ORDER BY clv_quartile;


-- ----------------------------------------------------------------------------
-- Query 10: RFM-Style Customer Segmentation
-- Purpose: Segment customers by Recency (tenure), Frequency (services), Monetary (charges)
-- Skill: Multiple CTEs, Window Functions (NTILE)
-- ----------------------------------------------------------------------------
WITH service_count AS (
    SELECT 
        customer_id,
        COUNT(CASE WHEN is_subscribed = TRUE THEN 1 END) AS num_services
    FROM customer_services
    GROUP BY customer_id
),
rfm_scores AS (
    SELECT 
        c.customer_id,
        c.tenure_months,
        COALESCE(sc.num_services, 0) AS num_services,
        c.monthly_charges,
        c.total_charges,
        c.churn,
        -- R: Recency score (higher tenure = more recent/engaged customer = higher score)
        NTILE(5) OVER (ORDER BY c.tenure_months ASC) AS r_score,
        -- F: Frequency score (more services = higher engagement)
        NTILE(5) OVER (ORDER BY COALESCE(sc.num_services, 0) ASC) AS f_score,
        -- M: Monetary score (higher charges = more valuable)
        NTILE(5) OVER (ORDER BY c.monthly_charges ASC) AS m_score
    FROM customers c
    LEFT JOIN service_count sc ON c.customer_id = sc.customer_id
),
rfm_segments AS (
    SELECT 
        *,
        r_score + f_score + m_score AS rfm_total,
        CASE 
            WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
            WHEN r_score >= 4 AND f_score >= 3 AND m_score >= 3 THEN 'Loyal Customers'
            WHEN r_score >= 3 AND m_score >= 4 THEN 'Big Spenders'
            WHEN r_score <= 2 AND f_score <= 2 THEN 'At Risk'
            WHEN r_score <= 2 AND m_score >= 3 THEN 'High Value At Risk'
            ELSE 'Regular'
        END AS rfm_segment
    FROM rfm_scores
)
SELECT 
    rfm_segment,
    COUNT(*) AS customers,
    SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(AVG(num_services), 1) AS avg_services,
    ROUND(AVG(tenure_months), 1) AS avg_tenure
FROM rfm_segments
GROUP BY rfm_segment
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- Query 11: Churn Probability Bands with Running Totals
-- Purpose: Create probability bands for retention targeting
-- Skill: Window Functions (SUM OVER, LAG), CTE
-- ----------------------------------------------------------------------------
WITH monthly_charge_bands AS (
    SELECT 
        CASE 
            WHEN monthly_charges < 30 THEN '$0-30'
            WHEN monthly_charges < 50 THEN '$30-50'
            WHEN monthly_charges < 70 THEN '$50-70'
            WHEN monthly_charges < 90 THEN '$70-90'
            ELSE '$90+'
        END AS charge_band,
        CASE 
            WHEN monthly_charges < 30 THEN 1
            WHEN monthly_charges < 50 THEN 2
            WHEN monthly_charges < 70 THEN 3
            WHEN monthly_charges < 90 THEN 4
            ELSE 5
        END AS band_order,
        COUNT(*) AS customers,
        SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) AS churned,
        SUM(monthly_charges) AS total_revenue
    FROM customers
    GROUP BY 1, 2
)
SELECT 
    charge_band,
    customers,
    churned,
    ROUND(100.0 * churned / customers, 2) AS churn_rate_pct,
    ROUND(total_revenue, 2) AS monthly_revenue,
    -- Running totals
    SUM(customers) OVER (ORDER BY band_order) AS cumulative_customers,
    SUM(churned) OVER (ORDER BY band_order) AS cumulative_churned,
    ROUND(100.0 * SUM(churned) OVER (ORDER BY band_order) / 
          SUM(customers) OVER (ORDER BY band_order), 2) AS cumulative_churn_rate,
    -- Compare to previous band
    ROUND(100.0 * churned / customers - 
          LAG(100.0 * churned / customers) OVER (ORDER BY band_order), 2) AS churn_rate_change
FROM monthly_charge_bands
ORDER BY band_order;


-- ----------------------------------------------------------------------------
-- Query 12: Service Bundle Analysis
-- Purpose: Analyze which service combinations have lowest churn
-- Skill: STRING_AGG, Complex JOINs, Grouping Sets equivalent
-- ----------------------------------------------------------------------------
WITH customer_bundles AS (
    SELECT 
        c.customer_id,
        c.churn,
        c.monthly_charges,
        STRING_AGG(
            CASE WHEN cs.is_subscribed = TRUE THEN st.service_name END, 
            ' + ' ORDER BY st.service_name
        ) AS service_bundle,
        COUNT(CASE WHEN cs.is_subscribed = TRUE THEN 1 END) AS service_count
    FROM customers c
    LEFT JOIN customer_services cs ON c.customer_id = cs.customer_id
    LEFT JOIN service_types st ON cs.service_type_id = st.service_type_id
    GROUP BY c.customer_id, c.churn, c.monthly_charges
)
SELECT 
    service_count,
    COUNT(*) AS customers,
    SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM customer_bundles
GROUP BY service_count
ORDER BY service_count;


-- ============================================================================
-- SECTION 4: BUSINESS INTELLIGENCE QUERIES (Queries 13-15)
-- Skills: Complex CTEs, Pivot-style queries, Actionable Insights
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Query 13: Monthly Cohort Retention Analysis
-- Purpose: Track retention by customer signup cohort (tenure-based proxy)
-- Skill: Multiple CTEs, Complex business logic, Pivot-style output
-- ----------------------------------------------------------------------------
WITH tenure_cohorts AS (
    SELECT 
        customer_id,
        churn,
        monthly_charges,
        -- Create cohort based on when they would have joined (72 - tenure = months ago)
        CASE 
            WHEN tenure_months >= 60 THEN 'Early Adopters (5+ years)'
            WHEN tenure_months >= 48 THEN 'Established (4-5 years)'
            WHEN tenure_months >= 36 THEN 'Mature (3-4 years)'
            WHEN tenure_months >= 24 THEN 'Growing (2-3 years)'
            WHEN tenure_months >= 12 THEN 'Developing (1-2 years)'
            ELSE 'New (< 1 year)'
        END AS cohort,
        CASE 
            WHEN tenure_months >= 60 THEN 1
            WHEN tenure_months >= 48 THEN 2
            WHEN tenure_months >= 36 THEN 3
            WHEN tenure_months >= 24 THEN 4
            WHEN tenure_months >= 12 THEN 5
            ELSE 6
        END AS cohort_order
    FROM customers
)
SELECT 
    cohort,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = FALSE THEN 1 ELSE 0 END) AS retained,
    SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn = FALSE THEN 1 ELSE 0 END) / COUNT(*), 2) AS retention_rate_pct,
    ROUND(SUM(CASE WHEN churn = FALSE THEN monthly_charges ELSE 0 END), 2) AS active_revenue,
    ROUND(SUM(CASE WHEN churn = TRUE THEN monthly_charges ELSE 0 END), 2) AS lost_revenue
FROM tenure_cohorts
GROUP BY cohort, cohort_order
ORDER BY cohort_order;


-- ----------------------------------------------------------------------------
-- Query 14: Contract Upgrade/Downgrade Opportunity Analysis
-- Purpose: Identify customers who should be targeted for contract changes
-- Skill: Complex CTE, Business rules, Actionable recommendations
-- ----------------------------------------------------------------------------
WITH customer_analysis AS (
    SELECT 
        c.customer_id,
        c.tenure_months,
        c.monthly_charges,
        c.total_charges,
        c.churn,
        ct.contract_name,
        cc.payment_method,
        -- Count services
        (SELECT COUNT(*) FROM customer_services cs 
         WHERE cs.customer_id = c.customer_id AND cs.is_subscribed = TRUE) AS num_services
    FROM customers c
    JOIN customer_contracts cc ON c.customer_id = cc.customer_id
    JOIN contract_types ct ON cc.contract_type_id = ct.contract_type_id
),
recommendations AS (
    SELECT 
        *,
        CASE 
            -- High value month-to-month customers: Offer annual contract
            WHEN contract_name = 'Month-to-month' 
                 AND tenure_months >= 12 
                 AND monthly_charges >= 70 
                 AND churn = FALSE 
            THEN 'Upgrade to Annual (High Value Loyal)'
            
            -- New high-paying month-to-month: Immediate retention risk
            WHEN contract_name = 'Month-to-month' 
                 AND tenure_months <= 6 
                 AND monthly_charges >= 80 
                 AND churn = FALSE 
            THEN 'Urgent: Offer Contract Incentive'
            
            -- Long-term annual customers: Offer 2-year for discount
            WHEN contract_name = 'One year' 
                 AND tenure_months >= 24 
                 AND churn = FALSE 
            THEN 'Upgrade to 2-Year Contract'
            
            -- Electronic check users with high churn risk
            WHEN payment_method = 'Electronic check' 
                 AND contract_name = 'Month-to-month' 
                 AND churn = FALSE 
            THEN 'Convert Payment Method + Offer Contract'
            
            ELSE 'No Action Needed'
        END AS recommendation
    FROM customer_analysis
)
SELECT 
    recommendation,
    COUNT(*) AS customer_count,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(SUM(monthly_charges), 2) AS total_monthly_value,
    ROUND(AVG(tenure_months), 1) AS avg_tenure,
    ROUND(AVG(num_services), 1) AS avg_services
FROM recommendations
WHERE churn = FALSE  -- Only active customers
GROUP BY recommendation
ORDER BY total_monthly_value DESC;


-- ----------------------------------------------------------------------------
-- Query 15: Executive Dashboard Query (Comprehensive Summary)
-- Purpose: Single query for dashboard KPIs
-- Skill: UNION ALL, Subqueries, Complete business summary
-- ----------------------------------------------------------------------------
WITH base_metrics AS (
    SELECT 
        COUNT(*) AS total_customers,
        SUM(CASE WHEN churn = TRUE THEN 1 ELSE 0 END) AS total_churned,
        SUM(CASE WHEN churn = FALSE THEN 1 ELSE 0 END) AS active_customers,
        SUM(monthly_charges) AS total_mrr,
        SUM(CASE WHEN churn = FALSE THEN monthly_charges ELSE 0 END) AS active_mrr,
        SUM(CASE WHEN churn = TRUE THEN monthly_charges ELSE 0 END) AS lost_mrr,
        SUM(total_charges) AS lifetime_revenue,
        AVG(tenure_months) AS avg_tenure
    FROM customers
),
contract_risk AS (
    SELECT 
        COUNT(*) AS mtm_customers,
        SUM(CASE WHEN c.churn = TRUE THEN 1 ELSE 0 END) AS mtm_churned
    FROM customers c
    JOIN customer_contracts cc ON c.customer_id = cc.customer_id
    JOIN contract_types ct ON cc.contract_type_id = ct.contract_type_id
    WHERE ct.contract_name = 'Month-to-month'
),
high_value_at_risk AS (
    SELECT 
        COUNT(*) AS hv_at_risk_count,
        SUM(monthly_charges) AS hv_at_risk_revenue
    FROM customers c
    JOIN customer_contracts cc ON c.customer_id = cc.customer_id
    JOIN contract_types ct ON cc.contract_type_id = ct.contract_type_id
    WHERE c.churn = FALSE 
      AND c.monthly_charges >= 70
      AND ct.contract_name = 'Month-to-month'
      AND c.tenure_months <= 12
)
SELECT 'Total Customers' AS metric, total_customers::TEXT AS value FROM base_metrics
UNION ALL SELECT 'Active Customers', active_customers::TEXT FROM base_metrics
UNION ALL SELECT 'Churned Customers', total_churned::TEXT FROM base_metrics
UNION ALL SELECT 'Churn Rate (%)', ROUND(100.0 * total_churned / total_customers, 2)::TEXT FROM base_metrics
UNION ALL SELECT 'Monthly Recurring Revenue ($)', ROUND(active_mrr, 2)::TEXT FROM base_metrics
UNION ALL SELECT 'Lost Monthly Revenue ($)', ROUND(lost_mrr, 2)::TEXT FROM base_metrics
UNION ALL SELECT 'Lifetime Revenue ($)', ROUND(lifetime_revenue, 2)::TEXT FROM base_metrics
UNION ALL SELECT 'Avg Customer Tenure (months)', ROUND(avg_tenure, 1)::TEXT FROM base_metrics
UNION ALL SELECT 'Month-to-Month Customers', mtm_customers::TEXT FROM contract_risk
UNION ALL SELECT 'Month-to-Month Churn Rate (%)', ROUND(100.0 * mtm_churned / mtm_customers, 2)::TEXT FROM contract_risk
UNION ALL SELECT 'High-Value At-Risk Customers', hv_at_risk_count::TEXT FROM high_value_at_risk
UNION ALL SELECT 'High-Value At-Risk Revenue ($)', ROUND(hv_at_risk_revenue, 2)::TEXT FROM high_value_at_risk;


