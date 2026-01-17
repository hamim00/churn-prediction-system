-- ===========================================
-- Customer Churn Prediction System
-- Database Schema (PostgreSQL)
-- ===========================================

-- Drop existing tables if recreating
DROP TABLE IF EXISTS customer_services CASCADE;
DROP TABLE IF EXISTS customer_contracts CASCADE;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS service_types CASCADE;
DROP TABLE IF EXISTS contract_types CASCADE;

-- ===========================================
-- LOOKUP TABLES
-- ===========================================

CREATE TABLE contract_types (
    contract_type_id SERIAL PRIMARY KEY,
    contract_name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO contract_types (contract_name, description) VALUES
    ('Month-to-month', 'No commitment, can cancel anytime'),
    ('One year', 'One year commitment with potential discount'),
    ('Two year', 'Two year commitment with maximum discount');

CREATE TABLE service_types (
    service_type_id SERIAL PRIMARY KEY,
    service_name VARCHAR(50) NOT NULL UNIQUE,
    service_category VARCHAR(30) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO service_types (service_name, service_category, description) VALUES
    ('PhoneService', 'phone', 'Basic phone service'),
    ('MultipleLines', 'phone', 'Multiple phone lines'),
    ('InternetService_DSL', 'internet', 'DSL Internet connection'),
    ('InternetService_Fiber', 'internet', 'Fiber optic Internet connection'),
    ('InternetService_No', 'internet', 'No internet service'),
    ('OnlineSecurity', 'support', 'Online security add-on'),
    ('OnlineBackup', 'support', 'Online backup add-on'),
    ('DeviceProtection', 'support', 'Device protection plan'),
    ('TechSupport', 'support', 'Technical support add-on'),
    ('StreamingTV', 'streaming', 'TV streaming service'),
    ('StreamingMovies', 'streaming', 'Movie streaming service');

-- ===========================================
-- MAIN TABLES
-- ===========================================

CREATE TABLE customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    gender VARCHAR(10) NOT NULL CHECK (gender IN ('Male', 'Female')),
    senior_citizen BOOLEAN NOT NULL DEFAULT FALSE,
    partner BOOLEAN NOT NULL DEFAULT FALSE,
    dependents BOOLEAN NOT NULL DEFAULT FALSE,
    tenure_months INTEGER NOT NULL CHECK (tenure_months >= 0),
    monthly_charges DECIMAL(10, 2) NOT NULL CHECK (monthly_charges >= 0),
    total_charges DECIMAL(10, 2) CHECK (total_charges >= 0),
    churn BOOLEAN NOT NULL DEFAULT FALSE,
    churn_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE customer_contracts (
    contract_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    contract_type_id INTEGER NOT NULL REFERENCES contract_types(contract_type_id),
    payment_method VARCHAR(50) NOT NULL,
    paperless_billing BOOLEAN NOT NULL DEFAULT FALSE,
    contract_start_date DATE,
    contract_end_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(customer_id)
);

CREATE TABLE customer_services (
    customer_service_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    service_type_id INTEGER NOT NULL REFERENCES service_types(service_type_id),
    is_subscribed BOOLEAN NOT NULL DEFAULT TRUE,
    subscription_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(customer_id, service_type_id)
);

CREATE TABLE predictions (
    prediction_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    churn_probability DECIMAL(5, 4) NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
    churn_prediction BOOLEAN NOT NULL,
    risk_level VARCHAR(10) NOT NULL CHECK (risk_level IN ('low', 'medium', 'high')),
    model_version VARCHAR(50),
    top_factors JSONB,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- INDEXES
-- ===========================================

CREATE INDEX idx_customers_churn ON customers(churn);
CREATE INDEX idx_customers_tenure ON customers(tenure_months);
CREATE INDEX idx_customers_monthly_charges ON customers(monthly_charges);
CREATE INDEX idx_contracts_customer ON customer_contracts(customer_id);
CREATE INDEX idx_contracts_type ON customer_contracts(contract_type_id);
CREATE INDEX idx_services_customer ON customer_services(customer_id);
CREATE INDEX idx_predictions_customer ON predictions(customer_id);
CREATE INDEX idx_predictions_risk ON predictions(risk_level);

-- ===========================================
-- VIEWS
-- ===========================================

CREATE OR REPLACE VIEW v_customer_full AS
SELECT 
    c.customer_id,
    c.gender,
    c.senior_citizen,
    c.partner,
    c.dependents,
    c.tenure_months,
    c.monthly_charges,
    c.total_charges,
    c.churn,
    ct.contract_name AS contract_type,
    cc.payment_method,
    cc.paperless_billing,
    (SELECT COUNT(*) FROM customer_services cs 
     WHERE cs.customer_id = c.customer_id AND cs.is_subscribed = TRUE) AS total_services
FROM customers c
LEFT JOIN customer_contracts cc ON c.customer_id = cc.customer_id
LEFT JOIN contract_types ct ON cc.contract_type_id = ct.contract_type_id;

CREATE OR REPLACE VIEW v_customer_services_summary AS
SELECT 
    cs.customer_id,
    BOOL_OR(CASE WHEN st.service_name = 'PhoneService' THEN cs.is_subscribed ELSE FALSE END) AS has_phone_service,
    BOOL_OR(CASE WHEN st.service_name = 'MultipleLines' THEN cs.is_subscribed ELSE FALSE END) AS has_multiple_lines,
    BOOL_OR(CASE WHEN st.service_name = 'InternetService_DSL' THEN TRUE ELSE FALSE END) AS has_dsl,
    BOOL_OR(CASE WHEN st.service_name = 'InternetService_Fiber' THEN TRUE ELSE FALSE END) AS has_fiber,
    BOOL_OR(CASE WHEN st.service_name = 'OnlineSecurity' THEN cs.is_subscribed ELSE FALSE END) AS has_online_security,
    BOOL_OR(CASE WHEN st.service_name = 'OnlineBackup' THEN cs.is_subscribed ELSE FALSE END) AS has_online_backup,
    BOOL_OR(CASE WHEN st.service_name = 'DeviceProtection' THEN cs.is_subscribed ELSE FALSE END) AS has_device_protection,
    BOOL_OR(CASE WHEN st.service_name = 'TechSupport' THEN cs.is_subscribed ELSE FALSE END) AS has_tech_support,
    BOOL_OR(CASE WHEN st.service_name = 'StreamingTV' THEN cs.is_subscribed ELSE FALSE END) AS has_streaming_tv,
    BOOL_OR(CASE WHEN st.service_name = 'StreamingMovies' THEN cs.is_subscribed ELSE FALSE END) AS has_streaming_movies
FROM customer_services cs
JOIN service_types st ON cs.service_type_id = st.service_type_id
GROUP BY cs.customer_id;

CREATE OR REPLACE VIEW v_churn_summary AS
SELECT 
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN churn THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(AVG(tenure_months), 1) AS avg_tenure_months,
    ROUND(SUM(CASE WHEN churn THEN monthly_charges ELSE 0 END), 2) AS monthly_revenue_at_risk
FROM customers;

-- ===========================================
-- FUNCTIONS
-- ===========================================

CREATE OR REPLACE FUNCTION get_risk_level(probability DECIMAL(5, 4))
RETURNS VARCHAR(10) AS $$
BEGIN
    RETURN CASE 
        WHEN probability >= 0.7 THEN 'high'
        WHEN probability >= 0.4 THEN 'medium'
        ELSE 'low'
    END;
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- TRIGGERS
-- ===========================================

CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER customers_update_timestamp
    BEFORE UPDATE ON customers
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER contracts_update_timestamp
    BEFORE UPDATE ON customer_contracts
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp();