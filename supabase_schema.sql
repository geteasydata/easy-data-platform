-- ============================================
-- Easy Data Platform - Supabase Schema
-- ============================================
-- Run this SQL in Supabase SQL Editor to create the required tables
-- Go to: https://supabase.com/dashboard/project/YOUR_PROJECT/sql

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ===========================================
-- Users Table
-- ===========================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    name TEXT,
    plan TEXT DEFAULT 'free' CHECK (plan IN ('free', 'pro', 'enterprise', 'admin')),
    role TEXT DEFAULT 'user' CHECK (role IN ('user', 'admin', 'guest')),
    stripe_customer_id TEXT,
    email_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT username_length CHECK (char_length(username) >= 3),
    CONSTRAINT email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_stripe_customer ON users(stripe_customer_id);

-- ===========================================
-- Subscriptions Table
-- ===========================================
CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    stripe_subscription_id TEXT UNIQUE NOT NULL,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'canceled', 'past_due', 'trialing', 'incomplete')),
    plan TEXT DEFAULT 'pro' CHECK (plan IN ('pro', 'enterprise')),
    current_period_start TIMESTAMP WITH TIME ZONE,
    current_period_end TIMESTAMP WITH TIME ZONE,
    cancel_at_period_end BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_subscriptions_user ON subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_stripe ON subscriptions(stripe_subscription_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);

-- ===========================================
-- Usage Logs Table
-- ===========================================
CREATE TABLE IF NOT EXISTS usage_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    metadata JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for faster analytics queries
CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_action ON usage_logs(action);
CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_logs(timestamp);

-- Partial index for today's usage (performance optimization)
CREATE INDEX IF NOT EXISTS idx_usage_today ON usage_logs(user_id, timestamp) 
    WHERE timestamp >= CURRENT_DATE;

-- ===========================================
-- Verification Tokens Table (Optional - can use local file)
-- ===========================================
CREATE TABLE IF NOT EXISTS verification_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_hash TEXT UNIQUE NOT NULL,
    email TEXT NOT NULL,
    token_type TEXT NOT NULL CHECK (token_type IN ('verify', 'reset')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_tokens_hash ON verification_tokens(token_hash);

-- Auto-delete expired tokens
CREATE OR REPLACE FUNCTION delete_expired_tokens()
RETURNS TRIGGER AS $$
BEGIN
    DELETE FROM verification_tokens WHERE expires_at < NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER cleanup_expired_tokens
    AFTER INSERT ON verification_tokens
    EXECUTE FUNCTION delete_expired_tokens();

-- ===========================================
-- Row Level Security (RLS) Policies
-- ===========================================

-- Enable RLS
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_logs ENABLE ROW LEVEL SECURITY;

-- Users can read their own data
CREATE POLICY "Users can read own data" ON users
    FOR SELECT USING (auth.uid()::text = id::text);

-- Subscriptions are visible to their owners
CREATE POLICY "Users can read own subscriptions" ON subscriptions
    FOR SELECT USING (user_id IN (SELECT id FROM users WHERE auth.uid()::text = id::text));

-- Usage logs are visible to their owners
CREATE POLICY "Users can read own usage" ON usage_logs
    FOR SELECT USING (user_id IN (SELECT id FROM users WHERE auth.uid()::text = id::text));

-- ===========================================
-- Service Role Policies (for backend operations)
-- ===========================================

-- Allow service role full access (for backend API calls)
-- Note: The service_role key bypasses RLS by default

-- ===========================================
-- Helper Functions
-- ===========================================

-- Function to get user's daily usage count
CREATE OR REPLACE FUNCTION get_daily_usage(p_user_id UUID)
RETURNS INTEGER AS $$
BEGIN
    RETURN (
        SELECT COUNT(*)
        FROM usage_logs
        WHERE user_id = p_user_id
        AND timestamp >= CURRENT_DATE
        AND action = 'analysis'
    );
END;
$$ LANGUAGE plpgsql;

-- Function to check if user can perform action
CREATE OR REPLACE FUNCTION can_use_service(p_user_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    v_plan TEXT;
    v_usage INTEGER;
    v_limit INTEGER;
BEGIN
    -- Get user's plan
    SELECT plan INTO v_plan FROM users WHERE id = p_user_id;
    
    -- Set limit based on plan
    v_limit := CASE v_plan
        WHEN 'free' THEN 3
        WHEN 'pro' THEN 999999
        WHEN 'enterprise' THEN 999999
        WHEN 'admin' THEN 999999
        ELSE 3
    END;
    
    -- Get today's usage
    v_usage := get_daily_usage(p_user_id);
    
    RETURN v_usage < v_limit;
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- Sample Admin User (Optional)
-- ===========================================
-- Uncomment and run to create an admin user
-- Note: Replace the password hash with a real bcrypt hash

-- INSERT INTO users (username, email, password_hash, name, plan, role, email_verified)
-- VALUES (
--     'admin',
--     'admin@geteasydata.com',
--     '$2b$12$YOUR_BCRYPT_HASH_HERE',
--     'Admin User',
--     'enterprise',
--     'admin',
--     TRUE
-- );

-- ===========================================
-- Done!
-- ===========================================
-- Your Supabase database is now ready for Easy Data Platform
-- Next steps:
-- 1. Copy your Supabase URL and anon key to secrets.toml
-- 2. Test the connection by running the app
-- 3. Create your first user through the app's registration
