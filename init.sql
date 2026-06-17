-- ── Phase H — docker/mysql/init.sql ──────────────────────────────────────────
-- Runs once when the MySQL container is first created.
-- Creates the database and grants privileges to the app user.

CREATE DATABASE IF NOT EXISTS nlp_classifier
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

-- Grant all privileges to the app user on the database
GRANT ALL PRIVILEGES ON nlp_classifier.* TO 'nlp_user'@'%';
FLUSH PRIVILEGES;
