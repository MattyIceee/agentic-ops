-- Initialize Langfuse database on first postgres startup
CREATE DATABASE langfuse;
GRANT ALL PRIVILEGES ON DATABASE langfuse TO ops_agent;
