import logfire


logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_requests()
logfire.instrument_httpx(capture_all=True)
logfire.instrument_system_metrics(base="full")
logfire.instrument_sqlite3()
logfire.instrument_redis(capture_statement=True)
