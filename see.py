import duckdb

con = duckdb.connect('attribution.duckdb')

# View data as pandas DataFrame (cleanest output)
df = con.execute("SELECT * FROM raw_clicks LIMIT 10").df()
print(df)

# Or get basic info
print("\n=== Row Count ===")
print(con.execute("SELECT COUNT(*) as total_rows FROM raw_clicks").fetchone())

print("\n=== Channel Distribution ===")
print(con.execute("SELECT channel, COUNT(*) as count FROM raw_clicks GROUP BY channel").df())

con.close()