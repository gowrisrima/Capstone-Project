"""Run the seven saved SQL queries, and export their results."""
from pathlib import Path
import sqlite3
import pandas as pd


def main():
    sql = Path('sql/queries.sql').read_text(encoding='utf-8')
    queries = [s.strip() for s in sql.split(';') if s.strip()]
    assert len(queries) == 7, f'Expected 7 queries, got {len(queries)}'
    Path('outputs').mkdir(exist_ok=True)
    with sqlite3.connect('data/instamart.db') as conn:
        results = []
        for i, query in enumerate(queries, 1):
            result = pd.read_sql_query(query, conn)
            result.to_csv(f'outputs/sql_q{i}.csv', index=False)
            results.append(result)
            print(f'Q{i}: {len(result)} rows')
            print(result.head(6).to_string(index=False))
            print()
    q1, q2, q3, q4, q5, q6, q7 = results
    assert len(q1) == 30
    assert len(q2) == 30 and abs(q2.revenue.sum() - 2998668.80) < 0.01
    assert len(q3) == 11
    assert len(q4) == 25
    assert len(q5) == 550 and q5.average_spend.iloc[0] == 2552.06
    assert len(q7) == 183 and abs(q7.cumulative_revenue.iloc[-1] - 2998668.80) < 0.01
    tops = q6.loc[q6.spend_rank.eq(1)].set_index('city')
    expected = {'Bengaluru': ('C0082', 12856.20), 'Chennai': ('C0930', 10151.40),
                'Delhi': ('C0291', 13742.60), 'Hyderabad': ('C0743', 12730.05),
                'Mumbai': ('C0693', 14766.55), 'Pune': ('C0426', 10377.10)}
    for city, (cid, amount) in expected.items():
        assert tops.loc[city, 'customer_id'] == cid
        assert abs(tops.loc[city, 'total_spend'] - amount) < 0.01
    print('All seven SQL query checks passed.')
    print('Q6 rank versus row_number mismatches:', int(q6.spend_rank.ne(q6.row_number).sum()))


if __name__ == '__main__':
    main()
