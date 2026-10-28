"""Part A: pandas cross-check, city aggregations, IQR and correlations."""
from pathlib import Path
import sqlite3
import pandas as pd


def main():
    orders = pd.read_csv('data/clean/orders.csv')
    customers = pd.read_csv('data/clean/customers.csv')
    products = pd.read_csv('data/clean/products.csv')
    done = orders.loc[orders.status.eq('completed')].copy()
    merged = done.merge(customers[['customer_id', 'city']], on='customer_id').merge(
        products[['product_id', 'category']], on='product_id')
    pivot = pd.pivot_table(merged, values='order_amount', index='city', columns='category',
                           aggfunc='sum', margins=True, margins_name='All', fill_value=0)
    by_city = merged.groupby('city').agg(total_revenue=('order_amount', 'sum'),
                                         mean_order_amount=('order_amount', 'mean'),
                                         completed_orders=('order_id', 'count')).round(2)
    with sqlite3.connect('data/instamart.db') as conn:
        sql_total = pd.read_sql_query("SELECT SUM(order_amount) AS total FROM orders WHERE status='completed'", conn).iloc[0, 0]
    assert abs(pivot.loc['All', 'All'] - sql_total) < 0.01
    amount = done.order_amount
    q1, q3 = amount.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    corr = done[['order_amount', 'discount_pct', 'quantity', 'delivery_minutes']].corr()
    pairs = corr.where(~pd.DataFrame([[i == j for j in corr.columns] for i in corr.index],
                                    index=corr.index, columns=corr.columns)).abs().stack()
    largest = pairs.sort_values(ascending=False).index[0]
    lines = [
        'PANDAS CROSS-CHECK', pivot.round(2).to_string(),
        f'Grand total: INR {pivot.loc["All", "All"]:,.2f} (SQL: INR {sql_total:,.2f})',
        '\nCITY AGGREGATIONS', by_city.to_string(),
        '\nCOMPLETED ORDER DISTRIBUTION',
        f'Mean: {amount.mean():.2f}; median: {amount.median():.2f}; skewness: {amount.skew():.4f}',
        f'Q1: {q1:.2f}; Q3: {q3:.2f}; IQR: {iqr:.2f}',
        f'Lower fence: {lower:.2f}; upper fence: {upper:.2f}',
        f'Outliers below: {int(amount.lt(lower).sum())}; above: {int(amount.gt(upper).sum())}',
        '\nCORRELATION MATRIX', corr.round(4).to_string(),
        f'Largest absolute correlation: {largest[0]} and {largest[1]} = {corr.loc[largest]:.4f}',
    ]
    report = '\n'.join(lines)
    Path('outputs').mkdir(exist_ok=True)
    Path('outputs/pandas_report.txt').write_text(report + '\n', encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
