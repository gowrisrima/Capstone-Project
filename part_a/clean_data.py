"""Part A: clean three tables, save CSVs and SQLite database."""
from pathlib import Path
import sqlite3
import pandas as pd

RAW = Path('data/raw')
CLEAN = Path('data/clean')
DB = Path('data/instamart.db')


def main():
    CLEAN.mkdir(parents=True, exist_ok=True)
    customers = pd.read_csv(RAW / 'customers.csv')
    products = pd.read_csv(RAW / 'products.csv')
    orders = pd.read_csv(RAW / 'orders.csv')

    # Customers: normalize city and gender, and impute invalid/missing ages.
    customers['city'] = customers['city'].str.strip().str.title()
    genders = customers['gender'].fillna('').str.strip().str.lower()
    customers['gender'] = genders.map({'m': 'male', 'male': 'male', 'f': 'female', 'female': 'female'}).fillna('unknown')
    customers['age'] = pd.to_numeric(customers['age'], errors='coerce')
    customers.loc[~customers['age'].between(10, 100), 'age'] = float('nan')
    customers['age'] = customers['age'].fillna(customers.groupby('city')['age'].transform('median'))
    customers['signup_date'] = pd.to_datetime(customers['signup_date'])

    # Orders: exact duplicates first, then latest row for each order_id.
    orders['order_ts'] = pd.to_datetime(orders['order_ts'])
    before = len(orders)
    orders = orders.drop_duplicates()
    exact_removed = before - len(orders)
    orders = (orders.sort_values(['order_ts'], kind='stable')
              .drop_duplicates(subset=['order_id'], keep='last')
              .sort_values('order_id').reset_index(drop=True))

    # Repair invalid order amounts using the product catalogue.
    orders['order_amount'] = pd.to_numeric(orders['order_amount'], errors='coerce')
    orders['quantity'] = pd.to_numeric(orders['quantity'])
    orders['discount_pct'] = pd.to_numeric(orders['discount_pct'])
    price = orders['product_id'].map(products.set_index('product_id')['price_inr'])
    calculated = (price * orders['quantity'] * (1 - orders['discount_pct'] / 100)).round(2)
    repaired_amounts = int(orders['order_amount'].isna().sum())
    orders['order_amount'] = orders['order_amount'].fillna(calculated)

    # Negative durations are impossible. Completed orders use category medians.
    orders['delivery_minutes'] = pd.to_numeric(orders['delivery_minutes'], errors='coerce')
    negatives = int(orders['delivery_minutes'].lt(0).sum())
    orders.loc[orders['delivery_minutes'] < 0, 'delivery_minutes'] = float('nan')
    orders['category'] = orders['product_id'].map(products.set_index('product_id')['category'])
    complete = orders['status'].eq('completed')
    missing_completed = int((complete & orders['delivery_minutes'].isna()).sum())
    category_median = orders.loc[complete].groupby('category')['delivery_minutes'].transform('median')
    orders.loc[complete, 'delivery_minutes'] = orders.loc[complete, 'delivery_minutes'].fillna(category_median)
    orders.loc[orders['status'].eq('cancelled'), 'delivery_minutes'] = float('nan')
    orders = orders.drop(columns='category')

    # SQLite stores dates as ISO strings; the CSVs use the same representation.
    customers['signup_date'] = customers['signup_date'].dt.strftime('%Y-%m-%d')
    orders['order_ts'] = orders['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
    for name, frame in [('customers', customers), ('products', products), ('orders', orders)]:
        frame.to_csv(CLEAN / f'{name}.csv', index=False)
    with sqlite3.connect(DB) as conn:
        for name, frame in [('customers', customers), ('products', products), ('orders', orders)]:
            frame.to_sql(name, conn, if_exists='replace', index=False)
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_order_id ON orders(order_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_orders_product ON orders(product_id)')

    print(f'Exact duplicates removed: {exact_removed}')
    print(f'Partial duplicates removed: {before - exact_removed - len(orders)}')
    print(f'Amounts repaired: {repaired_amounts}')
    print(f'Negative delivery durations invalidated: {negatives}')
    print(f'Completed-order delivery durations imputed: {missing_completed}')
    print(f'Clean customers: {len(customers)}; products: {len(products)}; orders: {len(orders)}')
    print(f'Completed revenue: INR {orders.loc[complete, "order_amount"].sum():,.2f}')


if __name__ == '__main__':
    main()
