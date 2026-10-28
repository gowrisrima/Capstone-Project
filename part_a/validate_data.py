"""Check reference figures from the assignment; print one success line."""
import sqlite3
import pandas as pd


def main():
    raw_orders = pd.read_csv('data/raw/orders.csv')
    raw_customers = pd.read_csv('data/raw/customers.csv')
    customers = pd.read_csv('data/clean/customers.csv')
    orders = pd.read_csv('data/clean/orders.csv')
    assert len(raw_orders) == 13084
    assert raw_orders.duplicated().sum() == 119
    assert raw_orders['status'].eq('pending').sum() == 656
    assert raw_orders['order_id'].nunique() == 12309
    assert pd.to_numeric(raw_orders['order_amount'], errors='coerce').isna().sum() == 120
    assert pd.to_numeric(raw_orders['delivery_minutes'], errors='coerce').lt(0).sum() == 106
    assert raw_orders['delivery_minutes'].isna().sum() == 2541
    assert raw_customers['age'].isna().sum() == 61
    assert raw_customers['city'].nunique() == 24
    assert len(customers) == 1200 and customers['city'].nunique() == 6
    assert customers['age'].notna().all() and customers['age'].between(18, 60).all()
    assert customers['gender'].value_counts().to_dict() == {'female': 610, 'male': 560, 'unknown': 30}
    assert customers['city'].value_counts().to_dict() == {'Pune': 227, 'Delhi': 210, 'Bengaluru': 197, 'Chennai': 191, 'Hyderabad': 191, 'Mumbai': 184}
    assert len(orders) == 12309 and orders['order_id'].is_unique
    assert orders['status'].value_counts().to_dict() == {'completed': 11110, 'cancelled': 1199}
    assert orders['order_amount'].notna().all()
    assert orders.loc[orders.status.eq('completed'), 'delivery_minutes'].notna().all()
    assert orders.loc[orders.status.eq('cancelled'), 'delivery_minutes'].isna().sum() == 1199
    assert abs(orders.loc[orders.status.eq('completed'), 'order_amount'].sum() - 2998668.80) < 0.001
    with sqlite3.connect('data/instamart.db') as conn:
        assert {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")} >= {'customers', 'products', 'orders'}
        assert conn.execute('SELECT COUNT(*) FROM orders').fetchone()[0] == 12309
    print('ALL CHECKPOINTS PASSED')


if __name__ == '__main__':
    main()
