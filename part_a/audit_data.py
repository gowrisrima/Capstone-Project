"""Part A: audit raw exports; writes a reproducible text report."""
from pathlib import Path
import pandas as pd

RAW = Path('data/raw')
OUT = Path('outputs')
OUT.mkdir(exist_ok=True)


def audit_table(name):
    df = pd.read_csv(RAW / f'{name}.csv')
    lines = [f'\n{name.upper()} shape={df.shape}', 'Dtypes:', df.dtypes.to_string(), 'Missing values:']
    missing = pd.DataFrame({'count': df.isna().sum(), 'percent': (df.isna().mean() * 100).round(2)})
    lines.append(missing.to_string())
    return df, '\n'.join(lines)


def main():
    customers, c = audit_table('customers')
    products, p = audit_table('products')
    orders, o = audit_table('orders')
    amounts = pd.to_numeric(orders['order_amount'], errors='coerce')
    delivery = pd.to_numeric(orders['delivery_minutes'], errors='coerce')
    ages = pd.to_numeric(customers['age'], errors='coerce')
    gender = customers['gender'].fillna('').astype(str)
    figures = {
        'raw order rows': len(orders),
        'exact duplicate order rows': int(orders.duplicated().sum()),
        'pending rows': int(orders['status'].eq('pending').sum()),
        'distinct order ids': int(orders['order_id'].nunique()),
        'non-numeric order amounts': int(amounts.isna().sum()),
        'negative delivery minutes': int(delivery.lt(0).sum()),
        'blank delivery minutes': int(delivery.isna().sum()),
        'blank ages': int(ages.isna().sum()),
        'ages outside 10 to 100': int(((ages < 10) | (ages > 100)).sum()),
        'distinct raw city spellings': int(customers['city'].nunique()),
        'distinct nonblank gender spellings': int(gender[gender.str.strip().ne('')].nunique()),
        'blank genders': int(gender.str.strip().eq('').sum()),
    }
    report = '\n\n'.join([c, p, o, 'AUDIT CHECKPOINTS\n' + '\n'.join(f'{k}: {v}' for k, v in figures.items())])
    (OUT / 'audit_report.txt').write_text(report + '\n', encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
