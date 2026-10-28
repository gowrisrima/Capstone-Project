-- Q1: WHERE then HAVING
SELECT customer_id, ROUND(SUM(order_amount), 2) AS qualifying_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY qualifying_spend DESC, customer_id;

-- Q2: three-table JOIN
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN products AS p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC, c.city, p.category;

-- Q3: scalar subquery
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC, product_id;

-- Q4: LEFT JOIN
SELECT c.customer_id, c.city
FROM customers AS c
LEFT JOIN orders AS o
  ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: two CTEs
WITH customer_spend AS (
  SELECT customer_id, SUM(order_amount) AS total_spend
  FROM orders WHERE status = 'completed' GROUP BY customer_id
), average_spend AS (
  SELECT AVG(total_spend) AS avg_spend FROM customer_spend
)
SELECT s.customer_id, ROUND(s.total_spend, 2) AS total_spend,
       ROUND(a.avg_spend, 2) AS average_spend
FROM customer_spend AS s CROSS JOIN average_spend AS a
WHERE s.total_spend > a.avg_spend
ORDER BY total_spend DESC, s.customer_id;

-- Q6: ranking functions by city
WITH spending AS (
  SELECT c.city, o.customer_id, ROUND(SUM(o.order_amount), 2) AS total_spend
  FROM orders AS o JOIN customers AS c ON o.customer_id = c.customer_id
  WHERE o.status = 'completed'
  GROUP BY c.city, o.customer_id
)
SELECT city, customer_id, total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_number
FROM spending
ORDER BY city, spend_rank, customer_id;

-- Q7: daily running revenue
WITH daily AS (
  SELECT SUBSTR(order_ts, 1, 10) AS day, ROUND(SUM(order_amount), 2) AS revenue
  FROM orders WHERE status = 'completed'
  GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day, revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily ORDER BY day;
