USE ROLE ACCOUNTADMIN;
USE WAREHOUSE ETL_WH;
USE DATABASE SALES_ANALYSIS_DB;
USE SCHEMA SALES_ANALYSIS_DB_SCHEMA;
SHOW WAREHOUSES;
--Query #1
--Total Sales Per Customer for a Given Year
SELECT
    c.CustomerID,
    c.CustomerName,
    SUM(o.Sales) AS Total_Sales
FROM
    Orders o
    JOIN Customers c ON o.CustomerID = c.CustomerID
WHERE
    YEAR(o.OrderDate) = 2022
GROUP BY
    c.CustomerID,
    c.CustomerName
ORDER BY
    Total_Sales DESC
Limit
    10;
    --Query #2
    -- Total Orders Per Customer for a Given Year
SELECT
    c.CustomerID,
    c.CustomerName,
    COUNT(OrderID) AS Total_Orders
FROM
    Orders o
    JOIN Customers c ON o.CustomerID = c.CustomerID
WHERE
    YEAR(OrderDate) = 2022
GROUP BY
    c.CustomerID,
    c.CustomerName
ORDER BY
    Total_Orders DESC
Limit
    10;
--Query 3
    --Total Profit for a Given Year
SELECT
    SUM(Profit) AS Total_Profit
FROM
    Orders
WHERE
    YEAR(OrderDate) = 2022;
    -- query3
    --Monthly Sales Trend Over the Year
SELECT
    MONTH(OrderDate) AS Month,
    SUM(Sales) AS Monthly_Sales
FROM
    Orders
WHERE
    YEAR(OrderDate) = 2022
GROUP BY
    MONTH(OrderDate)
ORDER BY
    Month;
    --query 4
    -- Top 10 customers by total sales
    WITH Customer_Sales AS (
        SELECT
            o.CustomerID,
            c.CustomerName,
            SUM(o.Sales) AS TotalSales,
            COUNT(DISTINCT o.OrderID) AS TotalOrders,
            RANK() OVER (
                ORDER BY
                    SUM(o.Sales) DESC
            ) AS SalesRank
        FROM
            Orders o
            JOIN Customers c ON o.CustomerID = c.CustomerID
        GROUP BY
            o.CustomerID,
            c.CustomerName
    )
SELECT
    *
FROM
    Customer_Sales
WHERE
    SalesRank <= 10;
-- query 5
    --Monthly Revenue Growth Rate
    WITH Monthly_Sales AS (
        SELECT
            DATE_TRUNC('month', OrderDate) AS Month,
            SUM(Sales) AS Monthly_Revenue
        FROM
            Orders
        GROUP BY
            Month
    )
SELECT
    Month,
    Monthly_Revenue,
    LAG(Monthly_Revenue) OVER (
        ORDER BY
            Month
    ) AS Previous_Month_Revenue,
    (
        Monthly_Revenue - LAG(Monthly_Revenue) OVER (
            ORDER BY
                Month
        )
    ) * 100.0 / NULLIF(
        LAG(Monthly_Revenue) OVER (
            ORDER BY
                Month
        ),
        0
    ) AS Growth_Rate_Percentage
FROM
    Monthly_Sales;
    --query 6
    -- Top-Selling Products with Category Contribution
SELECT
    p.ProductID,
    p.ProductName,
    p.Category,
    SUM(o.Sales) AS Total_Sales,
    (SUM(o.Sales) * 100.0) / (
        SELECT
            SUM(Sales)
        FROM
            Orders
    ) AS Category_Contribution
FROM
    Orders o
    JOIN Product p ON o.ProductID = p.ProductID
GROUP BY
    p.ProductID,
    p.ProductName,
    p.Category
ORDER BY
    Total_Sales DESC
LIMIT
    10;
    -- querry 7
    -- Customer Churn Prediction alert
    --This query categorizes customers into Active, At Risk, and Churned based on their last purchase date
SELECT
    c.CustomerID,
    c.CustomerName,
    MAX(o.OrderDate) AS Last_Order_Date,
    COUNT(o.OrderID) AS Total_Orders,
    CASE
        WHEN MAX(o.OrderDate) < DATEADD(
            month,
            -6,
            (
                SELECT
                    MAX(OrderDate)
                FROM
                    Orders
            )
        ) THEN 'Churned'
        WHEN MAX(o.OrderDate) BETWEEN DATEADD(
            month,
            -6,
            (
                SELECT
                    MAX(OrderDate)
                FROM
                    Orders
            )
        )
        AND DATEADD(
            month,
            -3,
            (
                SELECT
                    MAX(OrderDate)
                FROM
                    Orders
            )
        ) THEN 'At Risk'
        ELSE 'Active'
    END AS Customer_Status
FROM
    Orders o
    JOIN Customers c ON o.CustomerID = c.CustomerID
GROUP BY
    c.CustomerID,
    c.CustomerName;
    -- query 8
    -- Shipping Performance & Delay Analysis
SELECT
    ShipMode,
    COUNT(OrderID) AS Total_Orders,
    AVG(DATEDIFF(day, OrderDate, ShipDate)) AS Avg_Delivery_Time,
    SUM(
        CASE
            WHEN DATEDIFF(day, OrderDate, ShipDate) > 5 THEN 1
            ELSE 0
        END
    ) AS Late_Shipments,
    (
        SUM(
            CASE
                WHEN DATEDIFF(day, OrderDate, ShipDate) > 5 THEN 1
                ELSE 0
            END
        ) * 100.0
    ) / COUNT(OrderID) AS Late_Shipment_Percentage
FROM
    Orders
GROUP BY
    ShipMode
ORDER BY
    Late_Shipment_Percentage DESC;