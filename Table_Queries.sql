USE ROLE ACCOUNTADMIN;
USE WAREHOUSE ETL_WH;
USE DATABASE SALES_ANALYSIS_DB;
USE SCHEMA SALES_ANALYSIS_DB_SCHEMA;
SHOW WAREHOUSES;
-- create database
CREATE DATABASE IF NOT EXISTS SALES_ANALYSIS_DB;
-- create schema
CREATE SCHEMA IF NOT EXISTS SALES_ANALYSIS_DB_SCHEMA;
CREATE
OR REPLACE STAGE SALES_STAGE;
-- create our table
-- Dimension Table:Loyalty_Program
CREATE
OR REPLACE TABLE LoyaltyProgram (
    ProgramID INT PRIMARY KEY,
    ProgramTitle VARCHAR(100),
    MembershipLevel VARCHAR(50),
    RewardPoints INT
);
INSERT INTO
    LoyaltyProgram (
        ProgramID,
        ProgramTitle,
        MembershipLevel,
        RewardPoints
    )
VALUES
    (1, 'Gold Rewards', 'Gold', 3000),
    (2, 'Platinum Perks', 'Platinum', 4000),
    (3, 'Silver Savers', 'Silver', 1500),
    (4, 'Bronze Benefits', 'Bronze', 1000),
    (5, 'Exclusive Elite', 'Elite', 5500);
-- SELECT * FROM SALES_ANALYSIS_DB.SALES_ANALYSIS_DB_SCHEMA.LOYALTYPROGRAM;
    CREATE
    OR REPLACE TABLE Location (
        PostalCode VARCHAR(50) PRIMARY KEY,
        City VARCHAR(50),
        State VARCHAR(50),
        Region VARCHAR(50),
        Country VARCHAR(50),
        Address VARCHAR(255)
    );
    --drop table customer;
    -- Dimension Table: Customers
    CREATE
    OR REPLACE TABLE Customers (
        CustomerID VARCHAR(50) PRIMARY KEY,
        CustomerName VARCHAR(50),
        Gender VARCHAR(20),
        DOB DATE,
        ProgramID INT NULL,
        FOREIGN KEY (ProgramID) REFERENCES Loyalty_Program(PROGRAM_ID)
    );
    -- Dimension Table: Product
    CREATE
    OR REPLACE TABLE Product (
        ProductID VARCHAR(50) PRIMARY KEY,
        ProductName VARCHAR(255),
        Category VARCHAR(50),
        SubCategory VARCHAR(50),
        UnitPrice DECIMAL(10, 2)
    );
-- Dimension Table: Store
    CREATE
    OR REPLACE TABLE Store (
        StoreID VARCHAR(50) PRIMARY KEY,
        StoreName VARCHAR(100),
        StoreType VARCHAR(50),
        ManagerName VARCHAR(100)
    );
    -- Fact Table: Orders
    CREATE
    OR REPLACE TABLE Orders (
        OrderID VARCHAR(50) PRIMARY KEY,
        OrderDate DATE,
        ShipDate DATE,
        ShipMode VARCHAR(50),
        Segment VARCHAR(50),
        PostalCode VARCHAR(50),
        CustomerID VARCHAR(50),
        ProductID VARCHAR(50),
        StoreID VARCHAR(50),
        Quantity INT,
        Sales DECIMAL(10, 2),
        Discount DECIMAL(10, 2),
        ShippingCost DECIMAL(10, 2),
        Profit DECIMAL(10, 2),
        FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),
        FOREIGN KEY (ProductID) REFERENCES Product(ProductID),
        FOREIGN KEY (StoreID) REFERENCES Store(StoreID)
    );
-- create a new user
    -- grant it accountadmin access
    -- use that user to load the data
    SHOW DATABASES;