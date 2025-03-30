# Business Requirements Document (BRD) for Retail Analytics Dashboards




## 1. Overview

The Retail Analytics Dashboards provide key business insights into customer behavior, sales performance, and profitability across different categories, locations, and timeframes. The dashboards help stakeholders make data-driven decisions to optimize sales strategies and customer engagement.



## 2. Objectives

Track total sales, quantity sold, and profit trends over time.

Identify the top-performing and underperforming products and customer segments.

Compare year-over-year performance metrics.

Provide actionable insights on customer distribution based on order frequency.

Visualize revenue and profit trends for different product subcategories.

Enable filtering by year, product category, subcategory, and location.

3. Key Performance Indicators (KPIs)

Total Sales per Customer

Total Quantity Sold

Total Orders Placed

Total Profit

Year-over-Year Sales Growth (%)

Revenue & Profit by Product Subcategory

Top 10 Customers by Profit

Customers Distribution by Number of Orders

Revenue vs. Profit Trends Over Time



## About Dataset

I generated a fake dataset with 22 columns and 10 million records using Python.

The Orders database contains information on the following variables:

Continuous variables: Order ID, Order Date, Ship Date, Customer ID, Product ID, Sales, Quantity, Discount, Profit, LoyaltyProgram.

Categorical variables: Ship Mode, Customer Name, Segment, Postal Code, City, State, Country, Region, Category, Sub-Category, Product Name, Order Priority.



📫 Connect with me on [LinkedIn] https://www.linkedin.com/in/juniorzephir/


import polars as pl
import numpy as np
import re
from datetime import datetime
from dateutil.parser import parse
from typing import Dict, Any, Optional, Set, Type, List, Union
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_profiler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataQualityProfiler:
    def __init__(self, file_path: str, sample_size: int = 100000, test_mode: bool = False):
        self.file_path = Path(file_path)
        self.sample_size = sample_size
        self.test_mode = test_mode
        self.df: Optional[pl.DataFrame] = None
        self._compile_regex_patterns()

    def _compile_regex_patterns(self):
        """Pre-compile regex patterns for better performance"""
        self.date_patterns = [
            re.compile(p) for p in [
                # Date patterns
                r'^\d{4}-\d{2}-\d{2}$',                  # YYYY-MM-DD
                r'^\d{2}/\d{2}/\d{4}$',                  # MM/DD/YYYY
                r'^\d{2}-\d{2}-\d{4}$',                  # DD-MM-YYYY
                r'^\d{1,2}[A-Za-z]{3}\d{4}$',            # 01Jan2023
                
                # Datetime patterns
                r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}$',  # With space/T
                r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\.\d+$',  # With milliseconds
                r'^\d{2}/\d{2}/\d{4} \d{1,2}:\d{2} [AP]M$',  # AM/PM format
                r'^\d{4}\d{2}\d{2}\d{2}\d{2}\d{2}$',     # Compact format
                r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$',      # Without seconds
                r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[+-]\d{4}$',  # With TZ
                r'^\d{2}-\d{2}-\d{4} \d{2}h\d{2}m$',     # Custom format
            ]
        ]

    def _load_data(self) -> pl.DataFrame:
        """Load data efficiently with streaming"""
        logger.info(f"Loading data from {self.file_path}")
        try:
            df = pl.scan_parquet(self.file_path)
            if self.test_mode:
                df = df.head(self.sample_size)
            return df.collect()
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def _is_integer(self, val: Union[str, int, float]) -> bool:
        """Check if value can be an integer"""
        try:
            if isinstance(val, str):
                int(val)
            elif isinstance(val, float):
                return val.is_integer()
            return True
        except (ValueError, TypeError):
            return False

    def _is_float(self, val: Union[str, int, float]) -> bool:
        """Check if value can be a float"""
        try:
            float(val)
            return not self._is_integer(val)
        except (ValueError, TypeError):
            return False

    def _is_potential_date(self, val: Any) -> bool:
        """Comprehensive datetime detection with validation"""
        if isinstance(val, datetime):
            return True
            
        if not isinstance(val, str):
            return False
        
        val = str(val).strip()
        if not val:
            return False
        
        # Check against pre-compiled patterns
        if any(p.fullmatch(val) for p in self.date_patterns):
            try:
                parsed = parse(val, fuzzy=False)
                # Validate reasonable date ranges
                return 1970 <= parsed.year <= 2100
            except:
                return False
        return False

    def _is_potential_bool(self, val: Any) -> bool:
        """Check for common boolean representations"""
        if isinstance(val, bool):
            return True
        if not isinstance(val, str):
            return False
            
        return str(val).lower() in {'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'}

    def _detect_types(self, series: pl.Series) -> List[str]:
        """Detect data types present in the series"""
        type_counts = {
            'null': 0,
            'int': 0,
            'float': 0,
            'datetime': 0,
            'bool': 0,
            'str': 0
        }

        sample_size = min(self.sample_size, len(series))
        if sample_size == 0:
            return ['null']
        
        sample = series.sample(n=sample_size, seed=42)

        for val in sample:
            if val is None:
                type_counts['null'] += 1
                continue
                
            if self._is_potential_bool(val):
                type_counts['bool'] += 1
            elif self._is_integer(val):
                type_counts['int'] += 1
            elif self._is_float(val):
                type_counts['float'] += 1
            elif self._is_potential_date(val):
                type_counts['datetime'] += 1
            else:
                type_counts['str'] += 1

        detected_types = [typ for typ, count in type_counts.items() if count > 0]
        if len(detected_types) == 1 and detected_types[0] == 'null':
            return ['null']
        return [t for t in detected_types if t != 'null'] or ['untyped']

    def _suggest_better_type(self, series: pl.Series, detected_types: List[str]) -> str:
        """Suggest optimal data type with precision"""
        dtype = series.dtype
        
        # If already correct type, return as-is
        if (dtype == pl.Int64 and 'int' in detected_types) or \
           (dtype == pl.Float64 and 'float' in detected_types) or \
           (dtype == pl.Boolean and 'bool' in detected_types) or \
           (dtype == pl.Datetime and 'datetime' in detected_types):
            return str(dtype)
        
        sample_size = min(1000, len(series))
        sample = series.sample(n=sample_size, seed=42).drop_nulls()
        
        if len(sample) == 0:
            return str(dtype)
        
        # Enhanced datetime detection
        if dtype == pl.Utf8:
            date_samples = [x for x in sample if self._is_potential_date(x)]
            date_ratio = len(date_samples) / len(sample)
            
            if date_ratio > 0.8:  # 80% confidence threshold
                has_time = any(':' in str(x) for x in date_samples)
                has_ms = any('.' in str(x) for x in date_samples)
                has_tz = any('+' in str(x) or '-' in str(x) for x in date_samples)
                
                if has_tz:
                    return "datetime[ms, UTC]"
                elif has_ms:
                    return "datetime[ms]"
                elif has_time:
                    return "datetime[s]"
                else:
                    return "date"
        
        # Numeric detection
        if dtype == pl.Utf8:
            int_count = sum(self._is_integer(x) for x in sample)
            float_count = sum(self._is_float(x) for x in sample)
            
            if int_count / len(sample) > 0.9:
                return "int64"
            elif float_count / len(sample) > 0.9:
                return "float64"
        
        # Boolean detection
        if dtype == pl.Utf8:
            bool_count = sum(self._is_potential_bool(x) for x in sample)
            if bool_count / len(sample) > 0.9:
                return "bool"
        
        return str(dtype)

    def _calculate_duplicates(self, series: pl.Series) -> float:
        """Calculate percentage of duplicate values"""
        if len(series) == 0:
            return 0.0
        return (1 - (series.n_unique() / len(series))) * 100

    def _detect_outliers(self, series: pl.Series) -> Optional[str]:
        """Detect outliers for numeric columns using IQR method"""
        if series.dtype not in (pl.Int64, pl.Float64):
            return None
            
        try:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_count = series.filter(
                (series < lower_bound) | (series > upper_bound)
            ).len()
            
            if outlier_count > 0:
                return f"{outlier_count} ({(outlier_count/len(series))*100:.1f}%)"
            return "None"
        except Exception:
            return "Error"

    def _has_special_chars(self, val: Any) -> bool:
        """Check for problematic special characters"""
        if not isinstance(val, str):
            return False
        return bool(re.search(r'[\?\/@\*\$\%\^\&\#\!\\\<\>\{\}\[\]\=\+\-\|]', str(val)))

    def _check_inconsistent_data(self, series: pl.Series) -> str:
        """Check for various data inconsistencies"""
        issues = []
        
        if series.dtype == pl.Utf8:
            try:
                # Special character check
                sample_size = min(1000, len(series))
                sample = series.sample(n=sample_size, seed=42).to_list()
                special_char_count = sum(self._has_special_chars(x) for x in sample if x is not None)
                if special_char_count > 0:
                    issues.append(f"Special chars ({special_char_count})")
                
                # Empty string check
                empty_count = series.filter(pl.col(series.name).str.strip().eq("")).len()
                if empty_count > 0:
                    issues.append(f"Empty strings ({empty_count})")
            except Exception as e:
                logger.debug(f"Consistency check failed for {series.name}: {str(e)}")
        
        return ", ".join(issues) if issues else "None"

    def generate_quality_report(self) -> pl.DataFrame:
        """Generate comprehensive data quality report"""
        self.df = self._load_data()
        report_rows = []

        for col in self.df.columns:
            col_data = self.df[col]
            types_found = self._detect_types(col_data)
            current_type = str(col_data.dtype)
            suggested_type = self._suggest_better_type(col_data, types_found)
            
            report_rows.append({
                "Column Name": col,
                "Current Type": current_type,
                "Suggested Type": suggested_type,
                "Type Mismatch": suggested_type != current_type,
                "Mixed Types": ", ".join(types_found) if len(types_found) > 1 else "None",
                "Type Count": len(types_found),
                "Null %": f"{(col_data.null_count() / len(col_data)) * 100:.1f}%",
                "Duplicate %": f"{self._calculate_duplicates(col_data):.1f}%",
                "Outliers": self._detect_outliers(col_data) or "N/A",
                "Inconsistent Data": self._check_inconsistent_data(col_data),
                "Unique Values": col_data.n_unique(),
                "Sample Values": ", ".join(
                    str(x) for x in col_data.drop_nulls().unique().head(3).to_list()
                ) if len(col_data) > 0 else "Empty"
            })

        return pl.DataFrame(report_rows).select([
            "Column Name", "Current Type", "Suggested Type", "Type Mismatch",
            "Mixed Types", "Type Count", "Null %", "Duplicate %", "Outliers",
            "Inconsistent Data", "Unique Values", "Sample Values"
        ])

    def save_report(self, report: pl.DataFrame, output_format: str = "csv"):
        """Save report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_quality_report_{timestamp}"
        
        if output_format.lower() == "csv":
            report.write_csv(f"{filename}.csv")
        elif output_format.lower() == "parquet":
            report.write_parquet(f"{filename}.parquet")
        else:
            report.write_csv(f"{filename}.csv")
        
        logger.info(f"Report saved as {filename}.{output_format}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate data quality report for Parquet files")
    parser.add_argument("file_path", help="Path to the Parquet file")
    parser.add_argument("--sample_size", type=int, default=100000, help="Sample size for analysis")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode with limited data")
    parser.add_argument("--output", choices=["csv", "parquet"], default="csv", help="Output format")
    
    args = parser.parse_args()
    
    try:
        profiler = DataQualityProfiler(
            file_path=args.file_path,
            sample_size=args.sample_size,
            test_mode=args.test_mode
        )
        
        logger.info("Generating data quality report...")
        start_time = time.time()
        
        report = profiler.generate_quality_report()
        print("\nData Quality Report:")
        print(report)
        
        profiler.save_report(report, args.output)
        
        logger.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        exit(1)
