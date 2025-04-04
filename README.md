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
import pandas as pd
from datetime import datetime
from dateutil.parser import parse
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from pathlib import Path

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

class DataProfiler:
    def __init__(self, file_path: str, sample_size: int = 100000, test_mode: bool = False):
        self.file_path = Path(file_path)
        self.sample_size = sample_size
        self.test_mode = test_mode
        self.df: Optional[pl.DataFrame] = None
        self.df_pd: Optional[pd.DataFrame] = None
        self._compile_regex_patterns()
        
        # Configure what we consider as potential false nulls
        self.potential_false_nulls = {
            'numeric': [-1, 999, -999, 9999, -9999, float('inf'), float('-inf')],
            'temporal': ["0000-00-00", "1970-01-01", "1900-01-01"],
            'strings': ["NULL", "MISSING", "UNDEFINED", "NONE", "NA", "N/A", 
                       "\\N", "UNKNOWN", "NOT AVAILABLE", "?", "*"],
            'whitespace': [" ", "\t", "\n", "\r", "\u200B", "\u00A0"]
        }

    def _compile_regex_patterns(self):
        """Pre-compile regex patterns for better performance"""
        self.date_patterns = [
            re.compile(p) for p in [
                r'^\d{4}-\d{2}-\d{2}$',                  # YYYY-MM-DD
                r'^\d{2}/\d{2}/\d{4}$',                  # MM/DD/YYYY
                r'^\d{2}-\d{2}-\d{4}$',                  # DD-MM-YYYY
                r'^\d{1,2}[A-Za-z]{3}\d{4}$',            # 01Jan2023
                r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}$',
                r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\.\d+$',
                r'^\d{2}/\d{2}/\d{4} \d{1,2}:\d{2} [AP]M$',
                r'^\d{4}\d{2}\d{2}\d{2}\d{2}\d{2}$',
                r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$',
                r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[+-]\d{4}$',
                r'^\d{2}-\d{2}-\d{4} \d{2}h\d{2}m$',
            ]
        ]

    def _load_data(self) -> pl.DataFrame:
        """Load data efficiently with streaming"""
        logger.info(f"Loading data from {self.file_path}")
        try:
            df = pl.scan_parquet(self.file_path)
            if self.test_mode:
                df = df.head(self.sample_size)
            df_collected = df.collect()
            self.df_pd = df_collected.to_pandas()
            return df_collected
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def _is_null_value(self, val: Any) -> bool:
        """Check if value should be considered as null/none"""
        if val is None:
            return True
        if isinstance(val, float) and np.isnan(val):
            return True
        if isinstance(val, str) and val.strip().lower() in ('', 'null', 'na', 'n/a', 'none', 'nan'):
            return True
        return False

    def _is_potential_false_null(self, val: Any) -> Tuple[bool, str]:
        """Check if value could be a false null with type classification"""
        if isinstance(val, (int, float)):
            if val in self.potential_false_nulls['numeric']:
                return True, f"numeric_sentinel:{val}"
            if np.isinf(val):
                return True, f"numeric_infinity:{val}"
                
        if isinstance(val, str):
            val_stripped = val.strip()
            # Check exact matches
            if val_stripped in self.potential_false_nulls['strings']:
                return True, f"string_code:{val_stripped}"
            # Check whitespace variants
            if not val_stripped and val in self.potential_false_nulls['whitespace']:
                return True, f"whitespace_null:{repr(val)}"
            # Check temporal patterns
            if any(d in val for d in self.potential_false_nulls['temporal']):
                return True, f"temporal_sentinel:{val}"
                
        return False, ""

    def _detect_potential_false_nulls(self, series: pl.Series) -> Dict[str, int]:
        """Detect values that might be false nulls"""
        sample_size = min(self.sample_size, len(series))
        sample = series.sample(n=sample_size, seed=42)
        
        found_types = {}
        
        for val in sample:
            is_null, null_type = self._is_potential_false_null(val)
            if is_null:
                found_types[null_type] = found_types.get(null_type, 0) + 1
                
        return found_types

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
        """Check if value could be a date/datetime"""
        if isinstance(val, datetime):
            return True
        if not isinstance(val, str):
            return False
        
        val = str(val).strip()
        if not val:
            return False
        
        if any(p.fullmatch(val) for p in self.date_patterns):
            try:
                parsed = parse(val, fuzzy=False)
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

    def _detect_column_types(self, series: pl.Series) -> List[str]:
        """Detect data types present in the column"""
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
        return [t for t in detected_types if t != 'null'] or ['untyped']

    def _detect_outliers(self, series: pl.Series) -> str:
        """Detect outliers for numeric columns using IQR method"""
        if series.dtype not in (pl.Int64, pl.Float64):
            return "N/A"
            
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
                percentage = (outlier_count/len(series))*100
                return f"{outlier_count} ({percentage:.1f}%)"
            return "None"
        except Exception:
            return "Error"

    def calculate_null_stats(self) -> Dict[str, Tuple[float, int]]:
        """Calculate exact percentage of null values and non-null counts for each column"""
        if self.df is None:
            self.df = self._load_data()
        
        null_stats = {}
        total_rows = len(self.df)
        
        for col in self.df.columns:
            col_data = self.df[col]
            
            # First count standard nulls
            null_count = col_data.null_count()
            
            # Then check for other null-like values if column is string type
            if col_data.dtype == pl.Utf8:
                sample_size = min(self.sample_size, len(col_data))
                sample = col_data.sample(n=sample_size, seed=42)
                additional_nulls = sum(self._is_null_value(x) for x in sample if not x is None)
                
                # Scale up the additional nulls count to full dataset size
                if sample_size > 0:
                    null_count += additional_nulls * (total_rows / sample_size)
            
            percentage = (null_count / total_rows) * 100 if total_rows > 0 else 0
            non_null_count = int(total_rows - null_count)
            null_stats[col] = (percentage, non_null_count)
        
        return null_stats

    def calculate_duplicate_stats(self) -> Dict[str, Tuple[float, int, int]]:
        """Calculate percentage, count of duplicate values, and unique count for each column"""
        if self.df is None:
            self.df = self._load_data()
        
        duplicate_stats = {}
        total_rows = len(self.df)
        
        for col in self.df.columns:
            col_data = self.df[col]
            
            try:
                # Get non-null values only for duplicate calculation
                non_null_data = col_data.filter(col_data.is_not_null())
                non_null_count = len(non_null_data)
                
                if non_null_count == 0:
                    duplicate_stats[col] = (0.0, 0, 0)
                    continue
                
                # Get value counts
                value_counts = non_null_data.value_counts()
                count_col = 'count' if 'count' in value_counts.columns else 'counts'
                
                # Calculate unique values (distinct count)
                unique_count = len(value_counts)
                
                # Calculate duplicate count (total non-null values minus unique values)
                duplicate_count = non_null_count - unique_count
                
                # Calculate duplicate percentage
                duplicate_percentage = (duplicate_count / total_rows) * 100
                
                duplicate_stats[col] = (duplicate_percentage, duplicate_count, unique_count)
            except Exception as e:
                logger.warning(f"Could not calculate duplicates for column {col}: {str(e)}")
                duplicate_stats[col] = (0.0, 0, 0)
        
        return duplicate_stats

    def calculate_special_char_stats(self) -> Dict[str, Tuple[int, List[str]]]:
        """Calculate special character statistics for each column"""
        if self.df is None:
            self.df = self._load_data()
        
        special_char_stats = {}
        special_char_pattern = r'[^\w\s\.-]'  # Regex for special chars
        
        for col in self.df.columns:
            col_data = self.df[col]
            col_type = col_data.dtype
            
            # Skip non-string columns or columns that appear to be numeric
            if col_type != pl.Utf8 or col_data.str.contains(r'^[0-9\.]+$').any():
                special_char_stats[col] = (0, [])
                continue
            
            # Sample data for performance
            sample_size = min(self.sample_size, len(col_data))
            sample = col_data.sample(n=sample_size, seed=42)
            
            # Find special characters
            special_values = [str(x) for x in sample 
                            if isinstance(x, str) and re.search(special_char_pattern, x)]
            
            # Scale up to estimate full dataset count
            full_count = int(len(special_values) * (len(col_data) / sample_size)) if sample_size > 0 else 0
            sample_values = special_values[:3] if len(special_values) > 0 else []
            
            special_char_stats[col] = (full_count, sample_values)
        
        return special_char_stats

    def detect_mixed_types(self) -> Dict[str, Dict[str, int]]:
        """Detect columns with mixed data types"""
        if self.df_pd is None:
            self._load_data()
        
        type_summary = {}
        
        for col in self.df_pd.columns:
            non_null_values = self.df_pd[col].dropna()
            type_counts = non_null_values.apply(lambda x: type(x).__name__).value_counts().to_dict()
            type_summary[col] = type_counts
        
        return type_summary

    def get_mixed_type_columns(self) -> List[str]:
        """Get list of columns with mixed data types"""
        type_summary = self.detect_mixed_types()
        mixed_columns = []
        
        for col, type_counts in type_summary.items():
            if len(type_counts) > 1:
                mixed_columns.append(col)
        
        return mixed_columns

    def generate_profile_report(self) -> pl.DataFrame:
        """Generate a comprehensive data profile report"""
        null_stats = self.calculate_null_stats()
        duplicate_stats = self.calculate_duplicate_stats()
        special_char_stats = self.calculate_special_char_stats()
        mixed_type_info = self.detect_mixed_types()
        mixed_type_columns = self.get_mixed_type_columns()
        
        # Identify columns with >80% nulls
        high_null_cols = [col for col, (pct, _) in null_stats.items() if pct >= 80]
        
        report_data = []
        for col in self.df.columns:
            col_data = self.df[col]
            
            # Get null stats
            null_pct, non_null_count = null_stats.get(col, (0.0, 0))
            
            # Detect potential false nulls
            false_nulls = self._detect_potential_false_nulls(col_data)
            false_null_desc = ", ".join(
                f"{k}({v})" for k, v in false_nulls.items()
            ) if false_nulls else "None"
            
            # Sample values of false nulls (up to 3 examples)
            sample_false_nulls = []
            for val in col_data.sample(min(100, len(col_data)), seed=42):
                is_null, _ = self._is_potential_false_null(val)
                if is_null and val not in sample_false_nulls:
                    sample_false_nulls.append(str(val))
                    if len(sample_false_nulls) >= 3:
                        break
            
            dup_pct, dup_count, unique_count = duplicate_stats.get(col, (0.0, 0, 0))
            spec_char_count, spec_char_samples = special_char_stats.get(col, (0, []))
            outliers = self._detect_outliers(col_data)
            
            # Format type information
            type_counts = mixed_type_info.get(col, {})
            type_info = ", ".join([f"{k}:{v}" for k, v in type_counts.items()]) if type_counts else "Consistent"
            is_mixed = "Yes" if col in mixed_type_columns else "No"
            
            report_data.append({
                "Column Name": col,
                "Current Type": str(col_data.dtype),
                "Null Percentage (%)": null_pct,
                "High Null Column": "Yes" if null_pct >= 80 else "No",
                "Non-Null Count": non_null_count,
                "Duplicate Percentage (%)": dup_pct,
                "Duplicate Count": dup_count,
                "Unique Count": unique_count,
                "Special Char Count": spec_char_count,
                "Special Char Samples": str(spec_char_samples) if spec_char_samples else "None",
                "Potential False Nulls": false_null_desc,
                "False Null Examples": " | ".join(sample_false_nulls) if sample_false_nulls else "None",
                "Mixed Types?": is_mixed,
                "Outliers": outliers
            })
        
        # Create summary of high null columns
        if high_null_cols:
            logger.warning(f"Columns with 80%+ null values: {', '.join(high_null_cols)}")
        
        return pl.DataFrame(report_data).sort("Null Percentage (%)", descending=True)
    def save_report(self, report: pl.DataFrame, output_format: str = "csv"):
        """Save report to file with sample dataframe"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"data_profile_report_{timestamp}"
        
        # Save the report
        if output_format.lower() == "csv":
            report_path = f"{base_filename}.csv"
            report.write_csv(report_path)
        else:
            report_path = f"{base_filename}.parquet"
            report.write_parquet(report_path)
        
        # Save the sample dataframe
        if self.df is not None:
            sample_path = f"{base_filename}_sample.parquet"
            self.df.sample(min(1000, len(self.df))).write_parquet(sample_path)
            logger.info(f"Sample dataframe saved as {sample_path}")
        
        logger.info(f"Data profile report saved as {report_path}")
        

