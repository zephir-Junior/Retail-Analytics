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

    def _suggest_column_type(self, series: pl.Series, detected_types: List[str]) -> Tuple[str, str]:
        """Suggest optimal data type with reasoning"""
        dtype = series.dtype
        current_type = str(dtype)
        reasoning = []
        
        # If already correct type
        if (dtype == pl.Int64 and 'int' in detected_types) or \
           (dtype == pl.Float64 and 'float' in detected_types) or \
           (dtype == pl.Boolean and 'bool' in detected_types) or \
           (dtype == pl.Datetime and 'datetime' in detected_types):
            return current_type, "Already optimal type"
        
        sample_size = min(1000, len(series))
        sample = series.sample(n=sample_size, seed=42).drop_nulls()
        
        if len(sample) == 0:
            return current_type, "All null values"
        
        # Enhanced datetime detection
        if dtype == pl.Utf8:
            date_samples = [x for x in sample if self._is_potential_date(x)]
            date_ratio = len(date_samples) / len(sample)
            
            if date_ratio > 0.8:  # 80% confidence threshold
                has_time = any(':' in str(x) for x in date_samples)
                has_ms = any('.' in str(x) for x in date_samples)
                has_tz = any('+' in str(x) or '-' in str(x) for x in date_samples)
                
                if has_tz:
                    return "datetime[ms, UTC]", "80%+ values match datetime with timezone format"
                elif has_ms:
                    return "datetime[ms]", "80%+ values match datetime with milliseconds format"
                elif has_time:
                    return "datetime[s]", "80%+ values match datetime format"
                else:
                    return "date", "80%+ values match date format"
        
        # Numeric detection
        if dtype == pl.Utf8:
            int_count = sum(self._is_integer(x) for x in sample)
            float_count = sum(self._is_float(x) for x in sample)
            
            if int_count / len(sample) > 0.9:
                return "int64", "90%+ values are integers"
            elif float_count / len(sample) > 0.9:
                return "float64", "90%+ values are floats"
        
        # Boolean detection
        if dtype == pl.Utf8:
            bool_count = sum(self._is_potential_bool(x) for x in sample)
            if bool_count / len(sample) > 0.9:
                return "bool", "90%+ values are boolean-like"
        
        return current_type, "No better type detected"

    def _detect_outliers(self, series: pl.Series) -> Tuple[str, str]:
        """Detect outliers for numeric columns using IQR method"""
        if series.dtype not in (pl.Int64, pl.Float64):
            return "N/A", "Non-numeric column"
            
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
                return f"{outlier_count} ({percentage:.1f}%)", f"Values outside [{lower_bound:.2f}, {upper_bound:.2f}]"
            return "None", "No outliers detected"
        except Exception:
            return "Error", "Could not calculate outliers"

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
        
        report_data = []
        for col in self.df.columns:
            col_data = self.df[col]
            detected_types = self._detect_column_types(col_data)
            current_type = str(col_data.dtype)
            suggested_type, type_reason = self._suggest_column_type(col_data, detected_types)
            outliers, outlier_reason = self._detect_outliers(col_data)
            
            null_pct, non_null_count = null_stats.get(col, (0.0, 0))
            dup_pct, dup_count, unique_count = duplicate_stats.get(col, (0.0, 0, 0))
            spec_char_count, spec_char_samples = special_char_stats.get(col, (0, []))
            
            # Format type information
            type_counts = mixed_type_info.get(col, {})
            type_info = ", ".join([f"{k}:{v}" for k, v in type_counts.items()]) if type_counts else "Consistent"
            is_mixed = "Yes" if col in mixed_type_columns else "No"
            
            report_data.append({
                "Column Name": col,
                "Current Type": current_type,
                "Suggested Type": suggested_type,
                "Type Suggestion Reason": type_reason,
                "Null Percentage (%)": null_pct,
                "Non-Null Count": non_null_count,
                "Duplicate Percentage (%)": dup_pct,
                "Duplicate Count": dup_count,
                "Unique Count": unique_count,
                "Special Char Count": spec_char_count,
                "Special Char Samples": str(spec_char_samples) if spec_char_samples else "None",
                "Mixed Types?": is_mixed,
                "Outliers": outliers,
                "Outlier Reason": outlier_reason
            })
        
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
        
        
        #********************** for maindf****************
        
        import argparse
import polars as pl 
from pathlib import Path
from data_profiler import DataProfiler
import logging
from datetime import datetime
from dateutil.parser import parse
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from pathlib import Path
import time

# Configure logging at module level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_profiler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive data profiling for Parquet files")
    parser.add_argument("file_path", help="Path to the Parquet file")
    parser.add_argument("--sample_size", type=int, default=100000, help="Sample size for analysis")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode with limited data")
    parser.add_argument("--output", choices=["csv", "parquet"], default="csv", help="Output format")
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory to save reports and samples")
    
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize profiler
        profiler = DataProfiler(
            file_path=args.file_path,
            sample_size=args.sample_size,
            test_mode=args.test_mode
        )
        
        logger.info("Starting data profiling...")
        start_time = time.time()
        
        # Generate and display report
        report = profiler.generate_profile_report()
        print("\nData Profile Report:")
        with pl.Config(
            fmt_float="full",
            tbl_width_chars=200,
            tbl_cols=14,
            tbl_rows=50
        ):
            print(report)
        
        # Save report and sample data with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"data_profile_report_{timestamp}"
        
        # Save report
        report_path = output_dir / f"{base_filename}.{args.output}"
        if args.output.lower() == "csv":
            report.write_csv(report_path)
        else:
            report.write_parquet(report_path)
        logger.info(f"Report saved to {report_path}")
        
        # Save sample data
        if profiler.df is not None:
            sample_path = output_dir / f"{base_filename}_sample.parquet"
            sample_size = min(1000, len(profiler.df))
            profiler.df.sample(n=sample_size).write_parquet(sample_path)
            logger.info(f"Sample dataset ({sample_size} rows) saved to {sample_path}")
        
        # Print mixed type summary
        mixed_columns = profiler.get_mixed_type_columns()
        if mixed_columns:
            print("\nColumns with Mixed Data Types:")
            for col in mixed_columns:
                print(f"- {col}: {profiler.detect_mixed_types()[col]}")
        
        logger.info(f"Profiling completed in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during profiling: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()


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


