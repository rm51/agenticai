import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
import argparse

print(">>> RUNNING VERSION CHECK: deterministic fulfillment v3.1 (locked order lines) <<<")

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################
#
# Design overview (v2 - deterministic fulfillment)
# ------------------------------------------------
# The LLM agents decide WHAT the customer asked for (which catalog item each
# line maps to, how many units). Everything that has to be *correct* - stock
# checks, delivery-date vs. deadline comparison, restock decisions, pricing,
# discounts, recording transactions, and the final customer-facing response -
# is done by plain Python inside the tools below. This guarantees that:
#   * each line of the final response matches what was actually recorded
#     (SOLD or DECLINED), and the order is never confirmed as a whole when
#     only part of it went through;
#   * every delivery explanation is computed from real dates and the
#     supplier lead time, never written by the LLM;
#   * every sold line shows its customer-facing price and the order shows a
#     total that sums only the lines actually sold;
#   * no internal data (cash balance, supplier cost, transaction IDs) is
#     ever shown to the customer.

import re
import json
import difflib

# Set up and load your env parameters and instantiate your model.

from smolagents import CodeAgent, ToolCallingAgent, tool, OpenAIServerModel

dotenv.load_dotenv()

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base=os.getenv("OPENAI_API_BASE", "https://openai.vocareum.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ---------------------------------------------------------------------------
# Shared catalog data and per-request state
# ---------------------------------------------------------------------------

_CATALOG_ITEM_NAMES = [item["item_name"] for item in paper_supplies]
_CATALOG_UNIT_PRICES = {item["item_name"]: item["unit_price"] for item in paper_supplies}
_CATALOG_CATEGORIES = {item["item_name"]: item["category"] for item in paper_supplies}

SHEETS_PER_REAM = 500

# Per-request context, reset by call_your_multi_agent_system() for every request.
_REQUEST_CONTEXT = {"request_date": None, "deadline": None}
# One entry per processed order line, in the order processed. This - not the
# LLM's own text - is the source of truth for the customer response.
_LINE_RESULTS = []
# Guards against the same line being recorded twice if an agent retries.
_PROCESSED_LINE_KEYS = {}


def _to_iso_date(value) -> Union[str, None]:
    """Normalise a date-like value to 'YYYY-MM-DD', or return None if impossible."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.upper() in {"NONE", "NULL", "N/A", "UNKNOWN"}:
        return None
    s = s.split("T")[0]
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _extract_request_date(request_text: str) -> Union[str, None]:
    """Pull the request date out of '(Date of request: YYYY-MM-DD)'."""
    m = re.search(r"Date of request:\s*(\d{4}-\d{2}-\d{2})", request_text)
    return m.group(1) if m else None


_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)


def _extract_deadline(request_text: str) -> Union[str, None]:
    """
    Deterministically find the customer's delivery deadline in the request
    text (e.g. 'delivered by April 15, 2025'). Returns ISO date or None.
    The '(Date of request: ...)' suffix is removed first so it is never
    mistaken for the deadline.
    """
    text_wo_req = re.sub(r"\(Date of request:[^)]*\)", "", request_text)
    # Prefer a date that follows "by" / "before" / "no later than" (the deadline).
    m = re.search(rf"\b(?:by|before|no later than|until)\s+(?:\w+day,?\s+)?({_MONTHS})\.?\s+"
                  rf"(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})", text_wo_req, re.IGNORECASE)
    if m:
        return _to_iso_date(f"{m.group(1)[:3].title()} {int(m.group(2))}, {m.group(3)}")
    m = re.search(rf"\b({_MONTHS})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})", text_wo_req)
    if m:
        month = m.group(1)[:3]
        return _to_iso_date(f"{month} {int(m.group(2))}, {m.group(3)}")
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text_wo_req)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b", text_wo_req)
    if m:
        return _to_iso_date(m.group(1))
    return None


def _match_catalog_item(name: str) -> Union[str, None]:
    """
    Strictly validate a catalog item name chosen by an agent. Accepts an exact
    (case-insensitive) catalog name, or a near-identical spelling. Returns None
    for 'NONE' or anything that is not clearly a catalog item - we would rather
    decline a line than sell the customer the wrong product.
    """
    if not name:
        return None
    cleaned = str(name).strip().strip("'\"")
    if cleaned.upper() == "NONE":
        return None
    lowered = {n.lower(): n for n in _CATALOG_ITEM_NAMES}
    if cleaned.lower() in lowered:
        return lowered[cleaned.lower()]
    close = difflib.get_close_matches(cleaned.lower(), list(lowered.keys()), n=1, cutoff=0.85)
    return lowered[close[0]] if close else None


# Ordered keyword -> catalog item rules, used only when the agent said "NONE".
_ALIASES = [
    ("envelope", "Envelopes"),
    ("napkin", "Paper napkins"),
    ("paper cup", "Paper cups"),
    ("paper plate", "Paper plates"),
    ("flyer", "Flyers"),
    ("streamer", "Party streamers"),
    ("washi", "Decorative adhesive tape (washi tape)"),
    ("name tag", "Name tags with lanyards"),
    ("folder", "Presentation folders"),
    ("notepad", "Notepads"),
    ("sticky note", "Sticky notes"),
    ("invitation", "Invitation cards"),
    ("table cover", "Table covers"),
    ("poster board", "Large poster paper (24x36 inches)"),
    ("poster", "Poster paper"),
    ("cardstock", "Cardstock"),
    ("construction paper", "Construction paper"),
    ("glossy", "Glossy paper"),
    ("matte", "Matte paper"),
    ("recycled", "Recycled paper"),
    ("kraft", "Kraft paper"),
    ("a4", "A4 paper"),
    ("letter-size", "Letter-sized paper"),
    ("letter size", "Letter-sized paper"),
    ("printer paper", "Standard copy paper"),
    ("printing paper", "Standard copy paper"),
    ("copy paper", "Standard copy paper"),
    ("colored paper", "Colored paper"),
    ("colorful paper", "Colored paper"),
]


def _alias_item(requested_as: str) -> Union[str, None]:
    """Map the customer's wording to a catalog item when the agent gave up."""
    text = str(requested_as or "").lower()
    if not text:
        return None
    # Finished products first, so "kraft paper envelopes" is Envelopes, not Kraft paper.
    for keyword, name in _ALIASES[:13]:
        if re.search(rf"\b{re.escape(keyword)}", text):
            return name
    for name in sorted(_CATALOG_ITEM_NAMES, key=len, reverse=True):
        if name.lower() in text:
            return name
    for keyword, name in _ALIASES[13:]:
        if re.search(rf"\b{re.escape(keyword)}", text):
            return name
    return None


def _discount_for(quantity: int):
    """Bulk-discount policy. Returns (fraction, customer-facing rationale)."""
    if quantity >= 5000:
        return 0.15, "15% bulk discount applied (orders of 5,000+ units)"
    if quantity >= 1000:
        return 0.10, "10% bulk discount applied (orders of 1,000+ units)"
    if quantity >= 500:
        return 0.05, "5% bulk discount applied (orders of 500+ units)"
    return 0.0, "standard catalog price (bulk discounts start at 500 units)"


def _price_line(catalog_item: str, quantity: int) -> dict:
    """Customer-facing price for one line, with the discount rationale."""
    unit_price = _CATALOG_UNIT_PRICES[catalog_item]
    pct, why = _discount_for(quantity)
    list_total = round(quantity * unit_price, 2)
    total = round(list_total * (1 - pct), 2)
    return {
        "unit_price": unit_price,
        "list_total": list_total,
        "discount_pct": pct,
        "pricing_note": why,
        "total_price": total,
    }


def _unit_label(catalog_item: str, quantity: int) -> str:
    if _CATALOG_CATEGORIES.get(catalog_item) in ("paper", "specialty"):
        return "sheet" if quantity == 1 else "sheets"
    return "unit" if quantity == 1 else "units"


def _stock_on_hand(catalog_item: str, as_of_date: str) -> int:
    df = get_stock_level(catalog_item, as_of_date)
    value = df["current_stock"].iloc[0] if not df.empty else 0
    try:
        return int(value) if value is not None and not pd.isna(value) else 0
    except (TypeError, ValueError):
        return 0


def _fulfill_one_line(requested_as: str, catalog_item: str, quantity, unit: str,
                      request_date: str, deadline: Union[str, None]) -> dict:
    """
    Core deterministic fulfillment for a single order line. Records the
    restock (if any) and the sale in the database and returns the outcome.
    Never raises for business reasons - those become DECLINED results.
    """
    requested_as = (str(requested_as or "").strip() or str(catalog_item or "item"))

    # --- quantity ---------------------------------------------------------
    try:
        qty = int(float(str(quantity).replace(",", "")))
    except (TypeError, ValueError):
        qty = 0
    if qty <= 0:
        return {"requested_as": requested_as, "status": "DECLINED",
                "reason": "We could not determine a valid quantity for this item."}

    # --- catalog match ----------------------------------------------------
    # The agent's exact choice wins; if it said NONE, fall back to a fixed
    # keyword map so obvious products we DO carry are never declined.
    item = _match_catalog_item(catalog_item) or _alias_item(requested_as)
    if item is None:
        return {"requested_as": requested_as, "status": "DECLINED",
                "reason": "This is not a product we carry, so we cannot supply it."}

    # Reams -> sheets for sheet-priced paper (1 ream = 500 sheets).
    if str(unit or "").strip().lower() in ("ream", "reams") and \
            _CATALOG_CATEGORIES.get(item) in ("paper", "specialty"):
        qty = qty * SHEETS_PER_REAM

    # --- idempotency: same line twice in one request is recorded once ----
    key = (item, qty)
    if key in _PROCESSED_LINE_KEYS:
        return dict(_PROCESSED_LINE_KEYS[key], duplicate=True)

    label = _unit_label(item, qty)

    # --- stock and delivery timeline -------------------------------------
    in_stock = _stock_on_hand(item, request_date)
    shortfall = max(0, qty - in_stock)
    if shortfall > 0:
        ready_by = get_supplier_delivery_date(request_date, shortfall)
    else:
        ready_by = request_date

    if deadline and ready_by > deadline:  # ISO strings compare correctly
        if in_stock > 0:
            reason = (
                f"We have {in_stock:,} {_unit_label(item, in_stock)} in stock; restocking the "
                f"remaining {shortfall:,} would arrive on {ready_by}, which is after your "
                f"{deadline} deadline."
            )
        else:
            reason = (
                f"This item is currently out of stock; a supplier restock of {shortfall:,} "
                f"{label} would arrive on {ready_by}, which is after your {deadline} deadline."
            )
        result = {"requested_as": requested_as, "catalog_item": item, "quantity": qty,
                  "status": "DECLINED", "reason": reason}
        _PROCESSED_LINE_KEYS[key] = result
        return result

    # --- restock if needed (only after the timeline check passed) --------
    if shortfall > 0:
        restock_cost = round(shortfall * _CATALOG_UNIT_PRICES[item], 2)
        if get_cash_balance(request_date) < restock_cost:
            # Internal reason (cash) is deliberately not shown to the customer.
            result = {"requested_as": requested_as, "catalog_item": item, "quantity": qty,
                      "status": "DECLINED",
                      "reason": (f"We have {in_stock:,} {_unit_label(item, in_stock)} in stock and "
                                 f"are unable to source the remaining {shortfall:,} at this time.")}
            _PROCESSED_LINE_KEYS[key] = result
            return result
        create_transaction(item, "stock_orders", shortfall, restock_cost, request_date)

    # --- sale ---------------------------------------------------------------
    price = _price_line(item, qty)
    create_transaction(item, "sales", qty, price["total_price"], request_date)

    if shortfall == 0:
        delivery_note = f"Shipped from stock; ready on {ready_by}"
    elif in_stock > 0:
        delivery_note = (f"{in_stock:,} shipped from stock and {shortfall:,} from a supplier "
                         f"restock; complete order ready on {ready_by}")
    else:
        delivery_note = f"Supplied via restock; ready on {ready_by}"
    if deadline:
        delivery_note += f", on or before your {deadline} deadline."
    else:
        delivery_note += "."

    result = {
        "requested_as": requested_as,
        "catalog_item": item,
        "quantity": qty,
        "unit_label": label,
        "status": "SOLD",
        "unit_price": price["unit_price"],
        "list_total": price["list_total"],
        "discount_pct": price["discount_pct"],
        "pricing_note": price["pricing_note"],
        "total_price": price["total_price"],
        "ready_by": ready_by,
        "delivery_note": delivery_note,
    }
    _PROCESSED_LINE_KEYS[key] = result
    return result


def _line_name(r: dict) -> str:
    """'Customer wording (catalog item)', or just the catalog item if they match."""
    asked, item = r.get("requested_as", ""), r.get("catalog_item", "")
    if not item or asked.strip().lower() == item.lower():
        return item or asked
    return f"{asked} (our {item})"


def render_response() -> str:
    """
    Build the customer-facing response purely from what was actually
    recorded in _LINE_RESULTS. No LLM text is used here.
    """
    if not _LINE_RESULTS:
        return ("Thank you for your request. We were unable to identify any items we could "
                "process from it, so no order has been placed and you have not been charged. "
                "Please reply with the specific items and quantities you need.")

    sold = [r for r in _LINE_RESULTS if r["status"] == "SOLD"]
    declined = [r for r in _LINE_RESULTS if r["status"] != "SOLD"]

    if sold and not declined:
        header = "Thank you for your order. All requested items have been sold and confirmed."
    elif sold and declined:
        header = (f"Thank you for your order. Your order was PARTIALLY fulfilled: "
                  f"{len(sold)} of {len(_LINE_RESULTS)} items were sold; the rest could not be "
                  f"fulfilled and you have not been charged for them.")
    else:
        header = ("Thank you for your request. Unfortunately we were unable to fulfill this "
                  "order, and you have not been charged.")

    lines = [header, ""]
    if sold:
        lines.append("SOLD:")
        for r in sold:
            lines.append(
                f"- {_line_name(r)}: {r['quantity']:,} {r['unit_label']} "
                f"at ${r['unit_price']:.2f} each = ${r['list_total']:,.2f}; "
                f"{r['pricing_note']}. Price charged: ${r['total_price']:,.2f}. "
                f"{r['delivery_note']}"
            )
        lines.append("")
    if declined:
        lines.append("NOT FULFILLED:")
        for r in declined:
            lines.append(f"- {r['requested_as']}: {r['reason']}")
        lines.append("")

    total = round(sum(r["total_price"] for r in sold), 2)
    if sold:
        lines.append(f"Total charged: ${total:,.2f} (sold items only).")
        latest = max(r["ready_by"] for r in sold)
        lines.append(f"All sold items will be ready by {latest}.")
    else:
        lines.append("Total charged: $0.00.")
    if any("deadline" in r.get("reason", "") or "source" in r.get("reason", "") for r in declined):
        lines.append("If you can extend your deadline or adjust quantities for the unfulfilled "
                     "items, just let us know and we will be happy to quote them again.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Locked order lines
# ---------------------------------------------------------------------------
# The order lines are registered ONCE, straight from the customer's wording,
# before any agent looks at stock or prices. After that they are locked, so
# no agent can shrink a quantity to "what is in stock", drop a line, or add
# a new one. Every registered line is guaranteed an outcome (SOLD/DECLINED).

_ORDER_LINES = []                         # the locked lines for this request
_ORDER_STATE = {"locked": False, "attempts": 0, "fulfilled": False}
_MAX_REGISTER_ATTEMPTS = 3

# Numbers that describe a product rather than a quantity (e.g. "250 gsm",
# "100 lb", "24x36", "8.5x11", "100%", "36-inch", "A4").
_NON_QTY_AFTER = re.compile(r'^\s*(?:%|gsm\b|lb\b|lbs\b|x\s*\d|"|\'|-?\s*inch|in\b|cm\b|mm\b|pt\b)', re.I)


def _request_quantities(request_text: str) -> list:
    """
    Numbers in the request that look like order quantities. Used only to
    sanity-check the registered lines, never to create them.
    """
    text_wo_dates = re.sub(r"\(Date of request:[^)]*\)", "", request_text)
    text_wo_dates = re.sub(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}", " ",
                           text_wo_dates, flags=re.I)
    text_wo_dates = re.sub(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", " ", text_wo_dates)
    qtys = []
    for m in re.finditer(r"(?<![A-Za-z\d.])(\d{1,3}(?:,\d{3})+|\d+)(?![\d.,]*\d)", text_wo_dates):
        before = text_wo_dates[max(0, m.start() - 1):m.start()]
        after = text_wo_dates[m.end():m.end() + 8]
        if before in ("x", "X") or _NON_QTY_AFTER.match(after):
            continue
        value = int(m.group(1).replace(",", ""))
        if value >= 2:
            qtys.append(value)
    return qtys


def _register_lines(lines) -> dict:
    """Validate and lock the order lines. Returns {'ok': bool, ...}."""
    if _ORDER_STATE["locked"]:
        return {"ok": True, "message": "Order lines are already registered and locked.",
                "lines": list(_ORDER_LINES)}
    _ORDER_STATE["attempts"] += 1
    if isinstance(lines, str):
        try:
            lines = json.loads(lines)
        except json.JSONDecodeError:
            try:
                lines = ast.literal_eval(lines)
            except (ValueError, SyntaxError):
                return {"ok": False, "error": "lines must be a list of dicts."}
    if isinstance(lines, dict):
        lines = [lines]
    clean = []
    for line in lines or []:
        if not isinstance(line, dict):
            continue
        try:
            qty = int(float(str(line.get("quantity", 0)).replace(",", "")))
        except (TypeError, ValueError):
            qty = 0
        clean.append({
            "requested_as": str(line.get("requested_as", "")).strip(),
            "catalog_item": str(line.get("catalog_item", "NONE")).strip(),
            "quantity": qty,
            "unit": str(line.get("unit", "units")).strip(),
        })

    problems = []
    if not clean:
        problems.append("No lines were given. Register one line per requested item.")
    stated = _request_quantities(_REQUEST_CONTEXT.get("request_text", ""))
    used = [l["quantity"] for l in clean]
    for l in clean:
        if l["quantity"] <= 0:
            problems.append(f"'{l['requested_as']}' has no quantity; use the number the customer wrote.")
        elif stated and l["quantity"] not in stated:
            problems.append(
                f"'{l['requested_as']}' has quantity {l['quantity']}, which the customer never wrote. "
                f"Use the customer's own number (do NOT reduce it to what is in stock).")
    from collections import Counter
    remaining = Counter(used)
    missing = []
    for q in stated:              # multiset: two "100 sheets" lines need two lines
        if remaining[q] > 0:
            remaining[q] -= 1
        else:
            missing.append(q)
    if missing:
        problems.append(
            "These quantities from the request are not covered by any line: "
            f"{missing}. Add a line for every requested item, using catalog_item \"NONE\" "
            "if we do not carry it. Do not drop items.")
    seen = set()
    for l in clean:
        k = (l["requested_as"].lower(), l["quantity"])
        if k in seen:
            problems.append(f"'{l['requested_as']}' appears twice; list each item once.")
        seen.add(k)

    if problems and _ORDER_STATE["attempts"] < _MAX_REGISTER_ATTEMPTS:
        return {"ok": False, "errors": problems,
                "hint": "Fix these and call register_order_lines again with the full list."}

    # Accept (either valid, or out of retries - keep only usable lines).
    _ORDER_LINES.clear()
    _ORDER_LINES.extend(clean)
    _ORDER_STATE["locked"] = True
    return {"ok": True, "message": f"{len(clean)} order lines registered and locked.",
            "lines": list(_ORDER_LINES)}


def _run_fulfillment() -> list:
    """Process every locked line exactly once."""
    if _ORDER_STATE["fulfilled"]:
        return list(_LINE_RESULTS)
    for line in _ORDER_LINES:
        result = _fulfill_one_line(
            requested_as=line["requested_as"],
            catalog_item=line["catalog_item"],
            quantity=line["quantity"],
            unit=line["unit"],
            request_date=_REQUEST_CONTEXT["request_date"],
            deadline=_REQUEST_CONTEXT["deadline"],
        )
        if not result.get("duplicate"):
            _LINE_RESULTS.append(result)
    _ORDER_STATE["fulfilled"] = True
    return list(_LINE_RESULTS)


def _ctx_date(as_of_date: str) -> str:
    """Always use the real request date, whatever date an agent passes in."""
    return _REQUEST_CONTEXT.get("request_date") or _to_iso_date(as_of_date) or as_of_date


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# ---------------------------------------------------------------------------
# Tool for the orchestrator (intake)
# ---------------------------------------------------------------------------

@tool
def register_order_lines(lines: list) -> dict:
    """
    Register the customer's order lines exactly as the customer stated them.
    Call this FIRST, before asking any other agent anything. Once accepted, the
    lines are locked for the rest of the request. If it returns ok=False, fix
    the listed errors and call it again.

    Args:
        lines: One dict per requested item, with keys 'requested_as' (customer's
            wording), 'catalog_item' (EXACT catalog name, or "NONE" if we do not
            carry it), 'quantity' (the number the customer wrote) and 'unit'
            ("sheets", "units" or "reams").

    Returns:
        {'ok': True, 'lines': [...]} when locked, or {'ok': False, 'errors': [...]}.
    """
    return _register_lines(lines)


# ---------------------------------------------------------------------------
# Tools for inventory agent
# ---------------------------------------------------------------------------

@tool
def check_inventory(as_of_date: str) -> dict:
    """
    Get the full current inventory snapshot (every item with positive stock) as of a date.

    Args:
        as_of_date: ISO date string (YYYY-MM-DD) to check stock as of.

    Returns:
        A dict mapping item_name -> current stock quantity.
    """
    return get_all_inventory(_ctx_date(as_of_date))


@tool
def check_item_stock(catalog_item: str, as_of_date: str) -> dict:
    """
    Get the current stock level for one catalog item. The name must be an exact
    catalog item name (see the catalog list you were given).

    Args:
        catalog_item: Exact catalog item name, e.g. "Glossy paper".
        as_of_date: ISO date string (YYYY-MM-DD) to check stock as of.

    Returns:
        A dict with 'catalog_item', 'current_stock' and 'as_of_date', or 'error'
        if the name is not a catalog item.
    """
    date = _ctx_date(as_of_date)
    item = _match_catalog_item(catalog_item)
    if item is None:
        return {"catalog_item": catalog_item, "current_stock": 0, "as_of_date": date,
                "error": f"'{catalog_item}' is not a catalog item (we do not carry it)."}
    return {"catalog_item": item, "current_stock": _stock_on_hand(item, date), "as_of_date": date}


@tool
def check_delivery_timeline(order_date: str, quantity: int) -> str:
    """
    Estimate the supplier restock delivery date for a given order quantity.

    Args:
        order_date: ISO date string (YYYY-MM-DD) the restock order would be placed.
        quantity: Number of units being ordered from the supplier.

    Returns:
        Estimated delivery date as an ISO date string (YYYY-MM-DD).
    """
    return get_supplier_delivery_date(_ctx_date(order_date), quantity)


# ---------------------------------------------------------------------------
# Tools for quoting agent
# ---------------------------------------------------------------------------

@tool
def lookup_quote_history(search_terms: list, limit: int = 5) -> list:
    """
    Search historical quotes for similar past customer requests, to ground pricing
    and discount decisions in precedent.

    Args:
        search_terms: List of keywords to match against past requests/explanations
            (e.g. item names, event type, job type).
        limit: Maximum number of past quotes to return.

    Returns:
        A list of dicts, each a past quote with original_request, total_amount,
        quote_explanation, job_type, order_size, event_type, and order_date.
    """
    return search_quote_history(search_terms, limit)


@tool
def quote_line(catalog_item: str, quantity: int) -> dict:
    """
    Compute the official customer-facing price for one line using the company's
    bulk-discount policy (5% at 500+ units, 10% at 1,000+, 15% at 5,000+). This
    is the same pricing the ordering system will charge.

    Args:
        catalog_item: Exact catalog item name.
        quantity: Number of units (sheets for paper items).

    Returns:
        A dict with unit_price, list_total, discount_pct, pricing_note and
        total_price, or 'error' if the item is not in the catalog.
    """
    item = _match_catalog_item(catalog_item)
    if item is None:
        return {"catalog_item": catalog_item, "error": "Not a catalog item."}
    try:
        qty = int(quantity)
    except (TypeError, ValueError):
        return {"catalog_item": item, "error": "Invalid quantity."}
    return dict(_price_line(item, qty), catalog_item=item, quantity=qty)


# ---------------------------------------------------------------------------
# Tools for ordering agent
# ---------------------------------------------------------------------------

@tool
def get_company_financial_report(as_of_date: str) -> dict:
    """
    Get a full financial health report (cash, inventory value, total assets, top
    sellers) as of a date. Internal use only - never share these figures with a customer.

    Args:
        as_of_date: ISO date string (YYYY-MM-DD).

    Returns:
        Dict with as_of_date, cash_balance, inventory_value, total_assets,
        inventory_summary, and top_selling_products.
    """
    return generate_financial_report(_ctx_date(as_of_date))


@tool
def fulfill_order() -> list:
    """
    Finalize the customer's registered (locked) order lines. For each line this
    tool deterministically checks stock, works out whether a supplier restock
    can arrive by the customer's deadline, checks the company can afford the
    restock, records the restock and the sale, and applies bulk-discount pricing.
    Lines that cannot be met are DECLINED with the reason. Safe to call more
    than once - lines are only ever processed once.

    Returns:
        A list with one result per line: status SOLD (with price and ready date)
        or DECLINED (with reason).
    """
    if not _ORDER_STATE["locked"]:
        return [{"status": "ERROR",
                 "reason": "No order lines registered yet; the orchestrator must call "
                           "register_order_lines first."}]
    return _run_fulfillment()


# ---------------------------------------------------------------------------
# Set up your agents and create an orchestration agent that will manage them.
# ---------------------------------------------------------------------------

_CATALOG_TEXT = "\n".join(f"- {n}" for n in _CATALOG_ITEM_NAMES)

inventory_agent = ToolCallingAgent(
    name="inventory_agent",
    description=(
        "Reports current stock levels for exact catalog items and how long a supplier "
        "restock would take. Give it the exact catalog item names, quantities and the "
        "request date."
    ),
    tools=[check_inventory, check_item_stock, check_delivery_timeline],
    model=model,
)

quoting_agent = ToolCallingAgent(
    name="quoting_agent",
    description=(
        "Prepares a price quote. Give it exact catalog item names and quantities; it "
        "looks up similar past quotes for precedent and uses quote_line to compute the "
        "official discounted price for each line."
    ),
    tools=[lookup_quote_history, quote_line],
    model=model,
)

ordering_agent = ToolCallingAgent(
    name="ordering_agent",
    description=(
        "Finalizes the order that the orchestrator has already registered. It calls "
        "fulfill_order (no arguments), which processes every registered line - stock, "
        "deadline, cash, restock, sale and pricing - and returns SOLD/DECLINED per line."
    ),
    tools=[fulfill_order, get_company_financial_report],
    model=model,
)

orchestrator_agent = CodeAgent(
    name="orchestrator_agent",
    description=(
        "Manages the end-to-end handling of a customer paper-supply request: "
        "registers the order lines, then delegates to the inventory, quoting, and "
        "ordering agents in turn."
    ),
    tools=[register_order_lines],
    managed_agents=[inventory_agent, quoting_agent, ordering_agent],
    model=model,
)


def _reset_request_state(request_with_date: str) -> None:
    _LINE_RESULTS.clear()
    _PROCESSED_LINE_KEYS.clear()
    _ORDER_LINES.clear()
    _ORDER_STATE.update(locked=False, attempts=0, fulfilled=False)
    _REQUEST_CONTEXT["request_text"] = request_with_date
    _REQUEST_CONTEXT["request_date"] = (_extract_request_date(request_with_date)
                                        or datetime.now().strftime("%Y-%m-%d"))
    _REQUEST_CONTEXT["deadline"] = _extract_deadline(request_with_date)


def call_your_multi_agent_system(request_with_date: str) -> str:
    """
    Entry point used by run_test_scenarios(). The orchestrator registers the
    order lines (locked), then delegates to the inventory, quoting and ordering
    agents. Any registered line the agents did not finalize is finalized
    deterministically afterwards, and the customer-facing response is rendered
    from the recorded results, so it can never contradict the database.
    """
    _reset_request_state(request_with_date)
    request_date = _REQUEST_CONTEXT["request_date"]
    deadline_text = _REQUEST_CONTEXT["deadline"] or "none stated"

    task = (
        "You are the orchestrator for Munder Difflin, a paper supply company.\n\n"
        f"CUSTOMER REQUEST:\n{request_with_date}\n\n"
        f"Request date: {request_date}   (use this exact date in every message)\n"
        f"Delivery deadline: {deadline_text}\n\n"
        "CATALOG (the ONLY valid item names - use them exactly as written):\n"
        f"{_CATALOG_TEXT}\n\n"
        "STEPS:\n"
        "1. FIRST, before contacting any agent, call register_order_lines(lines=[...]) with "
        "one dict per requested item:\n"
        "   - requested_as: the customer's own wording for the item\n"
        "   - catalog_item: the single closest EXACT catalog name above, or \"NONE\" if "
        "nothing in the catalog is a reasonable match (e.g. tickets, balloons)\n"
        "   - quantity: the exact number the customer wrote (never reduce it to stock)\n"
        "   - unit: \"reams\" if they asked in reams, otherwise \"sheets\" or \"units\"\n"
        "   Include EVERY item the customer asked for, even ones we don't carry. If it "
        "returns ok=False, fix the errors and call it again.\n"
        f"2. Ask inventory_agent for the stock of the registered catalog items as of {request_date}.\n"
        "3. Ask quoting_agent for a quote on the registered lines.\n"
        "4. Ask ordering_agent to finalize the order by calling fulfill_order.\n"
        "5. Finish with final_answer(\"done\"). The customer response is generated "
        "automatically from the recorded results, so do not write one yourself."
    )
    try:
        orchestrator_agent.run(task)
    except Exception as e:  # The agent loop failing must not hide what was recorded.
        print(f"WARN: orchestrator raised {e!r}")

    # Safety net 1: the lines were never registered - ask once more, intake only.
    if not _ORDER_STATE["locked"]:
        try:
            orchestrator_agent.run(
                f"CUSTOMER REQUEST:\n{request_with_date}\n\nCATALOG:\n{_CATALOG_TEXT}\n\n"
                "Call register_order_lines with one line per requested item (requested_as, "
                "exact catalog_item or \"NONE\", quantity exactly as written, unit), fix any "
                "errors it reports, then final_answer(\"done\"). Do nothing else."
            )
        except Exception as e:
            print(f"WARN: intake retry raised {e!r}")

    # Safety net 2: every registered line gets an outcome, even if the
    # ordering agent never called fulfill_order.
    if _ORDER_STATE["locked"] and not _ORDER_STATE["fulfilled"]:
        _run_fulfillment()

    return render_response()

# Run your test scenarios by writing them here. Make sure to keep track of them.


def run_test_scenarios(only=None):
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
        quote_requests_sample = quote_requests_sample.reset_index(drop=True)
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")
        if only and (idx + 1) not in only:
            continue

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        try :
            response = call_your_multi_agent_system(request_with_date)
        except Exception as e:
            response = f"ERROR handling request {e}"

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, nargs="+", default=None,
                         help="Only run these request numbers, e.g. --only 6 7 11 1")
    args = parser.parse_args()

    results = run_test_scenarios(only=args.only)

