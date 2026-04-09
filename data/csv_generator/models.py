"""Pydantic models matching the exact CSV schemas for each entity type.

Field names and types match the column headers in the existing CSV files.
"""

from pydantic import BaseModel


class Customer(BaseModel):
    customer_id: str  # C0001
    first_name: str
    last_name: str
    email: str
    phone: str
    address: str
    city: str
    state: str
    zip_code: str
    registration_date: str  # YYYY-MM-DD
    date_of_birth: str  # YYYY-MM-DD
    risk_profile: str  # Aggressive, Conservative, Moderate, or ""
    employment_status: str  # Employed, Self-Employed, Retired, Unemployed
    annual_income: int
    credit_score: int
    is_fraudulent: bool = False


class Bank(BaseModel):
    bank_id: str  # B001
    name: str
    headquarters: str  # "City ST" format
    bank_type: str  # Commercial, Regional, Credit Union, Savings, Community
    total_assets_billions: float
    established_year: int
    routing_number: str  # 9-digit
    swift_code: str  # 8-char


class Account(BaseModel):
    account_id: str  # A00001
    account_number: str  # 10-digit
    customer_id: str
    bank_id: str
    account_type: str  # Checking, Savings, Investment
    balance: float
    currency: str  # USD
    opened_date: str  # YYYY-MM-DD
    status: str  # Active
    interest_rate: float


class Transaction(BaseModel):
    transaction_id: str  # T000001
    from_account_id: str
    to_account_id: str
    amount: float
    currency: str  # USD
    transaction_date: str  # YYYY-MM-DD
    transaction_time: str  # HH:MM:SS
    type: str  # Transfer, Payment, Deposit, Withdrawal
    status: str  # Completed
    description: str


class Company(BaseModel):
    company_id: str  # CO001
    name: str
    ticker_symbol: str
    industry: str
    sector: str
    market_cap_billions: float
    headquarters: str  # "City ST" format
    founded_year: int
    ceo: str
    employee_count: int
    annual_revenue_billions: float


class Stock(BaseModel):
    stock_id: str  # S001
    ticker: str
    company_id: str
    current_price: float
    previous_close: float
    opening_price: float
    day_high: float
    day_low: float
    volume: int
    market_cap_billions: float
    pe_ratio: float
    dividend_yield: float
    fifty_two_week_high: float
    fifty_two_week_low: float
    exchange: str  # NYSE, NASDAQ


class PortfolioHolding(BaseModel):
    holding_id: str  # H00001
    account_id: str
    stock_id: str
    shares: int
    purchase_price: float
    purchase_date: str  # YYYY-MM-DD
    current_value: float
    percentage_of_portfolio: float
