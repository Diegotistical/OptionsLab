# Assuming decorators are available in src/utils/decorators/
from src.utils.decorators.caching import cached_data, cached_resource
from src.utils.decorators.timing import timeit
from src.utils.utils import (
    FinancialError,
    NumericalStabilityError,
    OptionType,
    calculate_d1_d2,
    compute_moneyness,
    handle_edge_cases,
    safe_division,
    validate_inputs,
)

# Market data utilities (optional - requires yfinance)
try:
    from src.utils.market_data import (
        OptionChainData,
        OptionChainParser,
        StockData,
        YahooFinanceFetcher,
        fetch_option_chain,
        fetch_stock_history,
        load_from_parquet,
        save_to_parquet,
    )

    MARKET_DATA_AVAILABLE = True
except ImportError:
    MARKET_DATA_AVAILABLE = False

__all__ = [
    "validate_inputs",
    "safe_division",
    "calculate_d1_d2",
    "compute_moneyness",
    "handle_edge_cases",
    "OptionType",
    "FinancialError",
    "NumericalStabilityError",
    "cached_resource",
    "cached_data",
    "timeit",
    # Market data (if available)
    "MARKET_DATA_AVAILABLE",
]
