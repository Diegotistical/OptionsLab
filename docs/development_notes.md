# Development Notes - OptionsLab

_Personal reflections from building this project_

---

## December 2025 - January 2026 - Major Refactoring Session

### The Greeks Refactor

Started what I thought would be a simple cleanup. Ended up touching almost every pricing model.

**The Problem:** Every model had its own `delta()`, `gamma()`, `vega()` methods. At first this seemed fine - "separation of concerns," right? Wrong. I had ~500 lines of duplicate finite-difference code scattered across:

- `monte_carlo.py`
- `heston.py`
- `fdm_solver.py`
- `exotic_options.py`

**The Solution:** Created `unified_greeks.py` with a single `compute_greeks_unified()` function. Any model that implements `price(S, K, T, r, sigma, option_type, q)` now gets all Greeks for free.

**What I Learned:** The adapter pattern is beautiful. Models that don't quite fit the protocol (Heston uses `v0` not `sigma`, SABR uses forward prices) just get a thin wrapper. Zero changes to existing model code.

---

### Monte Carlo Optimization

The original MC file was 628 lines. A code review (that I did on myself) called it out:

> "This file is trying to be an engine, a lab, and a textbook at the same time."

Ouch. But fair.

**The Refactor:**

1. Extracted path generation to `src/simulation/` (numpy, numba, qmc backends)
2. Kept the pricer as a ~170 line orchestrator
3. Added a "fast mode" that uses single-step GBM

**The Result:**

- Single-step pricing: 2.65ms (was 124ms with 100 steps)
- Same accuracy for European options (math checks out)
- Antithetic variates built into all backends

**Controversial Decision:** Defaulting to `num_steps=1`. Some will say this is cheating. But for European options, multi-step simulation is just... more noise for no benefit. The terminal distribution is exact.

---

### Streamlit Pages

Created 5 new pages in one session. The tricky parts:

1. **Import Hell:** Each page needs `sys.path.insert(0, ROOT)` because Streamlit's execution model is weird. Took 20 minutes to figure out why imports failed only when running via `streamlit run` vs direct Python.

2. **Session State:** The portfolio Greeks page uses `st.session_state.positions` to track multiple positions. Getting the add/remove logic right required understanding that Streamlit reruns the entire script on every interaction.

3. **Rate Limiting:** Yahoo Finance will ban you if you hit it too hard. Added 500ms delays between requests and a 5-minute cache. Not elegant, but it works.

---

### The Live Market Integration

This was the feature I was most nervous about. Real data means real API issues.

**Gotchas I Hit:**

- `yfinance` returns `NaN` for some option strikes
- Not all tickers have options data
- The `fast_info` attribute changed in recent yfinance versions
- Options expiries come as strings, not dates

**What Made It Work:** Aggressive caching + fallbacks. If the API fails, don't crash - return empty data with an error message.

---

### Backtest Engine

Building the delta hedge simulator was fun. The interesting insight:

> If you sell an option at 20% IV and realized vol is 15%, you should make money. But day-to-day P&L is dominated by gamma, not the vol edge.

The backtest shows this beautifully. Your cumulative P&L is choppy, but over time the theta edge wins if you priced it right.

**Technical Note:** I'm using simple explicit FDM for the intra-day P&L. It's not perfectly accurate, but for a backtest that runs in seconds, it's good enough.

---

### Dupire Local Volatility (The Research Feature)

This is the feature that (I hope) sets this project apart.

Most student projects have Black-Scholes and Monte Carlo. Some have Heston. Almost none have local volatility calibration.

**Why It Matters:** Local vol is what desks actually use to:

1. Ensure model consistency with market prices
2. Price path-dependent exotics
3. Understand the "effective" volatility at each point in time/price

**Implementation Challenges:**

- Dupire's formula has divisions that can blow up
- Finite differences on a twice-differentiated surface amplify noise
- Need to handle boundaries carefully

**Current Status:** It works, but it's not production-grade. The explicit FDM pricer is slow and can be unstable. A real implementation would use implicit schemes with better boundary handling.

---

## What's Still Missing

Being honest with myself:

1. **No Greeks surface visualization** - Would be cool to show how delta/gamma change across spot/vol
2. **No position persistence** - Portfolio resets on page refresh
3. **No real calibration** - Heston calibration is stubbed out, needs proper optimization
4. **No American option Greeks** - LSM doesn't give clean Greeks

---

## Things I'd Do Differently

1. **Start with the unified interface.** Would have saved the big refactor.
2. **Use FastAPI from day one.** Streamlit is great for demos but not for production.
3. **Write tests earlier.** Added them late, found bugs that had been hiding.

---

## Final Thoughts

This project started as a way to understand options pricing. It became a lesson in software architecture.

The hard parts are:

- Keeping the codebase maintainable as it grows
- Making real data integration robust
- Presenting complex outputs in an understandable way

If you're reading this and building something similar: don't be afraid to delete code. My best improvements came from removing things, not adding them.

_- Diego, January 2026_
