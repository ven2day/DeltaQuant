"""
Agent Graph Module

LangGraph workflow orchestration for the trading agent system.
Defines the state graph and agent connections.

Enhanced: Support agents (news, sentiment, prediction) are now wired
into the pipeline as a parallel pre-processing step before regime detection.
"""

import logging
from typing import Any, Literal, cast

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.utils.serialization import to_native

from .market_regime import market_regime_node
from .news_analyst import NewsAnalyst
from .prediction import prediction_node
from .risk_compliance import check_kill_switch, risk_compliance_node
from .sentiment import sentiment_analysis_node
from .signal_validation import signal_validation_node
from .state import TradingState, create_initial_state
from .strategy_selection import strategy_selection_node

logger = logging.getLogger(__name__)


def _news_analyst_node(state: dict) -> dict:
    """Wrapper node for the class-based NewsAnalyst."""
    import asyncio

    try:
        analyst = NewsAnalyst()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # LangGraph can execute this synchronous node in a worker thread,
            # where there is no event loop to retrieve. Create one directly.
            sentiment = asyncio.run(analyst.get_market_sentiment())
        else:
            # A loop is already running in this thread, so run the coroutine in
            # a short-lived worker rather than attempting nested asyncio.run().
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                sentiment = pool.submit(asyncio.run, analyst.get_market_sentiment()).result(
                    timeout=30
                )

        # Canonical contract: news_sentiment is a dict ({"avg_sentiment": float}),
        # consumed by the sentiment, regime, and validation agents. Returning a
        # bare float here silently broke every downstream consumer.
        return {
            "news_sentiment": {"avg_sentiment": sentiment.avg_sentiment},
            "news_headlines": [
                {
                    "title": item.title,
                    "sentiment": "positive" if item.sentiment_score > 0 else "negative",
                }
                for item in sentiment.items[:5]
            ],
        }
    except Exception as e:
        logger.warning(f"News analyst failed: {e}")
        return {}


def support_agents_node(state: TradingState) -> TradingState:
    """
    Run support agents (news, sentiment, prediction) and merge results.

    Runs each support agent sequentially with graceful failure handling.
    Results are merged into the shared state for downstream agents.
    """
    logger.info("Running support agents (news, sentiment, prediction)...")

    merged_state = dict(state)

    agents = [
        ("news_analyst", _news_analyst_node),
        ("sentiment", sentiment_analysis_node),
        ("prediction", prediction_node),
    ]

    for name, node_fn in agents:
        try:
            result = node_fn(merged_state)
            if result:
                merged_state.update(result)
                logger.info(f"Support agent '{name}' completed successfully")
        except Exception as e:
            logger.warning(f"Support agent '{name}' failed (non-fatal): {e}")

    # The support agents (esp. the ML prediction agent) enrich the state with numpy scalars
    # via pandas/scikit-learn. Coerce to native Python types before the state is checkpointed
    # by MemorySaver — a stray numpy.float64 at the msgpack boundary crashes the cycle (#18).
    return cast(TradingState, to_native(merged_state))


def should_continue_after_regime(state: TradingState) -> Literal["strategy_selection", "end"]:
    """
    Decide whether to continue after regime classification.

    Returns "end" if:
    - Regime confidence is too low
    - Kill switch is triggered
    """
    if check_kill_switch(state):
        logger.warning("Kill switch triggered - ending workflow")
        return "end"

    regime_confidence = state.get("regime_confidence", 0)
    if regime_confidence < 0.3:
        logger.info(f"Regime confidence too low ({regime_confidence:.2f}) - skipping trading")
        return "end"

    return "strategy_selection"


def should_continue_after_validation(state: TradingState) -> Literal["risk_compliance", "end"]:
    """
    Decide whether to continue to risk checks.

    Returns "end" if no signals were validated.
    """
    validated_signals = state.get("validated_signals", [])

    if not validated_signals:
        logger.info("No validated signals - ending workflow")
        return "end"

    return "risk_compliance"


def create_trading_graph(
    checkpointer: Any = None,
    with_memory: bool = True,
    include_support_agents: bool = True,
) -> StateGraph:
    """
    Create the trading agent workflow graph.

    Enhanced flow:
    1. Support Agents (news, sentiment, prediction) -> enriches state
    2. Market Regime Agent -> classifies market conditions (with sentiment/news context)
    3. Strategy Selection Agent -> picks active strategies
    4. Signal Validation Agent -> filters signals (with prediction context)
    5. Risk & Compliance Agent -> final approval

    Args:
        checkpointer: Optional checkpointer for state persistence
        with_memory: Whether to include memory saver (default True)
        include_support_agents: Whether to run support agents (default True)

    Returns:
        Compiled StateGraph ready for execution
    """

    # Create the graph with TradingState
    workflow = StateGraph(TradingState)

    # Add nodes
    if include_support_agents:
        workflow.add_node("support_agents", support_agents_node)
    workflow.add_node("market_regime", market_regime_node)
    workflow.add_node("strategy_selection", strategy_selection_node)
    workflow.add_node("signal_validation", signal_validation_node)
    workflow.add_node("risk_compliance", risk_compliance_node)

    # Add edges
    if include_support_agents:
        # Start -> Support Agents -> Market Regime
        workflow.add_edge(START, "support_agents")
        workflow.add_edge("support_agents", "market_regime")
    else:
        # Start -> Market Regime (original flow)
        workflow.add_edge(START, "market_regime")

    # Market Regime -> (conditional) -> Strategy Selection or End
    workflow.add_conditional_edges(
        "market_regime",
        should_continue_after_regime,
        {
            "strategy_selection": "strategy_selection",
            "end": END,
        },
    )

    # Strategy Selection -> Signal Validation
    workflow.add_edge("strategy_selection", "signal_validation")

    # Signal Validation -> (conditional) -> Risk or End
    workflow.add_conditional_edges(
        "signal_validation",
        should_continue_after_validation,
        {
            "risk_compliance": "risk_compliance",
            "end": END,
        },
    )

    # Risk Compliance -> End
    workflow.add_edge("risk_compliance", END)

    # Compile with optional checkpointer
    if checkpointer is None and with_memory:
        checkpointer = MemorySaver()

    compiled = workflow.compile(checkpointer=checkpointer)

    agent_msg = "with support agents" if include_support_agents else "core only"
    logger.info(f"Trading graph compiled successfully ({agent_msg})")
    return compiled


async def run_trading_cycle(
    graph: StateGraph,
    market_data: dict[str, Any],
    indicators: dict[str, Any],
    signals: list[dict[str, Any]],
    memory_lessons: list[dict[str, Any]] = None,
    portfolio: dict[str, Any] = None,
    daily_stats: dict[str, Any] = None,
    thread_id: str = "default",
    precomputed_predictions: dict[str, dict[str, Any]] | None = None,
    data_namespace: str = "paper_market_data",
    trade_horizon: str = "SWING",
) -> TradingState:
    """
    Run a complete trading cycle through the agent graph.

    Args:
        graph: Compiled trading graph
        market_data: Current market data
        indicators: Calculated indicators
        signals: Raw signals from signal engine
        memory_lessons: Lessons from past trades
        portfolio: Current portfolio state
        daily_stats: Today's trading statistics
        thread_id: Thread ID for checkpointing
        precomputed_predictions: Optional ML predictions the caller already computed
            for its own ranking pass, keyed by "SYMBOL|TIMEFRAME". The support-agent
            prediction_node reuses these instead of re-fetching history and
            re-running the ensemble for symbols it already covers.
        data_namespace: Namespace of the current run's data (MOCK_NAMESPACE vs
            PAPER_DATA_NAMESPACE) -- threaded into TradingState so nodes reading
            historical performance/lesson data query the caller's actual namespace
            instead of always defaulting to paper_market_data (see H-9).
        trade_horizon: "SWING" (default, matches every pre-existing caller) or
            "SCALP" -- threaded into TradingState so strategy_selection's H-8 gate
            and risk_compliance's admission check/position sizing use the matching
            registry grain and limits (see src/backtesting/strategy_registry.py).
            One invocation handles one horizon's signals; a cycle that runs both
            calls this twice (see scripts/run_live_trading.py).

    Returns:
        Final trading state with decisions
    """

    # Create initial state with inputs. Coerce every input to native Python types: quotes,
    # indicators and P&L-derived stats can carry numpy scalars, which crash the msgpack
    # checkpoint boundary (see to_native / #18).
    state = create_initial_state(data_namespace=data_namespace, trade_horizon=trade_horizon)
    state["market_data"] = to_native(market_data)
    state["indicators"] = to_native(indicators)
    state["signals"] = to_native(
        [s.to_dict() if hasattr(s, "to_dict") else s for s in signals]
    )
    state["memory_lessons"] = to_native(memory_lessons or [])
    state["portfolio"] = to_native(portfolio or {"capital": 1000000, "positions": []})
    state["daily_stats"] = to_native(
        daily_stats or {"trades_count": 0, "profit_loss": 0, "max_drawdown": 0}
    )
    state["precomputed_predictions"] = to_native(precomputed_predictions or {})

    # Configure optional Langfuse tracing for this graph invocation.
    from src.observability.tracing import get_langfuse_callback

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "metadata": {
            "workflow_type": "trading_cycle",
            "signals_count": len(signals),
            "trade_horizon": trade_horizon,
        },
    }
    langfuse_callback = get_langfuse_callback()
    if langfuse_callback is not None:
        config["callbacks"] = [langfuse_callback]

    # Run the graph
    logger.info(f"Starting trading cycle with {len(signals)} signals")

    final_state = await graph.ainvoke(state, config=config)

    # Log results
    approved = len(final_state.get("approved_trades", []))
    rejected_signal = len(final_state.get("rejected_signals", []))
    rejected_risk = len(final_state.get("risk_rejected", []))

    logger.info(
        f"Trading cycle complete: "
        f"{approved} approved, {rejected_signal} signal-rejected, {rejected_risk} risk-rejected"
    )

    return final_state


def get_graph_visualization(graph: StateGraph) -> str:
    """
    Get a Mermaid diagram of the trading graph.

    Returns:
        Mermaid diagram string
    """
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        logger.warning(f"Could not generate graph visualization: {e}")
        return """
        graph TD
            START --> support_agents
            support_agents --> market_regime
            market_regime --> |high confidence| strategy_selection
            market_regime --> |low confidence| END
            strategy_selection --> signal_validation
            signal_validation --> |has signals| risk_compliance
            signal_validation --> |no signals| END
            risk_compliance --> END
        """
