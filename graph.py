import json
import re
import os
from typing import List, Dict, Any, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import asyncio
import operator

from MCP_code import get_mcp_tools

load_dotenv()
gemini_api_key = os.getenv("GEMINI_SECRET")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", api_key=gemini_api_key, temperature=0
)

PLAN_SYSTEM = """Break the user goal into an ordered JSON list of steps.
Each step MUST follow this EXACT schema:
  {"step": int, "description": str, "tool": str or null, "args": dict or null}

Available MCP tools and their EXACT argument names:
  - get_current_weather(city: str)    → get real weather for a city
  - get_weather_forecast(city: str, days: int)
  - search_web(query: str)            → Look up current information
  - search_news(query: str)           → Look up news
  - calculator(expression: str)       → math expression (e.g., '10.5 * 5')
  - add(a: float, b: float)
  - subtract(a: float, b: float)
  - multiply(a: float, b: float)
  - divide(a: float, b: float)
  - power(base: float, exponent: float)
  - square_root(number: float)

CRITICAL RULE FOR MATH TOOLS: Do NOT use variable names like "temperature" in the kwargs. ONLY use literal numbers. If you need a value from a previous step (like the temperature of a city), DO NOT CALL THE MATH TOOL IN THIS PLANNING PHASE because you do not know the actual number yet. Instead, set the tool to `null` and state the math operation in the description so the executor synthesizes it securely.

Return ONLY a valid JSON array. No markdown, no explanation."""

TOOL_ARG_MAP = {
    "get_current_weather": "city",
    "get_weather_forecast": None,  # Has 2 args, let safe_args handle raw dict
    "search_web": "query",
    "search_news": "query",
    "calculator": "expression",
    "add": None,
    "subtract": None,
    "multiply": None,
    "divide": None,
    "power": None,
    "square_root": "number",
}


class AgentState(TypedDict):
    goal: str
    plan: list
    current_step: int
    results: Annotated[list, operator.add]


def safe_args(tool_name: str, raw_args: dict) -> dict:
    expected = TOOL_ARG_MAP.get(tool_name)
    if not expected or (raw_args and expected in raw_args):
        return raw_args or {}
    if raw_args:
        first_val = next(iter(raw_args.values()), tool_name)
        print(f"  Remapped {raw_args} → {{{expected}: {first_val}}}")
        return {expected: str(first_val)}
    return {}


import sniffio


async def planner_node(state: AgentState):
    sniffio.current_async_library_cvar.set("asyncio")
    goal = state["goal"]
    print(f"Goal: {goal}")

    plan_resp = await llm.ainvoke(
        [SystemMessage(content=PLAN_SYSTEM), HumanMessage(content=goal)]
    )

    raw_text = (
        plan_resp.content
        if isinstance(plan_resp.content, str)
        else plan_resp.content[0].get("text", "")
    )
    clean_json = re.sub(r"```json|```", "", raw_text).strip()

    try:
        plan = json.loads(clean_json)
    except Exception as e:
        print("Failed to parse plan:", e)
        plan = []

    print(f"Plan ({len(plan)} steps):")
    for s in plan:
        print(f'  Step {s["step"]}: {s["description"]} | tool={s.get("tool")}')

    return {"plan": plan, "current_step": 0, "results": []}


async def executor_node(state: AgentState):
    sniffio.current_async_library_cvar.set("asyncio")
    plan = state["plan"]
    current_step_idx = state.get("current_step", 0)

    step = plan[current_step_idx]
    tool_name = step.get("tool")

    print(f'  Executing Step {step["step"]}: {step["description"]}')
    tools, tools_map = await get_mcp_tools(["math", "weather", "search"])

    if tool_name and tool_name in tools_map:
        corrected = safe_args(tool_name, step.get("args") or {})
        try:
            sniffio.current_async_library_cvar.set("asyncio")
            result = await tools_map[tool_name].ainvoke(corrected)

            # Unpack the response text if the tool returns a list of dictionaries (standard for some MCP adapters)
            if (
                isinstance(result, list)
                and isinstance(result[0], dict)
                and "text" in result[0]
            ):
                result = result[0]["text"]

        except Exception as e:
            result = str(e)
    else:
        context = "\n".join(
            [f'Step {r["step"]}: {r["result"]}' for r in state.get("results", [])]
        )
        sniffio.current_async_library_cvar.set("asyncio")
        response = await llm.ainvoke(
            [HumanMessage(content=f'{step["description"]}\n\nContext:\n{context}')]
        )
        result = response.content

    res_str = str(result)
    print(f"    {res_str[:150]}...\n")

    new_result = {
        "step": step["step"],
        "description": step["description"],
        "result": res_str,
    }
    return {"results": [new_result], "current_step": current_step_idx + 1}


def route_next_step(state: AgentState) -> str:
    if state["current_step"] >= len(state.get("plan", [])):
        return END
    return "executor_node"


builder = StateGraph(AgentState)
builder.add_node("planner_node", planner_node)
builder.add_node("executor_node", executor_node)

builder.add_edge(START, "planner_node")
builder.add_edge("planner_node", "executor_node")
builder.add_conditional_edges("executor_node", route_next_step)

graph = builder.compile()
