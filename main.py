import asyncio
from graph import graph


async def run():
    goal = input("Please enter your goal (or press Enter for default): ").strip()
    if not goal:
        goal = "What is the weather in Paris? Then multiply the temperature by 5."
    print(f"Starting agent with goal: {goal}")
    initial_state = {"goal": goal}

    # We use await graph.ainvoke for async graphs
    result = await graph.ainvoke(initial_state)

    print("\n=== FINAL OUTPUT ===")
    print(result.get("results", [])[-1].get("result", "No final result"))


if __name__ == "__main__":
    asyncio.run(run())
