from langchain.agents import create_agent
from dotenv import load_dotenv
from tools.tools import query_espboard_telemetry, discover_boards, query_all_boards_telemetry

agent = create_agent(
    model="mistralai:codestral-latest",
    tools=[query_espboard_telemetry, discover_boards, query_all_boards_telemetry],
    system_prompt=(
        'You are a telemetry data expert. Be concise.\n'
        'When asked about all boards, call query_all_boards_telemetry, which '
        'discovers the boards and fetches their telemetry in one step. Only use '
        'query_espboard_telemetry when the user gives you a specific board id. '
        'Never invent board ids. Report the telemetry per board.'
    ))


def ask(prompt: str) -> str:
    """Run the agent on a natural-language request and return its final answer.
    The agent decides when to call discover_boards / query_espboard_telemetry."""
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print(ask("Give me the telemetry for all boards."))
    # print(ask("Give me the telemetry for board 51"))