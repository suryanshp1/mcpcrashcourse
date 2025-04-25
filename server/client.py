import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from mcp_use import MCPAgent, MCPClient
import os

load_dotenv()

config_file = "server/weather.json"

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

async def run_memory_chat():
    """Run a memory chat using MCPAgent built-in conversation memory."""
    print("Starting chat...")

    client = MCPClient.from_config_file(config_file)
    llm = ChatGroq(model="qwen-qwq-32b")

    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=15,
        memory_enabled=True
    )

    print("\n===============Interactive MCP Chat===============\n")
    print("Enter 'exit' or 'quit' to end the chat.\n")
    print("Type 'clear' to clear the conversation history.\n")
    print("==================================================\n")

    try:
        while True:
            user_input = input("You: ")

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat...")
                break

            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared.")
                continue
            
            # Get response from agent
            print("\nAssistant: ", end="", flush=True)

            try:
                print("here in try------------------------")
                response = await agent.run(user_input)

                print("resp==================================")
                print(dir(response))
                print(response)
            except Exception as e:
                print(f"\nAn error occurred: {e}")
    except KeyboardInterrupt:
        print("\nChat ended by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("Chat ended.")
        # cleanup
        if client and client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_memory_chat())