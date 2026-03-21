import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Setup API Key (or set as environment variable)
os.environ["GOOGLE_API_KEY"] = "AIzaSyA_PKObaCLSE5qiu-wKZ7alTxw_NWjvmMg"

def test_gemini_init():
    # 2. Initialize using the unified factory method
    # This automatically detects the langchain-google-genai provider
    model = init_chat_model(
        model="gemini-1.5-pro",
        model_provider="google_genai",
        temperature=0.7,
    )

    # 3. Create a simple conversation
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello! Can you confirm your model version and current date?")
    ]

    # 4. Invoke and print
    try:
        print("--- Sending Request ---")
        response = model.invoke(messages)
        print(f"Response Content:\n{response.content}")
        print(f"\nMetadata: {response.response_metadata}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_gemini_init()