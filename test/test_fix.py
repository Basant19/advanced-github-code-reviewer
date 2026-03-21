import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
# Try the most stable 2.5 version first
def test_model(model_name):
    print(f"--- Testing {model_name} ---")
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_retries=0,  # Set to 0 so we see the immediate result
            api_key=os.environ.get("GOOGLE_API_KEY")
        )
        response = llm.invoke("Say 'System Online'")
        print(f"Success: {response.content}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

if __name__ == "__main__":
    # Test in order of likely availability
    models_to_try = [ 
        
        "gemini-2.5-flash-lite"
    ]
    
    for m in models_to_try:
        if test_model(m):
            print(f"\n✅ Use this model in your .env: {m}")
            break