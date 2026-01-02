from agent import build_graph
from ingestion import ingest_data

def main():
    print("Initializing Research Agent...")
    # --- Check / Run Data Ingestion ---
    try:
        print("Checking data availability...")
        ingest_data()
    except Exception as e:
        print(f"Warning: Data ingestion failed or skipped. Error: {e}")
        print("Continuing with existing index...")    
    app = build_graph()
    
    print("\nResearch Agent Ready! (Type 'quit' to exit)")
    
    while True:
        user_input = input("\nUser Query: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        initial_state = {
            'input': user_input,
            'chat_history': [],
            'intermediate_steps': []
        }
        
        print("\n--- Processing ---")
        # Invoke the graph
        try:
            result = app.invoke(initial_state)
            
            # Extract final answer from intermediate steps
            final_steps = result['intermediate_steps']
            final_answer_output = final_steps[-1][1] # Get observation of the last tool (final_answer)
            
            print("\n=== Final Report ===\n")
            print(final_answer_output)
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()