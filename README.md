#fast api ui 
python run.py
http://localhost:8000/docs

to test api endpoints of fastapi docs :
To verify that your entire platform is working correctly—from database connections to the LangGraph execution flow and the Human-in-the-Loop (HITL) gate—you should test the endpoints sequentially following your data flow.

Here is the correct order to test your endpoints:

Step 1: Smoke Test & Setup (Pre-requisites)
Before touching the AI graph, ensure your server is alive and seed your vector database.

GET /health

Why: Verifies the API is up, and your PostgreSQL connection is active.

POST /repos/index

Why: Indexes your repository chunks into ChromaDB. Do this first so that the RAG flow actually has context to inject when the review is triggered.

GET /repos/{owner}/{repo}/status

Why: Confirms that the background indexing task finished successfully before moving forward.

Step 2: The Core Review Execution (LangGraph Trigger)
Now, initiate the actual agentic review loop.

POST /reviews/trigger

Why: Spins up the LangGraph workflow, fetches the PR diff, pulls context from ChromaDB, handles linting/validation inside the Docker sandbox, and pauses at the HITL gate.

What to verify: It should return a 202 or 200 with a review_id and a status like pending_hitl.

Step 3: Inspecting State & Verification
Before deciding on the review, check if the data landed correctly in your system.

GET /reviews/id/{review_id}

Why: Fetches the step-by-step execution history. Verify that your fixes worked—specifically that analyze_code_node succeeded and lint_node shows as skipped/passed for your README.md file rather than crashing.

GET /reviews/ or GET /reviews/repo/{owner}/{repo}

Why: Verifies your dashboard list queries work without running into that SQLAlchemy ambiguity error (InvalidRequestError).

Step 4: Human-in-the-Loop (HITL) Action
Since the graph is sitting at an active checkpoint awaiting your permission, process the verdict.

POST /reviews/id/{review_id}/decision (or your explicit /reviews/{review_id}/approve endpoint)

Why: Submits the human approval or rejection. This resumes the LangGraph compilation, runs the final summary nodes, and attempts to post the comment back to GitHub.

GET /reviews/{review_id}/status

Why: Confirms the review state successfully transitioned from pending_hitl to completed.

Step 5: Follow-up Chat (Interactive Layer)
Once the review is final, users will want to talk to the agent about the results.

POST /chat/{thread_id}/messages

Why: Sends an interactive message to discuss a specific piece of code from the review steps.

GET /chat/{thread_id}/messages

Why: Verifies the chat history persistence layer works accurately.

----------------------------------------------------------------------------------------------

#run complete project 
open two terminal and use this command  per terminal 
front end : streamlit run streamlit_app/dashboard.py
backend : python run.py

