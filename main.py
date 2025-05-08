# import requests
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from flask_pymongo import pymongo
# # Import UTC
# from datetime import datetime, UTC # <--- Import UTC
# import logging
# from bson.json_util import dumps

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = app.logger

# # MongoDB Configuration
# # MONGO_URI = 'mongodb://localhost:27017/'
# # DB_NAME = 'second_mind_db'
# # COLLECTION_NAME = 'chat_history'

# client = pymongo.MongoClient('mongodb+srv://saikaushiksadu:secondmind@cluster0.jvfivfn.mongodb.net/')
# db = client.get_database('SecondMind')
# chats_collection = pymongo.collection.Collection(db, 'chats')

# # Set OpenRouter API key (replace with your actual key)
# # --- IMPORTANT: Consider loading keys from environment variables or a config file ---
# OPENROUTER_API_KEY = "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" # REMOVE BEFORE SHARING/COMMIT
# API_URL = "https://openrouter.ai/api/v1/chat/completions"

# # Research Keywords
# RESEARCH_KEYWORDS = [
#     "research", "study", "academic", "experiment", "data analysis",
#     "ai research", "scientific", "thesis", "hypothesis", "peer review", # Added lowercase ai research
#     "paper", "article", "literature review" # Added more keywords
# ]

# # --- Semantic Scholar Search ---
# class SemanticScholarSearchTool:
#     """Semantic Scholar Search Tool."""

#     @staticmethod
#     def search_semantic_scholar(query, max_results=8):
#         """Search for research papers using Semantic Scholar."""
#         logger.info(f"🔍 Searching Semantic Scholar for: {query}") # Use logger
#         api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
#         params = {
#             "query": query,
#             "limit": max_results,
#             "fields": "title,authors,year,abstract,url"
#         }
#         headers = {'User-Agent': 'SecondMindApp/1.0 (contact@example.com)'} # Good practice User-Agent

#         try:
#             response = requests.get(api_url, params=params, headers=headers, timeout=15) # Increased timeout slightly
#             response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

#             papers = response.json().get("data", [])
#             results = []
#             for paper in papers:
#                 title = paper.get("title", "No title")
#                 # Handle potentially missing author names gracefully
#                 authors_list = [author.get("name", "N/A") for author in paper.get("authors", []) if author]
#                 authors = ", ".join(authors_list) if authors_list else "Authors N/A"
#                 year = paper.get("year", "N/A")
#                 abstract = paper.get("abstract", "No abstract available.")
#                 url = paper.get("url", "No URL available.")
#                 # Ensure consistent formatting, add newline at the very end
#                 result = f"Title: {title}\nAuthors: {authors}\nYear: {year}\nAbstract: {abstract}\nLink: {url}"
#                 results.append(result)

#             if results:
#                 return "\n\n".join(results) # Separate papers by double newline
#             else:
#                  logger.warning(f"Semantic Scholar found no relevant papers for query: {query}")
#                  return "❌ No relevant research papers found via Semantic Scholar."

#         except requests.exceptions.RequestException as e:
#             logger.error(f"Error accessing Semantic Scholar API: {str(e)}")
#             return f"❌ Error accessing Semantic Scholar API: {str(e)}"
#         except Exception as e: # Catch other potential errors like JSONDecodeError
#             logger.error(f"Unexpected error during Semantic Scholar search: {str(e)}")
#             return f"❌ Unexpected error processing Semantic Scholar results: {str(e)}"


# # --- Sub Agent ---
# class SubAgent:
#     def __init__(self, role, model="mistralai/mistral-7b-instruct"):
#         self.role = role
#         self.model = model
#         # Add keywords that strongly suggest needing web search/papers
#         self.web_search_keywords = ["current", "latest", "today", "recent", "find papers on", "search for articles", "show me studies"]
#         self.base_research_keywords = RESEARCH_KEYWORDS # Keep original list

#     def is_research_query(self, query):
#         """Check if query contains general research keywords."""
#         return any(keyword in query.lower() for keyword in self.base_research_keywords)

#     def needs_web_search(self, query):
#         """Determine if query explicitly asks for current info or papers."""
#         q_lower = query.lower()
#         # Trigger if specific web search keywords are present OR if "paper"/"article"/"study" is present
#         # AND the query is generally research related. Avoids triggering on non-research paper requests.
#         needs_search = any(keyword in q_lower for keyword in self.web_search_keywords)
#         asks_for_papers = any(term in q_lower for term in ["paper", "article", "study", "publication"])

#         # Trigger search if explicitly asking for current info OR asking for papers within a research context
#         if needs_search or (asks_for_papers and self.is_research_query(query)):
#              logger.info(f"Query '{query[:50]}...' determined to need web search.")
#              return True
#         return False

#     def get_response(self, user_query):
#         # Initial check moved to orchestrator for efficiency
#         # if not self.is_research_query(user_query):
#         #    logger.info(f"{self.role} skipping non-research query: {user_query[:50]}...")
#         #    return None # Return None instead of error message here

#         if self.needs_web_search(user_query):
#             web_info = SemanticScholarSearchTool.search_semantic_scholar(user_query)
#             # Return a clear structure if results found
#             if not web_info.startswith("❌"):
#                  return f"🔍 Research Results (via Semantic Scholar):\n{web_info}"
#             else:
#                  return web_info # Return the error message from the tool

#         # Fallback to general LLM if not needing web search
#         logger.info(f"{self.role} querying LLM for: {user_query[:50]}...")
#         try:
#             headers = {
#                 "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#                 "Content-Type": "application/json",
#                 "HTTP-Referer": "http://localhost", # Optional: Referer policy might apply
#                 "X-Title": "SecondMindApp"          # Optional: Title for OpenRouter analytics
#             }
#             data = {
#                 "model": self.model,
#                 "messages": [
#                     {"role": "system", "content": f"You are a helpful {self.role}. Provide concise and accurate information related to research topics."},
#                     {"role": "user", "content": user_query}
#                 ]
#             }
#             response = requests.post(API_URL, headers=headers, data=json.dumps(data), timeout=45) # Increased timeout
#             response.raise_for_status() # Check for HTTP errors

#             # Check if response has expected structure
#             response_data = response.json()
#             if "choices" in response_data and len(response_data["choices"]) > 0 and "message" in response_data["choices"][0] and "content" in response_data["choices"][0]["message"]:
#                  llm_response = response_data["choices"][0]["message"]["content"]
#                  # Basic check for empty/useless response
#                  if not llm_response or llm_response.strip() == "":
#                      logger.warning(f"LLM returned empty response for query: {user_query[:50]}...")
#                      return "⚠ LLM returned an empty response."
#                  return llm_response
#             else:
#                  logger.error(f"Unexpected API response structure: {response_data}")
#                  return "❌ API Error: Unexpected response structure from LLM."

#         except requests.exceptions.RequestException as e:
#             logger.error(f"API Request Error in {self.role}: {str(e)}")
#             return f"❌ API Error in {self.role}: {str(e)}"
#         except Exception as e:
#             logger.error(f"Error in {self.role} get_response: {str(e)}")
#             return f"❌ Error in {self.role}: {str(e)}"


# # --- Master AI Orchestrator ---
# class MasterAIAgent:
#     def __init__(self):
#         self.agents = {
#             "Research_Assistant": SubAgent("Research_Assistant"),
#             "Data_Analyzer": SubAgent("Data_Analyzer", model="mistralai/mixtral-8x7b-instruct"), # Example: Use a different model
#             "Paper_Summarizer": SubAgent("Paper_Summarizer")
#         }
#         # Add a check if query is research-related upfront
#         self.research_keywords_lower = [k.lower() for k in RESEARCH_KEYWORDS]

#     def _is_research_query(self, query):
#          return any(keyword in query.lower() for keyword in self.research_keywords_lower)

#     def get_responses(self, user_query):
#         responses = {}
#         # Check if it's a research query *before* calling agents
#         if not self._is_research_query(user_query):
#             logger.info("Query determined as non-research. Skipping agent calls.")
#             # Return a specific message instead of None/empty dict
#             return {"System": "This query does not seem research-related. I primarily handle research topics."}

#         logger.info(f"Orchestrator getting responses for: {user_query[:50]}...")
#         for agent_name, agent in self.agents.items():
#             # Pass the check result if needed, or just let agents handle it (current SubAgent does)
#             response = agent.get_response(user_query)
#             if response: # Only add if agent provided a response (not None)
#                 responses[agent_name] = response
#             else:
#                  logger.info(f"Agent {agent_name} returned no response.")

#         # Handle case where no agent responded
#         if not responses:
#              logger.warning("No agent provided a response for the research query.")
#              # Provide a generic fallback or indicate failure
#              return {"System": "I couldn't generate a specific response from my research agents for this query."}

#         return responses

#     def rank_responses(self, responses):
#          # If only one response (or a system message), no need to rank
#         valid_responses = {k: v for k, v in responses.items() if not v.startswith("❌") and k != "System"}
#         if len(valid_responses) <= 1:
#             # Return the first valid response found, or the original dict if only errors/system messages
#             return next(iter(valid_responses.values()), list(responses.values())[0] if responses else "No valid response to rank.")


#         ranking_prompt = "You are an expert research analyst. Rank the following responses to a user's research query based on accuracy, relevance, and helpfulness. Present ONLY the single best response, without any extra explanation or commentary.\n\n--- Responses ---\n"
#         response_items = []
#         for i, (agent, response) in enumerate(valid_responses.items(), 1):
#              ranking_prompt += f"Response {i} (from {agent}):\n{response}\n\n"
#              response_items.append(response) # Store original responses

#         ranking_prompt += "--- Instruction ---\nSelect the single best response and output it directly."

#         logger.info("Orchestrator ranking responses...")
#         try:
#             headers = {
#                 "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#                 "Content-Type": "application/json",
#                 "HTTP-Referer": "http://localhost",
#                 "X-Title": "SecondMindApp-Ranker"
#             }
#             data = {
#                 # Use a potentially more capable model for ranking
#                 "model": "mistralai/mixtral-8x7b-instruct",
#                 "messages": [{"role": "user", "content": ranking_prompt}],
#                 "temperature": 0.1 # Low temperature for deterministic ranking
#             }

#             response = requests.post(API_URL, headers=headers, data=json.dumps(data), timeout=45)
#             response.raise_for_status()

#             ranked_data = response.json()
#             if "choices" in ranked_data and len(ranked_data["choices"]) > 0:
#                  best_response = ranked_data["choices"][0]["message"]["content"]
#                  # Optional: Try to match the ranked response back to one of the originals
#                  # This is difficult if the ranking model rephrases. For now, return its output.
#                  logger.info("Ranking complete. Returning ranked response.")
#                  return best_response
#             else:
#                  logger.error(f"API Error in ranking - unexpected structure: {ranked_data}")
#                  # Fallback: return the first valid response if ranking fails
#                  return response_items[0] if response_items else "❌ API Error in ranking: Unexpected response structure."

#         except requests.exceptions.RequestException as e:
#             logger.error(f"API Error in ranking: {str(e)}")
#              # Fallback: return the first valid response if ranking fails
#             return response_items[0] if response_items else f"❌ API Error in ranking: {str(e)}"
#         except Exception as e:
#             logger.error(f"Error in ranking responses: {str(e)}")
#              # Fallback: return the first valid response if ranking fails
#             return response_items[0] if response_items else f"❌ Error in ranking responses: {str(e)}"


#     def respond_with_iterations(self, user_query, iterations=1):
#         # Check moved to get_responses
#         # if not self._is_research_query(user_query):
#         #     return "❌ This system primarily handles research-based queries."

#         final_response = "Processing..."
#         current_query = user_query

#         for i in range(iterations):
#             logger.info(f"--- Iteration {i + 1} / {iterations} ---")
#             responses = self.get_responses(current_query)

#             # Check if get_responses returned a system message (e.g., non-research)
#             if len(responses) == 1 and "System" in responses:
#                  final_response = responses["System"]
#                  logger.info(f"Iteration {i + 1}: System message received, stopping iterations.")
#                  break # Stop iterating if it's not a research query or no agents responded

#             # If only one actual response, use it directly
#             valid_agent_responses = {k: v for k, v in responses.items() if k != "System"}
#             if len(valid_agent_responses) == 1:
#                 final_response = list(valid_agent_responses.values())[0]
#                 logger.info(f"Iteration {i + 1}: Only one agent response, using it.")
#                 # If not the last iteration, use this response as the next query
#                 if i < iterations - 1:
#                      current_query = f"Refine this response based on the original query '{user_query}':\n\n{final_response}"
#                 continue # Move to next iteration or finish

#             # Rank if multiple responses
#             ranked_response = self.rank_responses(responses)
#             final_response = ranked_response

#             # Prepare for next iteration if needed
#             if i < iterations - 1:
#                 # Create a refined query for the next iteration
#                 current_query = f"Based on the original query '{user_query}', critically review and improve the following response, focusing on accuracy and completeness:\n\n{ranked_response}"
#                 logger.info(f"Iteration {i + 1}: Using ranked response as input for next iteration.")
#             else:
#                  logger.info(f"Iteration {i + 1}: Final iteration, returning ranked response.")


#         # Handle cases where iterations completed but no valid response was generated
#         if final_response == "Processing...":
#              logger.error("Iterations completed without generating a final response.")
#              final_response = "❌ An error occurred during processing, and no final response could be generated."

#         return final_response

# # --- Initialize AI ---
# orchestrator = MasterAIAgent()

# # --- API Routes ---
# @app.route("/ask", methods=["POST"])
# def ask_ai():
#     # Check DB connection status first
#     try:
#         data = request.get_json()
#         if not data or "query" not in data:
#             logger.warning("Invalid request to /ask: Missing 'query' field.")
#             return jsonify({"error": "Invalid request. Please provide a 'query' field."}), 400

#         user_query = data["query"]
#         # Sanitize iterations, default to 1, max 5
#         try:
#             iterations = max(1, min(int(data.get("iterations", 1)), 5))
#         except (ValueError, TypeError):
#             iterations = 1
#             logger.warning("Invalid 'iterations' value received, defaulting to 1.")


#         logger.info(f"Received query: '{user_query[:100]}...' with {iterations} iterations")
#         ai_response = orchestrator.respond_with_iterations(user_query, iterations)

#         # Mongo query
#         try:
#             result = db['chats'].insert_one({
#                 "query":user_query,
#                 "response": ai_response
#             })
#             return jsonify({
#                 'status': 'Success',
#                 "inserted_id":str(result.inserted_id),
#                 "response": ai_response
#             }), 200
        
#         except Exception as e:
#             print(e)
#             return jsonify({
#                 'status': 'Error occurred'
#             })

#     except Exception as e:
#         logger.error(f"Error processing /ask request: {str(e)}")
#         return jsonify({"error": f"Server error: {str(e)}"}), 500


# @app.route("/history", methods=["GET"])
# def get_chat_history():
#     try:
#         result_chats = db['chats'].find()
#         chat_list = list(result_chats)
#         return dumps(chat_list), 200  
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route("/clear_history", methods=["POST"])
# def clear_history():
#     try:
#         result = db['chats'].delete_many({})
#         return jsonify({
#             "message": "Chat history cleared successfully.",
#             "deleted_count": result.deleted_count
#         }), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True, use_reloader=False)




















import requests
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
# Import UTC
from datetime import datetime, UTC # <--- Import UTC
import logging
from pymongo.errors import ConnectionFailure # <--- Import ConnectionFailure

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = app.logger

# MongoDB Configuration
MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'second_mind_db'
COLLECTION_NAME = 'chat_history'

# --- Try to connect to MongoDB ---
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000) # Add timeout
    # The ismaster command is cheap and does not require auth.
    client.admin.command('ismaster')
    logger.info("✅ Successfully connected to MongoDB.")
    db = client[DB_NAME]
    chat_history_collection = db[COLLECTION_NAME]
except ConnectionFailure as e:
    logger.error(f"❌❌❌ CRITICAL: Could not connect to MongoDB at {MONGO_URI}. Please ensure MongoDB is running.")
    logger.error(f"Error details: {e}")
    # You could choose to exit or run in a degraded mode if MongoDB is essential
    # For now, we'll let it continue, but endpoints will fail.
    client = None
    db = None
    chat_history_collection = None

# Set OpenRouter API key (replace with your actual key)
# --- IMPORTANT: Consider loading keys from environment variables or a config file ---
OPENROUTER_API_KEY = "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" # REMOVE BEFORE SHARING/COMMIT
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Research Keywords
RESEARCH_KEYWORDS = [
    "research", "study", "academic", "experiment", "data analysis",
    "ai research", "scientific", "thesis", "hypothesis", "peer review", # Added lowercase ai research
    "paper", "article", "literature review" # Added more keywords
]

# --- Semantic Scholar Search ---
class SemanticScholarSearchTool:
    """Semantic Scholar Search Tool."""

    @staticmethod
    def search_semantic_scholar(query, max_results=8):
        """Search for research papers using Semantic Scholar."""
        logger.info(f"🔍 Searching Semantic Scholar for: {query}") # Use logger
        api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,year,abstract,url"
        }
        headers = {'User-Agent': 'SecondMindApp/1.0 (contact@example.com)'} # Good practice User-Agent

        try:
            response = requests.get(api_url, params=params, headers=headers, timeout=15) # Increased timeout slightly
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            papers = response.json().get("data", [])
            results = []
            for paper in papers:
                title = paper.get("title", "No title")
                # Handle potentially missing author names gracefully
                authors_list = [author.get("name", "N/A") for author in paper.get("authors", []) if author]
                authors = ", ".join(authors_list) if authors_list else "Authors N/A"
                year = paper.get("year", "N/A")
                abstract = paper.get("abstract", "No abstract available.")
                url = paper.get("url", "No URL available.")
                # Ensure consistent formatting, add newline at the very end
                result = f"Title: {title}\nAuthors: {authors}\nYear: {year}\nAbstract: {abstract}\nLink: {url}"
                results.append(result)

            if results:
                return "\n\n".join(results) # Separate papers by double newline
            else:
                 logger.warning(f"Semantic Scholar found no relevant papers for query: {query}")
                 return "❌ No relevant research papers found via Semantic Scholar."

        except requests.exceptions.RequestException as e:
            logger.error(f"Error accessing Semantic Scholar API: {str(e)}")
            return f"❌ Error accessing Semantic Scholar API: {str(e)}"
        except Exception as e: # Catch other potential errors like JSONDecodeError
            logger.error(f"Unexpected error during Semantic Scholar search: {str(e)}")
            return f"❌ Unexpected error processing Semantic Scholar results: {str(e)}"


# --- Sub Agent ---
class SubAgent:
    def __init__(self, role, model="mistralai/mistral-7b-instruct"):
        self.role = role
        self.model = model
        # Add keywords that strongly suggest needing web search/papers
        self.web_search_keywords = ["current", "latest", "today", "recent", "find papers on", "search for articles", "show me studies"]
        self.base_research_keywords = RESEARCH_KEYWORDS # Keep original list

    def is_research_query(self, query):
        """Check if query contains general research keywords."""
        return any(keyword in query.lower() for keyword in self.base_research_keywords)

    def needs_web_search(self, query):
        """Determine if query explicitly asks for current info or papers."""
        q_lower = query.lower()
        # Trigger if specific web search keywords are present OR if "paper"/"article"/"study" is present
        # AND the query is generally research related. Avoids triggering on non-research paper requests.
        needs_search = any(keyword in q_lower for keyword in self.web_search_keywords)
        asks_for_papers = any(term in q_lower for term in ["paper", "article", "study", "publication"])

        # Trigger search if explicitly asking for current info OR asking for papers within a research context
        if needs_search or (asks_for_papers and self.is_research_query(query)):
             logger.info(f"Query '{query[:50]}...' determined to need web search.")
             return True
        return False

    def get_response(self, user_query):
        # Initial check moved to orchestrator for efficiency
        # if not self.is_research_query(user_query):
        #    logger.info(f"{self.role} skipping non-research query: {user_query[:50]}...")
        #    return None # Return None instead of error message here

        if self.needs_web_search(user_query):
            web_info = SemanticScholarSearchTool.search_semantic_scholar(user_query)
            # Return a clear structure if results found
            if not web_info.startswith("❌"):
                 return f"🔍 Research Results (via Semantic Scholar):\n{web_info}"
            else:
                 return web_info # Return the error message from the tool

        # Fallback to general LLM if not needing web search
        logger.info(f"{self.role} querying LLM for: {user_query[:50]}...")
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost", # Optional: Referer policy might apply
                "X-Title": "SecondMindApp"          # Optional: Title for OpenRouter analytics
            }
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": f"You are a helpful {self.role}. Provide concise and accurate information related to research topics."},
                    {"role": "user", "content": user_query}
                ]
            }
            response = requests.post(API_URL, headers=headers, data=json.dumps(data), timeout=45) # Increased timeout
            response.raise_for_status() # Check for HTTP errors

            # Check if response has expected structure
            response_data = response.json()
            if "choices" in response_data and len(response_data["choices"]) > 0 and "message" in response_data["choices"][0] and "content" in response_data["choices"][0]["message"]:
                 llm_response = response_data["choices"][0]["message"]["content"]
                 # Basic check for empty/useless response
                 if not llm_response or llm_response.strip() == "":
                     logger.warning(f"LLM returned empty response for query: {user_query[:50]}...")
                     return "⚠️ LLM returned an empty response."
                 return llm_response
            else:
                 logger.error(f"Unexpected API response structure: {response_data}")
                 return "❌ API Error: Unexpected response structure from LLM."

        except requests.exceptions.RequestException as e:
            logger.error(f"API Request Error in {self.role}: {str(e)}")
            return f"❌ API Error in {self.role}: {str(e)}"
        except Exception as e:
            logger.error(f"Error in {self.role} get_response: {str(e)}")
            return f"❌ Error in {self.role}: {str(e)}"


# --- Master AI Orchestrator ---
class MasterAIAgent:
    def __init__(self):
        self.agents = {
            "Research_Assistant": SubAgent("Research_Assistant"),
            "Data_Analyzer": SubAgent("Data_Analyzer", model="mistralai/mixtral-8x7b-instruct"), # Example: Use a different model
            "Paper_Summarizer": SubAgent("Paper_Summarizer")
        }
        # Add a check if query is research-related upfront
        self.research_keywords_lower = [k.lower() for k in RESEARCH_KEYWORDS]

    def _is_research_query(self, query):
         return any(keyword in query.lower() for keyword in self.research_keywords_lower)

    def get_responses(self, user_query):
        responses = {}
        # Check if it's a research query *before* calling agents
        if not self._is_research_query(user_query):
            logger.info("Query determined as non-research. Skipping agent calls.")
            # Return a specific message instead of None/empty dict
            return {"System": "This query does not seem research-related. I primarily handle research topics."}

        logger.info(f"Orchestrator getting responses for: {user_query[:50]}...")
        for agent_name, agent in self.agents.items():
            # Pass the check result if needed, or just let agents handle it (current SubAgent does)
            response = agent.get_response(user_query)
            if response: # Only add if agent provided a response (not None)
                responses[agent_name] = response
            else:
                 logger.info(f"Agent {agent_name} returned no response.")

        # Handle case where no agent responded
        if not responses:
             logger.warning("No agent provided a response for the research query.")
             # Provide a generic fallback or indicate failure
             return {"System": "I couldn't generate a specific response from my research agents for this query."}

        return responses

    def rank_responses(self, responses):
         # If only one response (or a system message), no need to rank
        valid_responses = {k: v for k, v in responses.items() if not v.startswith("❌") and k != "System"}
        if len(valid_responses) <= 1:
            # Return the first valid response found, or the original dict if only errors/system messages
            return next(iter(valid_responses.values()), list(responses.values())[0] if responses else "No valid response to rank.")


        ranking_prompt = "You are an expert research analyst. Rank the following responses to a user's research query based on accuracy, relevance, and helpfulness. Present ONLY the single best response, without any extra explanation or commentary.\n\n--- Responses ---\n"
        response_items = []
        for i, (agent, response) in enumerate(valid_responses.items(), 1):
             ranking_prompt += f"Response {i} (from {agent}):\n{response}\n\n"
             response_items.append(response) # Store original responses

        ranking_prompt += "--- Instruction ---\nSelect the single best response and output it directly."

        logger.info("Orchestrator ranking responses...")
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "SecondMindApp-Ranker"
            }
            data = {
                # Use a potentially more capable model for ranking
                "model": "mistralai/mixtral-8x7b-instruct",
                "messages": [{"role": "user", "content": ranking_prompt}],
                "temperature": 0.1 # Low temperature for deterministic ranking
            }

            response = requests.post(API_URL, headers=headers, data=json.dumps(data), timeout=45)
            response.raise_for_status()

            ranked_data = response.json()
            if "choices" in ranked_data and len(ranked_data["choices"]) > 0:
                 best_response = ranked_data["choices"][0]["message"]["content"]
                 # Optional: Try to match the ranked response back to one of the originals
                 # This is difficult if the ranking model rephrases. For now, return its output.
                 logger.info("Ranking complete. Returning ranked response.")
                 return best_response
            else:
                 logger.error(f"API Error in ranking - unexpected structure: {ranked_data}")
                 # Fallback: return the first valid response if ranking fails
                 return response_items[0] if response_items else "❌ API Error in ranking: Unexpected response structure."

        except requests.exceptions.RequestException as e:
            logger.error(f"API Error in ranking: {str(e)}")
             # Fallback: return the first valid response if ranking fails
            return response_items[0] if response_items else f"❌ API Error in ranking: {str(e)}"
        except Exception as e:
            logger.error(f"Error in ranking responses: {str(e)}")
             # Fallback: return the first valid response if ranking fails
            return response_items[0] if response_items else f"❌ Error in ranking responses: {str(e)}"


    def respond_with_iterations(self, user_query, iterations=1):
        # Check moved to get_responses
        # if not self._is_research_query(user_query):
        #     return "❌ This system primarily handles research-based queries."

        final_response = "Processing..."
        current_query = user_query

        for i in range(iterations):
            logger.info(f"--- Iteration {i + 1} / {iterations} ---")
            responses = self.get_responses(current_query)

            # Check if get_responses returned a system message (e.g., non-research)
            if len(responses) == 1 and "System" in responses:
                 final_response = responses["System"]
                 logger.info(f"Iteration {i + 1}: System message received, stopping iterations.")
                 break # Stop iterating if it's not a research query or no agents responded

            # If only one actual response, use it directly
            valid_agent_responses = {k: v for k, v in responses.items() if k != "System"}
            if len(valid_agent_responses) == 1:
                final_response = list(valid_agent_responses.values())[0]
                logger.info(f"Iteration {i + 1}: Only one agent response, using it.")
                # If not the last iteration, use this response as the next query
                if i < iterations - 1:
                     current_query = f"Refine this response based on the original query '{user_query}':\n\n{final_response}"
                continue # Move to next iteration or finish

            # Rank if multiple responses
            ranked_response = self.rank_responses(responses)
            final_response = ranked_response

            # Prepare for next iteration if needed
            if i < iterations - 1:
                # Create a refined query for the next iteration
                current_query = f"Based on the original query '{user_query}', critically review and improve the following response, focusing on accuracy and completeness:\n\n{ranked_response}"
                logger.info(f"Iteration {i + 1}: Using ranked response as input for next iteration.")
            else:
                 logger.info(f"Iteration {i + 1}: Final iteration, returning ranked response.")


        # Handle cases where iterations completed but no valid response was generated
        if final_response == "Processing...":
             logger.error("Iterations completed without generating a final response.")
             final_response = "❌ An error occurred during processing, and no final response could be generated."

        return final_response

# --- Initialize AI ---
orchestrator = MasterAIAgent()

# --- API Routes ---
@app.route("/ask", methods=["POST"])
def ask_ai():
    # Check DB connection status first
    if client is None or db is None or chat_history_collection is None:
         logger.error("Database connection not available. Cannot process /ask request.")
         return jsonify({"error": "Database connection failed. Cannot process request."}), 503 # Service Unavailable

    try:
        data = request.get_json()
        if not data or "query" not in data:
            logger.warning("Invalid request to /ask: Missing 'query' field.")
            return jsonify({"error": "Invalid request. Please provide a 'query' field."}), 400

        user_query = data["query"]
        # Sanitize iterations, default to 1, max 5
        try:
            iterations = max(1, min(int(data.get("iterations", 1)), 5))
        except (ValueError, TypeError):
            iterations = 1
            logger.warning("Invalid 'iterations' value received, defaulting to 1.")


        logger.info(f"Received query: '{user_query[:100]}...' with {iterations} iterations")
        ai_response = orchestrator.respond_with_iterations(user_query, iterations)

        # Save to MongoDB
        try:
            chat_record = {
                "query": user_query,
                "response": ai_response,
                # Use timezone-aware UTC time (FIXED)
                "timestamp": datetime.now(UTC)
            }
            insert_result = chat_history_collection.insert_one(chat_record)
            logger.info(f"Successfully saved chat to MongoDB (ID: {insert_result.inserted_id})")
            # Return the AI response even if saving failed previously
            return jsonify({"response": ai_response}), 200

        except ConnectionFailure as e: # Catch specific connection errors
             logger.error(f"MongoDB connection error during insert: {str(e)}")
             # Return the response but with a warning about history saving
             return jsonify({"response": ai_response, "warning": f"AI response generated, but failed to save chat history due to DB connection error: {e}"}), 200
        except Exception as e:
             logger.error(f"MongoDB generic error during insert: {str(e)}")
             # Return the response but with a warning about history saving
             return jsonify({"response": ai_response, "warning": f"AI response generated, but failed to save chat history: {e}"}), 200

    except Exception as e:
        logger.error(f"Error processing /ask request: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/history", methods=["GET"])
def get_chat_history():
     # Check DB connection status first
    if client is None or db is None or chat_history_collection is None:
         logger.error("Database connection not available. Cannot process /history request.")
         return jsonify({"error": "Database connection failed. Cannot retrieve history."}), 503 # Service Unavailable

    try:
        logger.info("Fetching chat history")
        # Add a limit to prevent fetching huge amounts of data
        history_cursor = chat_history_collection.find().sort("timestamp", -1).limit(100)
        history = list(history_cursor)

        # Convert MongoDB ObjectId to string and format timestamp
        for item in history:
            item["_id"] = str(item["_id"])
            # Ensure timestamp exists and is a datetime object before formatting
            ts = item.get("timestamp")
            if isinstance(ts, datetime):
                item["timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S UTC") # Indicate UTC
            else:
                 item["timestamp"] = "N/A" # Handle potential missing/invalid timestamps


        logger.info(f"Returning {len(history)} history items")
        return jsonify(history)

    except ConnectionFailure as e: # Catch specific connection errors
        logger.error(f"MongoDB connection error retrieving history: {str(e)}")
        return jsonify({"error": f"Failed to retrieve history due to DB connection error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        return jsonify({"error": f"Failed to retrieve history: {str(e)}"}), 500


@app.route("/clear_history", methods=["POST"])
def clear_history():
     # Check DB connection status first
    if client is None or db is None or chat_history_collection is None:
         logger.error("Database connection not available. Cannot process /clear_history request.")
         return jsonify({"error": "Database connection failed. Cannot clear history."}), 503 # Service Unavailable

    try:
        result = chat_history_collection.delete_many({})
        num_rows_deleted = result.deleted_count
        logger.info(f"History cleared successfully ({num_rows_deleted} records deleted)")
        return jsonify({
            "status": "success",
            "message": f"History cleared successfully ({num_rows_deleted} records deleted)"
        }), 200
    except ConnectionFailure as e: # Catch specific connection errors
        logger.error(f"MongoDB connection error clearing history: {str(e)}")
        return jsonify({"error": f"Failed to clear history due to DB connection error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({"error": f"Failed to clear history: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    db_status = "failed"
    db_error = "Client not initialized"
    http_status_code = 503 # Default to service unavailable

    if client:
        try:
            # Ping the database server.
            client.admin.command('ping')
            db_status = "ok"
            db_error = None
            http_status_code = 200
            logger.debug("Health check: MongoDB ping successful.")
        except ConnectionFailure as e:
            db_error = f"ConnectionFailure: {e}"
            logger.error(f"Health check failed: MongoDB connection error: {db_error}")
            http_status_code = 503
        except Exception as e:
             db_error = f"Exception: {e}"
             logger.error(f"Health check failed: Unexpected error during MongoDB ping: {db_error}")
             http_status_code = 500


    response_body = {
        "status": "healthy" if db_status == "ok" else "unhealthy",
        "dependencies": {
             "database_connection": {
                 "status": db_status,
                 "error": db_error
            }
            # Add other dependency checks here if needed (e.g., OpenRouter API)
        }
    }
    return jsonify(response_body), http_status_code


if __name__ == "__main__":
    # Check if MongoDB is connected before starting the app fully
    if not client:
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print("!!! CRITICAL: Failed to connect to MongoDB on startup.     !!!")
         print("!!! Please ensure MongoDB is running at mongodb://localhost:27017/ !!!")
         print("!!! The application will run, but database features (history) will fail. !!!")
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # Note: Consider adding host='0.0.0.0' if you need to access it from other devices on your network
    app.run(debug=True, use_reloader=False) # debug=True enables auto-reloading and provides debugger


















