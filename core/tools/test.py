# TEST TOOL KNOWLEDGE RETRIEVAL

import asyncio
from knowledge_retrieval import knowledge_retrieval_tool_function
 
async def test_query():
    query = "What is DDos Attack" 
    result = await knowledge_retrieval_tool_function(query)
    print("===== Result =====")
    print(result.get("retrieved_knowledge") or result.get("error"))

if __name__ == "__main__":
    asyncio.run(test_query())
