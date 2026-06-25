database = {
        "name" : "My Name is Sadia Javed",
        "father": "My Father is Muhammad Shamim Akhtar",
        "religion": "I am Muslim",
        "qualification": "I have done my bachelors in Computer Science",
}

def search_tool(query):
        query_lower = query.lower()
        results=[]
        for key, value in database.items():
                if key in query_lower:
                        results.append(value)
        if results:
                return" ".join(results)
        return f"\n Sorry. couldnt find '{query}' "

def cal_tool(query):
        try:
                allowed = "0123456789+-*/(). "
                if not all (c in allowed for c in query):
                        return "Invalid Expression"
                return f"Results: {eval(query)}"
        except Exception as e:
                return f"Math error:{e}"

def default_tool(query):
        return f"Sorry this '{query}' is out of my reach, Try something relevant"

def decide_tool(query):
         operators = ["+", "-", "/", "*"]
         if any(op in query for op in operators):
                 return "calculate", query
         query_lower = query.lower()
         for key in database.keys():
                 if key in query_lower:
                         return "search" , query
         return "default", query

def react_agent(query):
        tool, tool_input = decide_tool(query)
        if tool == "search":
                result = search_tool(tool_input)
        elif tool == "calculate":
                result = cal_tool(tool_input)
        else:
                result = default_tool(tool_input)
        return result

print("=" * 50)
print("🤖 My First ReAct Agent")
print("=" * 50)
print("You can ask about me:")
print("  - Math")
print("Type 'exit' to quit")

while True:
        user_input = input("\n❓ You: ")
        if user_input.lower() == "exit":
                print("Goodbye")
                break
        else:
                result = react_agent(user_input)
                print(f"\n🤖 Agent: {result}")  
