import os
import warnings
warnings.filterwarnings("ignore")
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""
os.environ["LANGCHAIN_VERBOSE"] = "false"
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box

os.environ["LANGCHAIN_TRACING"] = "false"


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print(" Error: GROQ_API_KEY not found")
    exit(1)


console = Console()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=api_key
)


SYSTEM_PROMPT = """
You are a helpful AI assistant named "CalderBot".

PERSONALITY:
- Friendly, professional, and knowledgeable
- You explain things clearly and concisely
- You are enthusiastic about helping people learn

RULES:
- If you don't know something, say so
- Provide accurate and helpful information
- Keep responses under 200 words when possible
- Be engaging and interactive
"""

class CalderBot:
    def __init__(self):
        self.messages = [SystemMessage(content=SYSTEM_PROMPT)]
        self.conversation_history = []
        self.total_tokens = 0
    
    def add_user_message(self, content):
        msg = HumanMessage(content=content)
        self.messages.append(msg)
        self.conversation_history.append((" You", content))
    
    def add_ai_message(self, content, token_count):
        msg = AIMessage(content=content)
        self.messages.append(msg)
        self.conversation_history.append((" CalderBot", content))
        self.total_tokens += token_count
    
    def get_response(self, user_input):
        try:
            self.add_user_message(user_input)
            response = llm.invoke(self.messages)
            token_count = len(response.content) // 4
            self.add_ai_message(response.content, token_count)
            return response.content, token_count
        except Exception as e:
            return f" Error: {e}", 0
    
    def clear_history(self):
        self.messages = [SystemMessage(content=SYSTEM_PROMPT)]
        self.conversation_history = []
        self.total_tokens = 0
    
    def get_stats(self):
        return {
            "turns": len(self.conversation_history) // 2,
            "tokens": self.total_tokens,
            "messages": len(self.messages)
        }


def display_welcome():
    console.clear()
    welcome_panel = Panel(
        "[bold cyan] CalderBot[/bold cyan]\n"
        "[green]Your AI Engineering Assistant[/green]\n\n"
        "I can help with:\n"
        "  • Coding questions\n"
        "  • AI concepts\n"
        "  • Problem-solving\n"
        "  • Learning new topics\n\n"
        "[yellow]Commands:[/yellow]\n"
        "  [white]/clear[/white] - Clear chat history\n"
        "  [white]/stats[/white]  - Show statistics\n"
        "  [white]/exit[/white]   - Quit the chat",
        title="Welcome to CalderBot",
        border_style="cyan"
    )
    console.print(welcome_panel)
    console.print()

def display_stats(stats):
    table = Table(title="📊 Chat Statistics", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Conversation Turns", str(stats["turns"]))
    table.add_row("Total Tokens Used", str(stats["tokens"]))
    table.add_row("Messages in Context", str(stats["messages"]))
    console.print(table)
    console.print()

def display_message(role, content):
    color = "cyan" if role == " You" else "green"
    panel = Panel(
        content,
        title=role,
        border_style=color,
        width=80
    )
    console.print(panel)

def main():
    bot = CalderBot()
    display_welcome()
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            if user_input.lower() == "/exit":
                console.print("[bold yellow] Goodbye![/bold yellow]")
                break
            
            if user_input.lower() == "/clear":
                bot.clear_history()
                console.print("[bold green] Chat history cleared![/bold green]")
                continue
            
            if user_input.lower() == "/stats":
                stats = bot.get_stats()
                display_stats(stats)
                continue
            
            if not user_input.strip():
                continue
            
            with console.status("[bold yellow] Thinking...[/bold yellow]"):
                response, tokens = bot.get_response(user_input)
            
            display_message(" You", user_input)
            display_message(" CalderBot", response)
            console.print(f"[dim]Tokens used: {tokens} | Total: {bot.total_tokens}[/dim]")
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red] Error: {e}[/bold red]")

if __name__ == "__main__":
    main()
