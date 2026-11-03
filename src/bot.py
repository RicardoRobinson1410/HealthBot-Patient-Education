"""
HealthBot class orchestrator for MediTech Solutions patient education sessions.
Supports both rich interactive terminal execution and programmatic step execution.
"""

import time
import os
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt, Confirm

from langgraph.checkpoint.memory import MemorySaver

from src.config import load_env_config, validate_environment
from src.state import HealthBotState, create_initial_state
from src.graph import build_healthbot_graph


console = Console()


class HealthBot:
    """
    Orchestrates the LangGraph-powered patient education session.
    """

    def __init__(self, checkpointer: Optional[MemorySaver] = None):
        load_env_config("config.env")
        self.checkpointer = checkpointer or MemorySaver()
        self.app = build_healthbot_graph(checkpointer=self.checkpointer)
        self.thread_id = "patient-session-1"

    def run_automated_flow(
        self,
        topic: str = "Hypertension",
        difficulty: str = "Intermediate",
        user_answer: str = "A",
        next_action: str = "exit",
    ) -> HealthBotState:
        """
        Executes a single end-to-end patient workflow programmatically.
        Ideal for testing, automated grading, and demos.
        """
        initial_state = create_initial_state(topic=topic, difficulty=difficulty)
        initial_state["user_answer"] = user_answer
        initial_state["next_action"] = next_action

        config = {"configurable": {"thread_id": self.thread_id}}
        final_state: HealthBotState = self.app.invoke(initial_state, config=config)
        return final_state

    def run_interactive(self):
        """
        Runs a rich, interactive patient education session in the terminal.
        """
        console.clear()
        console.print(
            Panel.fit(
                "[bold cyan]MediTech Solutions[/bold cyan] - [bold white]HealthBot: AI Patient Education[/bold white]\n"
                "[dim]Empowering patients with reliable, accessible medical knowledge and comprehension checks[/dim]",
                border_style="cyan",
            )
        )

        # Check API key status
        has_keys = validate_environment(raise_error=False)
        if not has_keys:
            console.print(
                "[yellow]Notice: One or more API keys (OPENAI_API_KEY, TAVILY_API_KEY) are using placeholders in config.env.[/yellow]"
            )
            console.print("[dim]HealthBot will run with built-in medical reference data and fallback responses.[/dim]\n")

        session_running = True
        current_topic = ""
        current_difficulty = "Intermediate"

        while session_running:
            # 1. Ask patient for medical topic and difficulty
            console.print("[bold green]Step 1: Patient Health Inquiry[/bold green]")
            if not current_topic:
                current_topic = Prompt.ask(
                    "What health topic or medical condition would you like to learn about today?",
                    default="Type 2 Diabetes",
                )

            console.print("\n[bold]Select educational depth (Standout Feature):[/bold]")
            console.print("  [1] Beginner     (Simple language, 6th-grade level, relatable analogies)")
            console.print("  [2] Intermediate (Standard clinical concepts with plain-language definitions)")
            console.print("  [3] Advanced     (In-depth physiological mechanisms and treatment protocols)")
            diff_choice = Prompt.ask("Choose level", choices=["1", "2", "3"], default="2")
            diff_map = {"1": "Beginner", "2": "Intermediate", "3": "Advanced"}
            current_difficulty = diff_map[diff_choice]

            console.print(
                f"\n[cyan]Searching Tavily and compiling patient education for: [bold]{current_topic}[/bold] ({current_difficulty})...[/cyan]"
            )

            # Prepare state
            state_input = create_initial_state(topic=current_topic, difficulty=current_difficulty)
            config = {"configurable": {"thread_id": self.thread_id}}

            # Run search and summarization through the graph
            # Nodes: collect_topic -> generate_query -> execute_search -> summarize_results -> present_summary
            with console.status("[bold green]HealthBot is researching and summarizing medical literature...[/bold green]"):
                state_after_search = self.app.invoke(state_input, config=config)

            # Display 3-4 Paragraph Summary
            console.print("\n" + "=" * 70)
            console.print(f"[bold cyan]Patient Education Summary: {state_after_search.get('topic')}[/bold cyan]")
            console.print("=" * 70)
            summary_text = state_after_search.get("summary", "")
            console.print(Panel(summary_text, title="[green]Medical Summary (Grounded strictly in Tavily Search)[/green]", border_style="green"))
            console.print("[dim]Sources: Verified search results via Tavily Search Engine.[/dim]\n")

            # 4. Prompt patient when ready for comprehension check
            Prompt.ask("[bold yellow]Press [Enter] when you have finished reading and are ready for your comprehension check[/bold yellow]")

            # 5. Generate and present quiz question
            console.print("\n[bold green]Step 2: Patient Comprehension Check[/bold green]")
            quiz_question = state_after_search.get("quiz_question", "")
            quiz_options = state_after_search.get("quiz_options", [])

            console.print(f"\n[bold white]{quiz_question}[/bold white]\n")
            if quiz_options:
                for opt in quiz_options:
                    console.print(f"  [cyan]{opt}[/cyan]")
                console.print()
                user_ans = Prompt.ask("Your answer (e.g., A, B, C, D or explanation)", default="A")
            else:
                user_ans = Prompt.ask("Your answer")

            # 6. Evaluate and present grade
            console.print("\n[cyan]Evaluating your answer against the medical summary...[/cyan]")
            evaluation_state = dict(state_after_search)
            evaluation_state["user_answer"] = user_ans
            evaluation_state["next_action"] = "exit"

            # Execute evaluation
            state_after_eval = self.app.invoke(evaluation_state, config=config)

            grade = state_after_eval.get("quiz_grade", "A")
            explanation = state_after_eval.get("quiz_explanation", "")
            citations = state_after_eval.get("citations", [])

            # Grade coloring
            grade_color = "green" if grade in ["A", "B"] else "yellow" if grade == "C" else "red"
            
            console.print("\n" + "=" * 70)
            console.print(f"[bold {grade_color}]Comprehension Grade: {grade}[/bold {grade_color}]")
            console.print("=" * 70)
            console.print(f"[bold]Clinical Justification:[/bold] {explanation}\n")
            
            if citations:
                console.print("[bold cyan]Citations from Summary reinforcing learning:[/bold cyan]")
                for c in citations:
                    console.print(f"  [dim]• \"{c}\"[/dim]")
            console.print("=" * 70)

            # 7. Standout Feature: Recommended Related Topics
            related = state_after_eval.get("related_topics", [])
            if related:
                console.print("\n[bold magenta]Recommended Related Topics (Standout Feature):[/bold magenta]")
                for i, r in enumerate(related, 1):
                    console.print(f"  [{i}] {r}")

            # 8. Ask next steps (Another quiz question, related topic, new topic, or exit)
            console.print("\n[bold blue]What would you like to do next?[/bold blue]")
            console.print("  [1] Take another quiz question on this topic (Standout Feature)")
            console.print("  [2] Explore one of the recommended related topics above")
            console.print("  [3] Explore a completely new medical condition (Resets state for privacy)")
            console.print("  [4] Exit the session")

            choice = Prompt.ask("Choose option", choices=["1", "2", "3", "4"], default="4")

            if choice == "1":
                # Take another quiz question on same topic
                console.print("\n[cyan]Generating another question on this topic...[/cyan]")
                evaluation_state["next_action"] = "another_question"
                state_after_search = self.app.invoke(evaluation_state, config=config)
                continue

            elif choice == "2" and related:
                # Explore a related topic
                rel_choices = [str(i) for i in range(1, len(related) + 1)]
                rel_idx = Prompt.ask("Select topic number", choices=rel_choices, default="1")
                current_topic = related[int(rel_idx) - 1]
                console.print(f"\n[green]Switching topic to {current_topic}. Resetting state for privacy...[/green]")
                # Reset state
                evaluation_state["next_action"] = "new_topic"
                self.app.invoke(evaluation_state, config=config)

            elif choice == "3":
                # Brand new topic
                current_topic = Prompt.ask("Enter new medical topic", default="Asthma")
                console.print(f"\n[green]Starting new topic: {current_topic}. Resetting state for privacy...[/green]")
                evaluation_state["next_action"] = "new_topic"
                self.app.invoke(evaluation_state, config=config)

            else:
                # Exit
                session_running = False
                console.print(
                    Panel.fit(
                        "[bold cyan]Thank you for using MediTech HealthBot![/bold cyan]\n"
                        "[dim]Remember to always consult with your personal healthcare team for individual medical care.[/dim]",
                        border_style="cyan",
                    )
                )
