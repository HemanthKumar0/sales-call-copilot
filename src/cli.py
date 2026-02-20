"""CLI interface for Sales Call Copilot with Rich formatting."""

import logging
import re

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import get_settings, setup_logging
from src.ingestion import IngestionEngine
from src.prompts import QA_PROMPT
from src.retriever import SalesCallRetriever
from src.storage import MetadataStore, get_vector_store

logger = logging.getLogger(__name__)
console = Console()


class CommandParser:
    """Parse user input into structured commands or fallback to RAG query."""

    # Pattern: "summarise call <id>" or "summarize call <id>"
    SUMMARIZE_CALL_PATTERN = re.compile(
        r"^summari[sz]e\s+call\s+(.+)$", re.IGNORECASE
    )
    # Pattern: "summarise the last call" or "summarize the last call"
    SUMMARIZE_LAST_PATTERN = re.compile(
        r"^summari[sz]e\s+the\s+last\s+call$", re.IGNORECASE
    )
    # Pattern: "ingest <filepath>"
    INGEST_PATTERN = re.compile(r"^ingest\s+(.+)$", re.IGNORECASE)

    def parse(self, user_input: str) -> tuple[str, dict]:
        """Parse user input into (command_type, params).

        Returns one of:
        - ('list', {})
        - ('summarize_last', {})
        - ('summarize_call', {'call_id': '<id>'})
        - ('ingest', {'filepath': '<filepath>'})
        - ('exit', {})
        - ('query', {'query': '<original input>'})
        """
        text = user_input.strip()
        if not text:
            return ("query", {"query": ""})

        lower = text.lower()

        # Exit commands
        if lower in ("exit", "quit", "bye"):
            return ("exit", {})

        # List commands
        if lower in ("list", "list my call ids"):
            return ("list", {})

        # Summarize last call (check before summarize_call to avoid false match)
        if self.SUMMARIZE_LAST_PATTERN.match(text):
            return ("summarize_last", {})

        # Summarize specific call
        match = self.SUMMARIZE_CALL_PATTERN.match(text)
        if match:
            call_id = match.group(1).strip()
            return ("summarize_call", {"call_id": call_id})

        # Ingest command
        match = self.INGEST_PATTERN.match(text)
        if match:
            filepath = match.group(1).strip()
            return ("ingest", {"filepath": filepath})

        # Fallback: treat as RAG query
        return ("query", {"query": text})


class SalesCallCLI:
    """Main CLI application wiring all components together."""

    def __init__(self) -> None:
        """Initialize all components: config, stores, retriever, ingestion engine."""
        self.parser = CommandParser()
        self.settings = get_settings()
        self.metadata_store = MetadataStore(self.settings.metadata_file)
        self.vector_store = get_vector_store(self.settings)
        self.ingestion_engine = IngestionEngine(
            self.vector_store, self.metadata_store, self.settings
        )
        self.retriever = SalesCallRetriever(self.vector_store, self.settings)

    def run(self) -> None:
        """Main loop: prompt, parse, execute, display."""
        console.print(
            Panel(
                "[bold green]Sales Call Copilot[/bold green]\n"
                "Type a command or ask a question. Type [bold]exit[/bold] to quit.",
                title="Welcome",
            )
        )
        try:
            while True:
                try:
                    user_input = console.input("[bold cyan]You>[/bold cyan] ")
                except EOFError:
                    break

                command, params = self.parser.parse(user_input)

                if command == "exit":
                    console.print("[bold green]Goodbye! 👋[/bold green]")
                    break

                try:
                    if command == "list":
                        self._handle_list()
                    elif command == "summarize_last":
                        self._handle_summarize_last()
                    elif command == "summarize_call":
                        self._handle_summarize_call(params["call_id"])
                    elif command == "ingest":
                        self._handle_ingest(params["filepath"])
                    elif command == "query":
                        query = params.get("query", "")
                        if query:
                            self._handle_query(query)
                except Exception as exc:
                    logger.error("Error handling command: %s", exc)
                    console.print(
                        Panel(
                            f"[bold red]Error:[/bold red] {exc}",
                            title="Error",
                            border_style="red",
                        )
                    )
        except KeyboardInterrupt:
            console.print("\n[bold green]Goodbye! Thanks for using Sales Call Copilot! 👋[/bold green]")

    def _handle_list(self) -> None:
        """Display all call IDs in a Rich table."""
        calls = self.metadata_store.load_all()
        if not calls:
            console.print(
                Panel(
                    "No calls ingested yet. Use [bold]ingest <filepath>[/bold] to add a transcript.",
                    title="No Calls",
                    border_style="yellow",
                )
            )
            return

        table = Table(title="Ingested Calls")
        table.add_column("Call ID", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("Date", style="yellow")
        table.add_column("Filename", style="dim")

        for call in calls:
            table.add_row(call.call_id, call.title, call.date, call.filename)

        console.print(table)

    def _handle_summarize_last(self) -> None:
        """Summarize the most recently ingested call."""
        last_call = self.metadata_store.get_last()
        if not last_call:
            console.print(
                Panel(
                    "No calls ingested yet.",
                    title="No Calls",
                    border_style="yellow",
                )
            )
            return

        console.print(f"[dim]Summarizing call: {last_call.call_id} ({last_call.title})...[/dim]")
        summary = self.retriever.summarize_call(last_call.call_id)
        console.print(
            Panel(summary, title=f"Summary: {last_call.title}", border_style="green")
        )

    def _handle_summarize_call(self, call_id: str) -> None:
        """Summarize a specific call by ID."""
        call = self.metadata_store.get_by_id(call_id)
        if not call:
            console.print(
                Panel(
                    f"Call ID [bold]{call_id}[/bold] not found.",
                    title="Not Found",
                    border_style="red",
                )
            )
            return

        console.print(f"[dim]Summarizing call: {call.call_id} ({call.title})...[/dim]")
        summary = self.retriever.summarize_call(call.call_id)
        console.print(
            Panel(summary, title=f"Summary: {call.title}", border_style="green")
        )

    def _handle_ingest(self, filepath: str) -> None:
        """Ingest a new transcript file."""
        console.print(f"[dim]Ingesting {filepath}...[/dim]")
        call_metadata = self.ingestion_engine.ingest_file(filepath)
        console.print(
            Panel(
                f"[bold green]Successfully ingested![/bold green]\n"
                f"Call ID: [cyan]{call_metadata.call_id}[/cyan]\n"
                f"Title: {call_metadata.title}\n"
                f"Date: {call_metadata.date}",
                title="Ingestion Complete",
                border_style="green",
            )
        )

    def _handle_query(self, query: str) -> None:
        """Execute a free-form RAG query."""
        console.print("[dim]Searching transcripts...[/dim]")
        response = self.retriever.query_with_llm(query, QA_PROMPT)
        console.print(
            Panel(response, title="Answer", border_style="blue")
        )


def main() -> None:
    """Entry point for the CLI application."""
    setup_logging()
    cli = SalesCallCLI()
    cli.run()


if __name__ == "__main__":
    main()
