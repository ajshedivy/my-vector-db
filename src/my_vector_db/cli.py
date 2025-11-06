#!/usr/bin/env python3
"""
Interactive CLI for Vector Database Management

An interactive command-line interface for managing libraries, documents,
and chunks in the vector database.

Usage:
    python cli.py
    python cli.py --url http://localhost:8000

Commands:
    /help                           - Show all commands
    /list_libraries                 - List all libraries
    /list_docs --library <name>     - List documents in library
    /list_chunks --document <id>    - List chunks in document
    /create_library --name <name>   - Create new library
    /delete_library --id <id>       - Delete library
    /search --library <id> --k <n>  - Search for similar vectors
    /exit                           - Exit the CLI
"""

import sys
import shlex

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.document import Document
except ImportError:
    print("Error: prompt_toolkit is required. Install with: pip install prompt-toolkit")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    print("Error: rich is required. Install with: pip install rich")
    sys.exit(1)

from my_vector_db.sdk import (
    VectorDBClient,
    VectorDBError,
    NotFoundError,
    ValidationError,
)


class VectorDBCompleter(Completer):
    """Custom completer with context-aware command and option suggestions."""

    def __init__(self, commands, command_options):
        """Initialize completer with commands and their valid options."""
        self.commands = commands
        self.command_options = command_options

    def get_completions(self, document: Document, complete_event):
        """Get completions for the current input."""
        text = document.text_before_cursor
        words = text.split()

        # If we're at the start or typing a command (starts with /)
        if not words or (len(words) == 1 and not text.endswith(" ")):
            word = words[0] if words else ""
            # Filter commands that match
            for cmd in self.commands:
                if cmd.lower().startswith(word.lower()):
                    yield Completion(
                        cmd,
                        start_position=-len(word),
                        display=cmd,
                    )
        else:
            # Get the command to determine valid options
            command = words[0] if words else ""
            valid_options = self.command_options.get(command, [])

            # Find already used options (those starting with --)
            used_options = {word for word in words if word.startswith("--")}

            # We're typing options (--flag)
            current_word = words[-1] if not text.endswith(" ") else ""
            if current_word.startswith("--") or not current_word:
                for opt in valid_options:
                    # Skip options that have already been used
                    if opt in used_options:
                        continue
                    if opt.lower().startswith(current_word.lower()):
                        yield Completion(
                            opt,
                            start_position=-len(current_word),
                            display=opt,
                        )


class VectorDBCLI:
    """Interactive CLI for Vector Database."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize CLI with database client."""
        self.console = Console()
        self.client = VectorDBClient(base_url=base_url)
        self.history = InMemoryHistory()

        # Command list for autocomplete
        self.commands = [
            "/help",
            "/exit",
            "/quit",
            "/clear",
            "/list_libraries",
            "/list_docs",
            "/list_chunks",
            "/create_library",
            "/create_document",
            "/delete_library",
            "/delete_document",
            "/delete_chunk",
            "/update_library",
            "/update_document",
            "/update_chunk",
            "/get_library",
            "/get_document",
            "/get_chunk",
            "/search",
            "/build_index",
            "/save_snapshot",
            "/restore_snapshot",
            "/status",
        ]

        # Map commands to their valid options
        self.command_options = {
            # Library commands
            "/list_libraries": [],
            "/get_library": ["--id"],
            "/create_library": ["--name", "--index_type"],
            "/update_library": ["--id", "--name", "--index_type"],
            "/delete_library": ["--id"],
            "/build_index": ["--library"],
            # Document commands
            "/list_docs": ["--library"],
            "/get_document": ["--id"],
            "/create_document": ["--library", "--name"],
            "/update_document": ["--id", "--name"],
            "/delete_document": ["--id"],
            # Chunk commands
            "/list_chunks": ["--document"],
            "/get_chunk": ["--id"],
            "/update_chunk": ["--id", "--text", "--embedding"],
            "/delete_chunk": ["--id"],
            # Search
            "/search": ["--library", "--embedding", "--k"],
            # Persistence
            "/save_snapshot": [],
            "/restore_snapshot": [],
            "/status": [],
            # General
            "/help": [],
            "/exit": [],
            "/quit": [],
            "/clear": [],
        }

        self.completer = VectorDBCompleter(self.commands, self.command_options)

    def show_banner(self):
        """Display welcome banner."""
        banner = Panel.fit(
            "[bold cyan]Vector Database CLI[/bold cyan]\n"
            "Type [yellow]/help[/yellow] for available commands\n"
            "Type [yellow]/exit[/yellow] to quit",
            border_style="cyan",
        )
        self.console.print(banner)

    def show_help(self):
        """Display help information."""
        table = Table(
            title="Available Commands", show_header=True, header_style="bold magenta"
        )
        table.add_column("Command", style="cyan", width=40)
        table.add_column("Description", style="white")

        commands_help = [
            ("/help", "Show this help message"),
            ("/exit, /quit", "Exit the CLI (or Ctrl+C)"),
            ("/clear", "Clear the screen"),
            ("", ""),
            ("[bold]Libraries[/bold]", ""),
            ("/list_libraries", "List all libraries"),
            ("/get_library --id <uuid>", "Get library details"),
            (
                "/create_library --name <name> [--index_type flat|hnsw]",
                "Create new library",
            ),
            ("/update_library --id <uuid> --name <name>", "Update library"),
            ("/delete_library --id <uuid>", "Delete library"),
            ("/build_index --library <uuid>", "Build vector index"),
            ("", ""),
            ("[bold]Documents[/bold]", ""),
            ("/list_docs --library <uuid>", "List documents in library"),
            ("/get_document --id <uuid>", "Get document details"),
            ("/create_document --library <uuid> --name <name>", "Create document"),
            ("/update_document --id <uuid> --name <name>", "Update document"),
            ("/delete_document --id <uuid>", "Delete document"),
            ("", ""),
            ("[bold]Chunks[/bold]", ""),
            ("/list_chunks --document <uuid>", "List chunks in document"),
            ("/get_chunk --id <uuid>", "Get chunk details"),
            ("/update_chunk --id <uuid> --text <text>", "Update chunk text"),
            ("/delete_chunk --id <uuid>", "Delete chunk"),
            ("", ""),
            ("[bold]Search & Persistence[/bold]", ""),
            (
                "/search --library <uuid> --embedding <v1,v2,v3> --k <n>",
                "Search for similar vectors",
            ),
            ("/save_snapshot", "Save database snapshot"),
            ("/restore_snapshot", "Restore from snapshot"),
            ("/status", "Show database status"),
        ]

        for cmd, desc in commands_help:
            table.add_row(cmd, desc)

        self.console.print(table)

    def parse_args(self, line: str) -> tuple[str, dict[str, str | bool]]:
        """Parse command line into command and arguments."""
        try:
            parts = shlex.split(line)
        except ValueError as e:
            raise ValueError(f"Invalid command syntax: {e}")

        if not parts:
            return "", {}

        command = parts[0]
        args: dict[str, str | bool] = {}

        i = 1
        while i < len(parts):
            if parts[i].startswith("--"):
                key = parts[i][2:]
                if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                    args[key] = parts[i + 1]
                    i += 2
                else:
                    args[key] = True
                    i += 1
            else:
                i += 1

        return command, args

    def cmd_list_libraries(self, args: dict):
        """List all libraries."""
        libraries = self.client.list_libraries()

        if not libraries:
            self.console.print("[yellow]No libraries found[/yellow]")
            return

        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("ID", style="cyan", no_wrap=True, min_width=36)
        table.add_column("Name", style="green", no_wrap=True)
        table.add_column("Index Type", style="blue", no_wrap=True)
        table.add_column("Documents", style="yellow", justify="right")
        table.add_column("Created", style="white", no_wrap=True)

        for lib in libraries:
            table.add_row(
                str(lib.id),
                lib.name,
                lib.index_type,
                str(len(lib.document_ids)),
                lib.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        self.console.print(table)

    def cmd_list_docs(self, args: dict):
        """List documents in a library."""
        library_id = args.get("library")
        if not library_id:
            self.console.print("[red]Error: --library <uuid> required[/red]")
            return

        try:
            documents = self.client.list_documents(library_id=library_id)

            if not documents:
                self.console.print("[yellow]No documents found[/yellow]")
                return

            table = Table(show_header=True, header_style="bold magenta", expand=True)
            table.add_column("ID", style="cyan", no_wrap=True, min_width=36)
            table.add_column("Name", style="green", no_wrap=True)
            table.add_column("Chunks", style="yellow", justify="right")
            table.add_column("Created", style="white", no_wrap=True)

            for doc in documents:
                table.add_row(
                    str(doc.id),
                    doc.name,
                    str(len(doc.chunk_ids)),
                    doc.created_at.strftime("%Y-%m-%d %H:%M"),
                )

            self.console.print(table)

        except NotFoundError:
            self.console.print(f"[red]Library {library_id} not found[/red]")

    def cmd_list_chunks(self, args: dict):
        """List chunks in a document."""
        document_id = args.get("document")
        if not document_id:
            self.console.print("[red]Error: --document <uuid> required[/red]")
            return

        try:
            chunks = self.client.list_chunks(document_id=document_id)

            if not chunks:
                self.console.print("[yellow]No chunks found[/yellow]")
                return

            table = Table(show_header=True, header_style="bold magenta", expand=True)
            table.add_column("ID", style="cyan", no_wrap=True, min_width=36)
            table.add_column("Text", style="green", max_width=60)
            table.add_column("Embedding Dim", style="blue", justify="right")
            table.add_column("Created", style="white", no_wrap=True)

            for chunk in chunks:
                table.add_row(
                    str(chunk.id),
                    chunk.text[:57] + "..." if len(chunk.text) > 60 else chunk.text,
                    str(len(chunk.embedding)),
                    chunk.created_at.strftime("%Y-%m-%d %H:%M"),
                )

            self.console.print(table)

        except NotFoundError:
            self.console.print(f"[red]Document {document_id} not found[/red]")

    def cmd_create_library(self, args: dict):
        """Create a new library."""
        name = args.get("name")
        if not name:
            self.console.print("[red]Error: --name <name> required[/red]")
            return

        index_type = args.get("index_type", "flat")

        try:
            library = self.client.create_library(name=name, index_type=index_type)
            self.console.print(
                f"[green]✓ Created library: {library.name} (ID: {library.id})[/green]"
            )
        except ValidationError as e:
            self.console.print(f"[red]Validation error: {e}[/red]")

    def cmd_create_document(self, args: dict):
        """Create a new document."""
        library_id = args.get("library")
        name = args.get("name")

        if not library_id or not name:
            self.console.print(
                "[red]Error: --library <uuid> --name <name> required[/red]"
            )
            return

        try:
            document = self.client.create_document(library_id=library_id, name=name)
            self.console.print(
                f"[green]✓ Created document: {document.name} (ID: {document.id})[/green]"
            )
        except NotFoundError:
            self.console.print(f"[red]Library {library_id} not found[/red]")
        except ValidationError as e:
            self.console.print(f"[red]Validation error: {e}[/red]")

    def cmd_delete_chunk(self, args: dict):
        """Delete a chunk."""
        chunk_id = args.get("id")
        if not chunk_id:
            self.console.print("[red]Error: --id <uuid> required[/red]")
            return

        try:
            self.client.delete_chunk(chunk_id=chunk_id)
            self.console.print(f"[green]✓ Deleted chunk {chunk_id}[/green]")
        except NotFoundError:
            self.console.print(f"[red]Chunk {chunk_id} not found[/red]")

    def cmd_delete_document(self, args: dict):
        """Delete a document."""
        doc_id = args.get("id")
        if not doc_id:
            self.console.print("[red]Error: --id <uuid> required[/red]")
            return

        try:
            self.client.delete_document(document_id=doc_id)
            self.console.print(f"[green]✓ Deleted document {doc_id}[/green]")
        except NotFoundError:
            self.console.print(f"[red]Document {doc_id} not found[/red]")

    def cmd_delete_library(self, args: dict):
        """Delete a library."""
        lib_id = args.get("id")
        if not lib_id:
            self.console.print("[red]Error: --id <uuid> required[/red]")
            return

        try:
            self.client.delete_library(library_id=lib_id)
            self.console.print(f"[green]✓ Deleted library {lib_id}[/green]")
        except NotFoundError:
            self.console.print(f"[red]Library {lib_id} not found[/red]")

    def cmd_update_chunk(self, args: dict):
        """Update a chunk."""
        chunk_id = args.get("id")
        text = args.get("text")

        if not chunk_id:
            self.console.print("[red]Error: --id <uuid> required[/red]")
            return

        try:
            chunk = self.client.update_chunk(chunk_id, text=text)
            self.console.print(f"[green]✓ Updated chunk {chunk_id}[/green]")
            if text:
                self.console.print(f"  New text: {chunk.text[:60]}...")
        except NotFoundError:
            self.console.print(f"[red]Chunk {chunk_id} not found[/red]")

    def cmd_get_library(self, args: dict):
        """Get library details."""
        lib_id = args.get("id")
        if not lib_id:
            self.console.print("[red]Error: --id <uuid> required[/red]")
            return

        try:
            library = self.client.get_library(library_id=lib_id)

            table = Table(show_header=False, box=None)
            table.add_column("Field", style="cyan", width=20)
            table.add_column("Value", style="white")

            table.add_row("ID", str(library.id))
            table.add_row("Name", library.name)
            table.add_row("Index Type", library.index_type)
            table.add_row("Index Config", str(library.index_config))
            table.add_row("Documents", str(len(library.document_ids)))
            table.add_row("Created", library.created_at.strftime("%Y-%m-%d %H:%M:%S"))
            table.add_row("Updated", library.updated_at.strftime("%Y-%m-%d %H:%M:%S"))

            self.console.print(
                Panel(
                    table,
                    title=f"[bold]Library: {library.name}[/bold]",
                    border_style="cyan",
                )
            )

        except NotFoundError:
            self.console.print(f"[red]Library {lib_id} not found[/red]")

    def cmd_get_document(self, args: dict):
        """Get document details."""
        doc_id = args.get("id")
        if not doc_id:
            self.console.print("[red]Error: --id <uuid> required[/red]")
            return

        try:
            document = self.client.get_document(document_id=doc_id)

            table = Table(show_header=False, box=None)
            table.add_column("Field", style="cyan", width=20)
            table.add_column("Value", style="white")

            table.add_row("ID", str(document.id))
            table.add_row("Name", document.name)
            table.add_row("Library ID", str(document.library_id))
            table.add_row("Chunks", str(len(document.chunk_ids)))
            table.add_row("Created", document.created_at.strftime("%Y-%m-%d %H:%M:%S"))
            table.add_row("Updated", document.updated_at.strftime("%Y-%m-%d %H:%M:%S"))

            self.console.print(
                Panel(
                    table,
                    title=f"[bold]Document: {document.name}[/bold]",
                    border_style="cyan",
                )
            )

        except NotFoundError:
            self.console.print(f"[red]Document {doc_id} not found[/red]")

    def cmd_get_chunk(self, args: dict):
        """Get chunk details."""
        chunk_id = args.get("id")
        if not chunk_id:
            self.console.print("[red]Error: --id <uuid> required[/red]")
            return

        try:
            chunk = self.client.get_chunk(chunk_id=chunk_id)

            table = Table(show_header=False, box=None)
            table.add_column("Field", style="cyan", width=20)
            table.add_column("Value", style="white")

            table.add_row("ID", str(chunk.id))
            table.add_row("Document ID", str(chunk.document_id))
            table.add_row("Text", chunk.text)
            table.add_row("Embedding Dim", str(len(chunk.embedding)))
            table.add_row(
                "Embedding",
                str(chunk.embedding[:5]) + "..."
                if len(chunk.embedding) > 5
                else str(chunk.embedding),
            )

            if chunk.metadata:
                table.add_row("Metadata", str(chunk.metadata))

            table.add_row("Created", chunk.created_at.strftime("%Y-%m-%d %H:%M:%S"))
            table.add_row("Updated", chunk.updated_at.strftime("%Y-%m-%d %H:%M:%S"))

            self.console.print(
                Panel(table, title="[bold]Chunk Details[/bold]", border_style="cyan")
            )

        except NotFoundError:
            self.console.print(f"[red]Chunk {chunk_id} not found[/red]")

    def cmd_status(self, args: dict):
        """Show database status."""
        try:
            status = self.client.get_persistence_status()

            table = Table(show_header=False, box=None)
            table.add_column("Field", style="cyan", width=25)
            table.add_column("Value", style="white")

            table.add_row(
                "Persistence Enabled", "✓ Yes" if status["enabled"] else "✗ No"
            )
            if status["enabled"]:
                table.add_row("Storage Directory", status["storage_dir"])
                table.add_row("Save Threshold", str(status["save_threshold"]))
                table.add_row(
                    "Operations Since Save", str(status["operations_since_save"])
                )
                table.add_row(
                    "Snapshot Exists", "✓ Yes" if status["snapshot_exists"] else "✗ No"
                )

            table.add_row("", "")
            table.add_row("Libraries", str(status["current_stats"]["libraries"]))
            table.add_row("Documents", str(status["current_stats"]["documents"]))
            table.add_row("Chunks", str(status["current_stats"]["chunks"]))

            self.console.print(
                Panel(table, title="[bold]Database Status[/bold]", border_style="green")
            )

        except Exception as e:
            self.console.print(f"[red]Error getting status: {e}[/red]")

    def cmd_search(self, args: dict):
        """Search for similar vectors."""
        library_id = args.get("library")
        embedding_str = args.get("embedding")
        k = int(args.get("k", 10))

        if not library_id or not embedding_str:
            self.console.print(
                "[red]Error: --library <uuid> --embedding <v1,v2,v3> required[/red]"
            )
            return

        try:
            # Parse embedding
            embedding = [float(x.strip()) for x in embedding_str.split(",")]

            results = self.client.search(
                library_id=library_id, embedding=embedding, k=k
            )

            if not results.results:
                self.console.print("[yellow]No results found[/yellow]")
                return

            table = Table(show_header=True, header_style="bold magenta", expand=True)
            table.add_column("#", style="cyan", width=4, justify="right")
            table.add_column("Score", style="yellow", width=10, justify="right")
            table.add_column("Text", style="green", max_width=60)
            table.add_column("Chunk ID", style="blue", no_wrap=True, min_width=36)

            for i, result in enumerate(results.results, 1):
                table.add_row(
                    str(i),
                    f"{result.score:.4f}",
                    result.text[:57] + "..." if len(result.text) > 60 else result.text,
                    str(result.chunk_id),
                )

            self.console.print(table)
            self.console.print(f"[dim]Query time: {results.query_time_ms:.2f}ms[/dim]")

        except NotFoundError:
            self.console.print(f"[red]Library {library_id} not found[/red]")
        except ValueError as e:
            self.console.print(f"[red]Invalid embedding format: {e}[/red]")

    def execute_command(self, line: str):
        """Execute a command."""
        try:
            command, args = self.parse_args(line)

            if not command:
                return True

            # Route commands
            if command in ["/exit", "/quit"]:
                return False
            elif command == "/clear":
                self.console.clear()
            elif command == "/help":
                self.show_help()
            elif command == "/list_libraries":
                self.cmd_list_libraries(args)
            elif command == "/list_docs":
                self.cmd_list_docs(args)
            elif command == "/list_chunks":
                self.cmd_list_chunks(args)
            elif command == "/create_library":
                self.cmd_create_library(args)
            elif command == "/create_document":
                self.cmd_create_document(args)
            elif command == "/delete_chunk":
                self.cmd_delete_chunk(args)
            elif command == "/delete_document":
                self.cmd_delete_document(args)
            elif command == "/delete_library":
                self.cmd_delete_library(args)
            elif command == "/update_chunk":
                self.cmd_update_chunk(args)
            elif command == "/get_library":
                self.cmd_get_library(args)
            elif command == "/get_document":
                self.cmd_get_document(args)
            elif command == "/get_chunk":
                self.cmd_get_chunk(args)
            elif command == "/status":
                self.cmd_status(args)
            elif command == "/search":
                self.cmd_search(args)
            elif command == "/save_snapshot":
                result = self.client.save_snapshot()
                self.console.print(
                    f"[green]✓ Snapshot saved: {result['snapshot_path']}[/green]"
                )
            elif command == "/restore_snapshot":
                result = self.client.restore_snapshot()
                self.console.print("[green]✓ Restored from snapshot[/green]")
            else:
                self.console.print(f"[red]Unknown command: {command}[/red]")
                self.console.print("Type [yellow]/help[/yellow] for available commands")

        except VectorDBError as e:
            self.console.print(f"[red]Database error: {e}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")

        return True

    def run(self):
        """Run the interactive CLI."""
        self.show_banner()

        session = PromptSession(
            completer=self.completer,
            history=self.history,
        )

        try:
            while True:
                try:
                    line = session.prompt("my_vector_db> ")
                    if line.strip():
                        if not self.execute_command(line.strip()):
                            break
                except KeyboardInterrupt:
                    # Ctrl+C exits the CLI
                    self.console.print(
                        "\n[yellow]Interrupt received. Use /exit to quit or continue typing...[/yellow]"
                    )
                    confirm = session.prompt("Exit? (y/N): ")
                    if confirm.lower() in ["y", "yes"]:
                        break
                    continue
                except EOFError:
                    # Ctrl+D exits
                    break

        finally:
            self.console.print("\n[cyan]Goodbye![/cyan]")
            self.client.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Interactive Vector Database CLI")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    cli = VectorDBCLI(base_url=args.url)
    cli.run()


if __name__ == "__main__":
    main()
