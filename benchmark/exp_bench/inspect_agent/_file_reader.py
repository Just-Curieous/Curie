from inspect_ai.tool import Tool, ToolError, tool
from inspect_ai.util import sandbox


@tool
def read_file_chunk():
    async def execute(file: str, start_line: int = 1, max_lines: int = 50) -> str:
        """Read a chunk of lines from a file.

        Args:
            file (str): Path to the file to read
            start_line (int): Line number to start reading from (1-indexed)
            max_lines (int): Maximum number of lines to read (default: 50, max: 50)

        Returns:
            str: The requested lines from the file

        Raises:
            ToolError: If the file cannot be read or if invalid line numbers are provided
        """
        if start_line < 1:
            raise ToolError("start_line must be >= 1")

        if max_lines < 1:
            raise ToolError("max_lines must be >= 1")

        if max_lines > 50:
            raise ToolError("max_lines cannot exceed 50")

        try:
            # Read the file
            content = await sandbox().read_file(file)

            # Split into lines
            lines = content.splitlines()

            if start_line > len(lines):
                raise ToolError(
                    f"start_line ({start_line}) is beyond the total number of lines ({len(lines)}) in the file."
                )
            # Calculate end line
            end_line = min(start_line + max_lines - 1, len(lines))

            # Get the requested chunk
            chunk = lines[start_line - 1 : end_line]

            # Add line numbers and join
            numbered_lines = [f"{i+start_line}: {line}" for i, line in enumerate(chunk)]

            # Add summary info
            total_lines = len(lines)
            summary = (
                f"File has {total_lines} total lines. Showing lines {start_line} to {end_line}.\n\n"
            )

            return summary + "\n".join(numbered_lines)

        except FileNotFoundError:
            raise ToolError(f"File '{file}' not found")
        except Exception as e:
            raise ToolError(f"Error reading file: {str(e)}")

    return execute


@tool
def search_file():
    async def execute(
        file: str, query: str, context_lines: int = 2, max_matches: int = 5, page: int = 1
    ) -> str:
        """Search for a keyword or phrase in a file and return matching lines with context.

        Args:
            file (str): Path to the file to search
            query (str): Text to search for (case-insensitive)
            context_lines (int): Number of lines of context to show before and after each match (default: 2)
            max_matches (int): Maximum number of matches to return per page (default: 5)
            page (int): Which page of results to return (1-indexed, default: 1)

        Returns:
            str: Matching lines with their line numbers and context, paginated

        Raises:
            ToolError: If the file cannot be read or if invalid parameters are provided
        """
        if not query:
            raise ToolError("Query cannot be empty.")
        if context_lines < 0:
            raise ToolError("context_lines must be >= 0")
        if max_matches < 1:
            raise ToolError("max_matches must be >= 1")
        if page < 1:
            raise ToolError("page must be >= 1")

        try:
            # Read the file
            content = await sandbox().read_file(file)

            # Split into lines
            lines = content.splitlines()

            # Find all matches (case-insensitive)
            all_matches = []
            query = query.lower()

            for i, line in enumerate(lines):
                if query in line.lower():
                    # Calculate context range
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)

                    # Get context lines
                    context = []
                    for j in range(start, end):
                        prefix = ">>> " if j == i else "    "  # Highlight matching line
                        context.append(f"{prefix}{j+1}: {lines[j]}")

                    all_matches.append("\n".join(context))

            if not all_matches:
                return f"No matches found for '{query}' in {file}"

            # Calculate pagination
            total_matches = len(all_matches)
            total_pages = (total_matches + max_matches - 1) // max_matches

            if page > total_pages:
                return f"Invalid page number. There are only {total_pages} pages of results."

            start_idx = (page - 1) * max_matches
            end_idx = min(start_idx + max_matches, total_matches)

            # Get matches for this page
            matches = all_matches[start_idx:end_idx]

            # Build summary with pagination info
            summary = [
                f"Found {total_matches} matches for '{query}' in {file}",
                f"Showing matches {start_idx + 1}-{end_idx} (Page {page} of {total_pages})",
                "",  # Empty line for spacing
            ]

            # Add match index to each result
            numbered_matches = []
            for i, match in enumerate(matches, start=start_idx + 1):
                numbered_matches.append(f"[Match {i} of {total_matches}]\n{match}")

            return "\n\n".join(summary + numbered_matches)

        except FileNotFoundError:
            raise ToolError(f"File '{file}' not found")
        except Exception as e:
            raise ToolError(f"Error searching file: {str(e)}")

    return execute
