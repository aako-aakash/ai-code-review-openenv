"""
Predefined task catalogue for the AI Code Review environment.
Each task has a code snippet, ground-truth bugs, expected fixes, and
expected improvements.  The grader uses this ground truth to score
agent behaviour deterministically.
"""

from env.models import BugRecord, FixRecord, ImprovementRecord, TaskDefinition

# ---------------------------------------------------------------------------
# TASK 1 – Easy (syntax bug)
# ---------------------------------------------------------------------------

TASK_1 = TaskDefinition(
    task_id="task_1",
    title="Fix the Off-by-One Syntax Bug",
    difficulty="easy",
    goal=(
        "Review the Python function below. "
        "Detect the syntax/logic bug, suggest the correct fix, "
        "and approve or request changes when done."
    ),
    code_snippet='''\
def calculate_average(numbers):
    """Return the average of a list of numbers."""
    total = 0
    for i in range(len(numbers) + 1):   # BUG: off-by-one, should be range(len(numbers))
        total += numbers[i]
    return total / len(numbers)

# Example usage
data = [10, 20, 30, 40, 50]
print(calculate_average(data))
''',
    bugs=[
        BugRecord(
            bug_id="bug_1_a",
            description="Off-by-one error in range: range(len(numbers) + 1) causes IndexError",
            line_number=4,
            bug_type="syntax",
        )
    ],
    fixes=[
        FixRecord(
            bug_id="bug_1_a",
            description="Change range(len(numbers) + 1) to range(len(numbers))",
            correct_fix="range(len(numbers))",
        )
    ],
    improvements=[
        ImprovementRecord(
            improvement_id="imp_1_a",
            description="Use Python built-in sum() instead of manual loop for clarity",
            keywords=["sum", "built-in", "pythonic"],
        )
    ],
    max_steps=15,
)

# ---------------------------------------------------------------------------
# TASK 2 – Medium (logic bug + performance)
# ---------------------------------------------------------------------------

TASK_2 = TaskDefinition(
    task_id="task_2",
    title="Fix Logic Bug and Optimize Search",
    difficulty="medium",
    goal=(
        "Review the function that searches for duplicate entries. "
        "Detect the logic bug (wrong comparison operator) and the "
        "performance issue (O(n²) when a set would be O(n)), "
        "suggest fixes for both, then close the review."
    ),
    code_snippet='''\
def find_duplicates(items):
    """Return a list of duplicate items."""
    duplicates = []
    for i in range(len(items)):
        for j in range(len(items)):          # BUG: should start j from i+1
            if i != j and items[i] == items[j]:
                if items[i] not in duplicates:
                    duplicates.append(items[i])
    return duplicates

# PERFORMANCE ISSUE: O(n^2) – could use a set/Counter for O(n)

sample = [1, 2, 3, 2, 4, 3, 5]
print(find_duplicates(sample))
''',
    bugs=[
        BugRecord(
            bug_id="bug_2_a",
            description=(
                "Inner loop starts at 0 instead of i+1, "
                "leading to redundant comparisons and false duplicates "
                "when items[i] == items[i] (same index)."
            ),
            line_number=5,
            bug_type="logic",
        ),
        BugRecord(
            bug_id="bug_2_b",
            description=(
                "O(n²) nested loop – performance issue for large inputs; "
                "a Counter or set-based approach is O(n)."
            ),
            line_number=4,
            bug_type="performance",
        ),
    ],
    fixes=[
        FixRecord(
            bug_id="bug_2_a",
            description="Change inner range to start from i+1",
            correct_fix="range(i+1, len(items))",
        ),
        FixRecord(
            bug_id="bug_2_b",
            description="Replace nested loops with collections.Counter",
            correct_fix="counter",
        ),
    ],
    improvements=[
        ImprovementRecord(
            improvement_id="imp_2_a",
            description="Use collections.Counter for O(n) duplicate detection",
            keywords=["counter", "collections", "o(n)", "linear"],
        ),
        ImprovementRecord(
            improvement_id="imp_2_b",
            description="Add type hints to the function signature",
            keywords=["type hint", "typing", "list[", "-> list"],
        ),
    ],
    max_steps=20,
)

# ---------------------------------------------------------------------------
# TASK 3 – Hard (multi-bug + security + performance)
# ---------------------------------------------------------------------------

TASK_3 = TaskDefinition(
    task_id="task_3",
    title="Multi-Bug: SQL Injection, Logic Error, and N+1 Query",
    difficulty="hard",
    goal=(
        "Review the database query helper below. "
        "Detect: (1) SQL injection vulnerability, "
        "(2) logic error in pagination offset calculation, "
        "(3) N+1 query anti-pattern. "
        "Suggest fixes for all three, then request changes."
    ),
    code_snippet='''\
import sqlite3

def get_user_posts(db_path, username, page=1, page_size=10):
    """Fetch paginated posts for a given username."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # BUG 1 – SQL Injection: user input directly interpolated
    query = f"SELECT id, title FROM posts WHERE author = \'{username}\'"
    cursor.execute(query)
    post_rows = cursor.fetchall()

    # BUG 2 – Logic error: offset should be (page-1)*page_size, not page*page_size
    offset = page * page_size
    paginated = post_rows[offset : offset + page_size]

    # BUG 3 – N+1 query: fetching comments inside a loop
    results = []
    for post_id, title in paginated:
        cursor.execute(f"SELECT body FROM comments WHERE post_id = {post_id}")
        comments = cursor.fetchall()
        results.append({"title": title, "comments": comments})

    conn.close()
    return results
''',
    bugs=[
        BugRecord(
            bug_id="bug_3_a",
            description=(
                "SQL Injection: username is interpolated directly into the query "
                "string. Attacker can manipulate SQL logic."
            ),
            line_number=9,
            bug_type="security",
        ),
        BugRecord(
            bug_id="bug_3_b",
            description=(
                "Wrong pagination offset: offset = page * page_size should be "
                "offset = (page - 1) * page_size. Page 1 skips the first page."
            ),
            line_number=14,
            bug_type="logic",
        ),
        BugRecord(
            bug_id="bug_3_c",
            description=(
                "N+1 query anti-pattern: a separate SELECT is issued for every post "
                "in the loop, causing N extra DB round-trips."
            ),
            line_number=18,
            bug_type="performance",
        ),
    ],
    fixes=[
        FixRecord(
            bug_id="bug_3_a",
            description="Use parameterised query: cursor.execute(query, (username,))",
            correct_fix="parameterised",
        ),
        FixRecord(
            bug_id="bug_3_b",
            description="Change offset to (page - 1) * page_size",
            correct_fix="(page - 1)",
        ),
        FixRecord(
            bug_id="bug_3_c",
            description="Use a JOIN query to fetch posts and comments in one round-trip",
            correct_fix="join",
        ),
    ],
    improvements=[
        ImprovementRecord(
            improvement_id="imp_3_a",
            description="Use a context manager (with statement) for the DB connection",
            keywords=["context manager", "with statement", "with conn", "with sqlite"],
        ),
        ImprovementRecord(
            improvement_id="imp_3_b",
            description="Add LIMIT/OFFSET directly in SQL rather than Python slicing",
            keywords=["limit", "offset", "sql limit", "sql offset"],
        ),
    ],
    max_steps=25,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, TaskDefinition] = {
    "task_1": TASK_1,
    "task_2": TASK_2,
    "task_3": TASK_3,
}
