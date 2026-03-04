import os


def create_structure():
    # -----------------------------
    # Folder structure (minimal but scalable)
    # -----------------------------
    folders = [
        os.path.join("app", "api", "routes"),
        os.path.join("app", "core"),
        os.path.join("app", "db", "models"),
        os.path.join("app", "services"),
        os.path.join("app", "graph"),
        os.path.join("app", "mcp"),
        os.path.join("app", "sandbox"),
    ]

    # -----------------------------
    # Files to create (EMPTY)
    # -----------------------------
    files = [
        "README.md",
        ".env.example",
        ".gitignore",
        os.path.join("app", "main.py"),
        os.path.join("app", "api", "deps.py"),
        os.path.join("app", "api", "routes", "webhook.py"),
        os.path.join("app", "api", "routes", "review.py"),
        os.path.join("app", "api", "routes", "chat.py"),
        os.path.join("app", "core", "config.py"),
        os.path.join("app", "core", "logging.py"),
        os.path.join("app", "core", "exceptions.py"),
        os.path.join("app", "db", "session.py"),
        os.path.join("app", "db", "base.py"),
        os.path.join("app", "db", "models", "repository.py"),
        os.path.join("app", "db", "models", "pull_request.py"),
        os.path.join("app", "db", "models", "review.py"),
        os.path.join("app", "db", "models", "thread.py"),
        os.path.join("app", "db", "models", "message.py"),
        os.path.join("app", "services", "review_service.py"),
        os.path.join("app", "services", "chat_service.py"),
        os.path.join("app", "services", "repository_service.py"),
        os.path.join("app", "graph", "state.py"),
        os.path.join("app", "graph", "nodes.py"),
        os.path.join("app", "graph", "workflow.py"),
        os.path.join("app", "mcp", "github_client.py"),
        os.path.join("app", "mcp", "filesystem_client.py"),
        os.path.join("app", "mcp", "sandbox_client.py"),
        os.path.join("app", "sandbox", "docker_runner.py"),
    ]

    # -----------------------------
    # Create folders
    # -----------------------------
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # -----------------------------
    # Create __init__.py in all app subfolders
    # -----------------------------
    for root, dirs, _ in os.walk("app"):
        init_file = os.path.join(root, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w"):
                pass

    # -----------------------------
    # Create files (empty)
    # -----------------------------
    for file_path in files:
        if not os.path.exists(file_path):
            with open(file_path, "w"):
                pass
            print(f"Created: {file_path}")

    print("\n✅ Project structure created successfully.")


if __name__ == "__main__":
    create_structure()