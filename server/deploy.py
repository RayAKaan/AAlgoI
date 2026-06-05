import os
import subprocess


def deploy():
    token = os.environ.get('AALGOI_GITHUB_TOKEN')
    if not token:
        raise RuntimeError("AALGOI_GITHUB_TOKEN not set")

    subprocess.run([
        "flyctl", "secrets", "set",
        f"AALGOI_GITHUB_TOKEN={token}",
        "--app", "aalgoi-federation"
    ], check=True)

    subprocess.run([
        "flyctl", "deploy",
        "--app", "aalgoi-federation",
        "--region", "iad",
        "--strategy", "rolling",
    ], check=True)

    print("✅ Deployed to: https://aalgoi-federation.fly.dev")
    print("   Health: https://aalgoi-federation.fly.dev/health")


if __name__ == "__main__":
    deploy()
