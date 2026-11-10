import argparse
import os
import uvicorn

from txguard.config import PROJECT_ROOT
from txguard.ml.trainer import run_cli_training


def interactive_setup() -> None:
    print("=" * 72)
    print("   TxGuard × PayPal AI Risk Orchestration & Agentic Commerce Engine")
    print("=" * 72)
    print("\n[INFO] Configure your environment keys below.")
    print("       (Tip: Press ENTER on any prompt to use Hybrid Sandbox Mode without keys!)\n")

    env_path = PROJECT_ROOT / ".env"
    existing_vars = {}
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    existing_vars[k.strip()] = v.strip()

    paypal_id = input(
        f"PayPal Sandbox Client ID [{existing_vars.get('PAYPAL_CLIENT_ID', 'None')}]: "
    ).strip() or existing_vars.get("PAYPAL_CLIENT_ID", "")

    paypal_secret = input(
        f"PayPal Sandbox Client Secret [{existing_vars.get('PAYPAL_CLIENT_SECRET', 'None')}]: "
    ).strip() or existing_vars.get("PAYPAL_CLIENT_SECRET", "")

    gemini_key = input(
        f"Google Gemini API Key [{existing_vars.get('GEMINI_API_KEY', 'None')}]: "
    ).strip() or existing_vars.get("GEMINI_API_KEY", "")

    new_env_content = (
        "# TxGuard × PayPal Developer Platform Configuration\n"
        f"PAYPAL_CLIENT_ID={paypal_id}\n"
        f"PAYPAL_CLIENT_SECRET={paypal_secret}\n"
        "PAYPAL_API_BASE=https://api-m.sandbox.paypal.com\n"
        "PAYPAL_WEBHOOK_ID=\n\n"
        "# Google Gemini API Configuration (gemini-3.8-flash)\n"
        f"GEMINI_API_KEY={gemini_key}\n"
        "GEMINI_MODEL=gemini-3.8-flash\n"
    )
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(new_env_content)

    os.environ["PAYPAL_CLIENT_ID"] = paypal_id
    os.environ["PAYPAL_CLIENT_SECRET"] = paypal_secret
    os.environ["GEMINI_API_KEY"] = gemini_key
    print(f"\n[OK] Configuration saved to {env_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TxGuard × PayPal AI Unified CLI Runner")
    parser.add_argument("--setup", action="store_true", help="Prompt interactively for API keys before starting")
    parser.add_argument("--train", action="store_true", help="Run the PyTorch model training pipeline")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server bind host")
    parser.add_argument("--port", type=int, default=8000, help="Server bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args, unknown = parser.parse_known_args()

    if args.train:
        run_cli_training()
        return

    if args.setup:
        interactive_setup()

    print(f"[INFO] Starting TxGuard Web Studio at http://{args.host}:{args.port} ...")
    uvicorn.run("txguard.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

