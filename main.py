"""Small broker-modules import smoke example.

Run with:
    uv run python main.py
"""

from __future__ import annotations

from brokers.kis import Credentials as KisCredentials
from brokers.kis import KisClient
from brokers.kiwoom import Credentials as KiwoomCredentials
from brokers.kiwoom import KiwoomClient
from brokers.krx import Credentials as KrxCredentials
from brokers.krx import KrxClient
from brokers.toss import Credentials as TossCredentials
from brokers.toss import TossClient

def main() -> None:
    """Show that broker-modules is installed and importable.

    This deliberately avoids live API calls. Fill real credentials before
    using the clients against broker APIs.
    """

    examples = [
        (
            "KIS",
            KisClient,
            KisCredentials(
                app_key="KIS_APP_KEY",
                app_secret="KIS_APP_SECRET",
                account_number="00000000-00",
            ),
        ),
        (
            "Kiwoom",
            KiwoomClient,
            KiwoomCredentials(
                app_key="KIWOOM_APP_KEY",
                secret_key="KIWOOM_SECRET_KEY",
            ),
        ),
        (
            "KRX",
            KrxClient,
            KrxCredentials(
                auth_key="KRX_AUTH_KEY",
            ),
        ),
        (
            "Toss",
            TossClient,
            TossCredentials(
                client_id="TOSS_CLIENT_ID",
                client_secret="TOSS_CLIENT_SECRET",
            ),
        ),
    ]

    print("broker-modules import check")
    for broker_name, client_cls, credentials in examples:
        print(
            f"- {broker_name}: {client_cls.__name__} "
            f"with {type(credentials).__name__}"
        )


if __name__ == "__main__":
    main()
