from __future__ import annotations

from benchmark_daily_lstm import DEFAULT_BILSTM_OUTPUT_ROOT, main


if __name__ == "__main__":
    main(
        default_bidirectional=True,
        default_output_root=DEFAULT_BILSTM_OUTPUT_ROOT,
        description="Daily Bi-LSTM baseline",
    )
