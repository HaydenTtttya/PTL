from finetune import (
    PRETRAIN_RUNS_DIR,
    build_daily_target_config,
    find_latest_pretrain_run,
    main,
)


if __name__ == "__main__":
    latest_pretrain_dir = find_latest_pretrain_run(PRETRAIN_RUNS_DIR)
    if latest_pretrain_dir is None:
        print("未找到可用的 pretrain run。请先生成包含 config.json 和 model.pth 的预训练结果。")
    else:
        print(f"自动选择最新 pretrain run: {latest_pretrain_dir}")
        main(
            pretrain_model_dir=latest_pretrain_dir,
            custom_config=build_daily_target_config(),
            seed=42,
        )
