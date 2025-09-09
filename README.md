# futu_autotrade

This is a simplified demonstration project.

## 模型输入体检与常见错误

通过 `debug_probe.py` 可以对模型输入进行体检，常见问题包括列缺失、`NaN`/`inf` 值、数据类型不一致等。`bridge.build_latest_feature_row` 会自动补齐缺列并将异常值填充为 `0`，同时通过 Prometheus 指标 `features_nan_rate` 记录替换比例。
