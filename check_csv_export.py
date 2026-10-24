import csv

import torch

from pykt.models.dkt import DKT
from pykt.models.evaluate_model import evaluate


torch.manual_seed(42)
model = DKT(num_c=3, emb_size=8, dropout=0)

# 两条示例序列，第二条的最后一个位置不参与评估。
batch = {
    "qseqs": torch.empty((2, 0), dtype=torch.long),
    "shft_qseqs": torch.empty((2, 0), dtype=torch.long),
    "cseqs": torch.tensor([[0, 1], [1, 2]]),
    "shft_cseqs": torch.tensor([[1, 2], [2, 0]]),
    "rseqs": torch.tensor([[0, 1], [1, 0]]),
    "shft_rseqs": torch.tensor([[1, 0], [0, 0]]),
    "masks": torch.tensor([[True, True], [True, False]]),
    "smasks": torch.tensor([[True, True], [True, False]]),
}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

before = evaluate(model, [batch], "dkt")
after = evaluate(
    model,
    [batch],
    "dkt",
    export_csv="local_results/export_test.csv",
)

assert before == after, "导出功能改变了评估结果"

with open(
    "local_results/export_test.csv",
    encoding="utf-8-sig",
    newline="",
) as handle:
    rows = list(csv.DictReader(handle))

assert len(rows) == 3, "应导出三个有效位置"
assert [int(row["actual"]) for row in rows] == [1, 0, 0]
assert all(0 <= float(row["probability"]) <= 1 for row in rows)

print("验证通过：导出三条有效预测，评估指标保持一致。")