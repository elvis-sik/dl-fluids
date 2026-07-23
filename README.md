# Deep Learning for Low-Reynolds-Number Flows

> [!NOTE]
> **Historical project.** This bachelor’s-thesis repository dates from 2020 and
> is archived; it is not maintained as a reusable package.

The project generated low-Reynolds-number flow cases around polygonal bodies
with OpenFOAM, then trained neural networks to predict velocity and pressure
fields and the drag coefficient.

```bash
python solve_cases.py
```

The training code lives in `train.py`; the accompanying analysis is in
`Analysis.ipynb`. Reproducing the original environment may require adapting
legacy dependencies and local OpenFOAM configuration.
