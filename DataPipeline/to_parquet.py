"""
to_parquet.py — Converte a ABT de CSV para Parquet e gera o perfil das colunas.

Por que existe:
  - 658 MB de CSV viram ~150 MB de Parquet colunar, e o DuckDB lê só as
    colunas pedidas — é o que torna a API viável sem carregar a base em RAM.
  - `SUMMARIZE` produz, numa passada, min/max/nulos/cardinalidade de todas as
    colunas. Sem esse perfil, /stats/missing teria que varrer 473 colunas a
    cada request.
  - Converte uma ABT que já existe em disco, sem re-rodar a agregação das
    9 tabelas (que exige os 2,6 GB de dados brutos).

Uso:
    python DataPipeline/to_parquet.py
    python DataPipeline/to_parquet.py --csv Dados/abt.csv --parquet Dados/abt.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _lit(path: Path | str) -> str:
    """Caminho como literal SQL. O DuckDB não aceita parâmetro preparado em
    read_csv/read_parquet/COPY, então interpolamos — com as aspas escapadas.
    Os caminhos vêm do config.yaml, nunca de entrada de usuário."""
    return "'" + str(path).replace("'", "''") + "'"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def convert(csv_path: Path, pq_path: Path, id_col: str, target: str,
            profile_out: Path | None) -> None:
    con = duckdb.connect()

    print(f"[parquet] lendo {csv_path.name} ({csv_path.stat().st_size / 1e6:.0f} MB)")
    # sample_size=-1: inspeciona a base toda antes de inferir tipo. Sem isso,
    # colunas quase todas nulas no início do arquivo viram VARCHAR por engano.
    con.execute("CREATE VIEW src AS SELECT * FROM read_csv("
                f"{_lit(csv_path)}, header=true, sample_size=-1)")

    pq_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[parquet] gravando {pq_path.name}")
    con.execute(f"COPY src TO {_lit(pq_path)} (FORMAT PARQUET, COMPRESSION ZSTD)")

    # ---- validação: converter errado em silêncio é pior que não converter ----
    print("[parquet] validando")
    con.execute(f"CREATE VIEW dst AS SELECT * FROM read_parquet({_lit(pq_path)})")

    n_src, n_dst = (con.execute(f"SELECT count(*) FROM {v}").fetchone()[0]
                    for v in ("src", "dst"))
    c_src = len(con.execute("DESCRIBE src").fetchall())
    c_dst = len(con.execute("DESCRIBE dst").fetchall())
    t_src, t_dst = (con.execute(f"SELECT sum({target}) FROM {v}").fetchone()[0]
                    for v in ("src", "dst"))
    uniq = con.execute(
        f"SELECT count(*) = count(DISTINCT {id_col}) FROM dst").fetchone()[0]

    checks = [
        ("linhas", n_src == n_dst, f"{n_src} == {n_dst}"),
        ("colunas", c_src == c_dst, f"{c_src} == {c_dst}"),
        (f"sum({target})", t_src == t_dst, f"{t_src} == {t_dst}"),
        (f"{id_col} único", bool(uniq), "1 linha por cliente"),
    ]
    for name, ok, detail in checks:
        print(f"    {'OK  ' if ok else 'FALHA'} {name:16} {detail}")
    if not all(ok for _, ok, _ in checks):
        raise SystemExit("[parquet] validação falhou — parquet NÃO confiável")

    size_mb = pq_path.stat().st_size / 1e6
    print(f"[parquet] {pq_path.name}: {size_mb:.0f} MB "
          f"({csv_path.stat().st_size / pq_path.stat().st_size:.1f}x menor)")

    # ---- perfil das colunas (alimenta /stats/missing e /stats/distribution) ----
    if profile_out:
        print("[parquet] gerando perfil das colunas (SUMMARIZE)")
        rows = con.execute("SUMMARIZE SELECT * FROM dst").fetchdf()
        rows = rows.where(rows.notna(), None)
        profile = {
            "source": str(pq_path.relative_to(ROOT)),
            "n_rows": int(n_dst),
            "n_columns": int(c_dst),
            "columns": json.loads(rows.to_json(orient="records")),
        }
        profile_out.parent.mkdir(parents=True, exist_ok=True)
        profile_out.write_text(json.dumps(profile))
        with_na = sum(1 for c in profile["columns"]
                      if float(c.get("null_percentage") or 0) > 0)
        print(f"[parquet] perfil salvo em {profile_out.name} "
              f"({c_dst} colunas, {with_na} com nulos)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "DataPipeline" / "config.yaml"))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--no-profile", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    p = cfg["paths"]
    csv_path = Path(args.csv) if args.csv else ROOT / p["abt"]
    pq_path = Path(args.parquet) if args.parquet else ROOT / p.get(
        "abt_parquet", "Dados/abt.parquet")
    profile = None if args.no_profile else ROOT / p.get(
        "profile_out", "artifacts/abt_profile.json")

    if not csv_path.exists():
        raise SystemExit(f"[parquet] {csv_path} não existe — rode abt_transform.py antes")

    convert(csv_path, pq_path, cfg["id_col"], cfg["target"], profile)


if __name__ == "__main__":
    main()
