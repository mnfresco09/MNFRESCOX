from __future__ import annotations

from dataclasses import dataclass
from math import isclose

import polars as pl


TOL = 1e-7


@dataclass(frozen=True)
class HuellaDatos:
    etapa: str
    filas: int
    columnas: tuple[str, ...]
    ts_inicio: object
    ts_fin: object


def huella_dataframe(etapa: str, df: pl.DataFrame) -> HuellaDatos:
    if df.is_empty():
        raise ValueError(f"[INTEGRIDAD] {etapa}: DataFrame vacio.")
    if "timestamp" not in df.columns:
        raise ValueError(f"[INTEGRIDAD] {etapa}: falta columna timestamp.")

    _verificar_timestamps(etapa, df)
    _verificar_ohlc(etapa, df)

    return HuellaDatos(
        etapa=etapa,
        filas=df.height,
        columnas=tuple(df.columns),
        ts_inicio=df["timestamp"][0],
        ts_fin=df["timestamp"][-1],
    )


def verificar_resampleo(
    origen: pl.DataFrame,
    destino: pl.DataFrame,
    timeframe: str,
) -> None:
    huella_dataframe(f"resampleo {timeframe}", destino)

    if timeframe == "1m":
        if destino.height != origen.height:
            raise ValueError(
                "[INTEGRIDAD] Resampleo 1m no debe cambiar el numero de filas."
            )
        if not destino["timestamp"].equals(origen["timestamp"]):
            raise ValueError("[INTEGRIDAD] Resampleo 1m altero timestamps.")
        return

    if "volume" in origen.columns and "volume" in destino.columns:
        vol_origen = float(origen["volume"].sum())
        vol_destino = float(destino["volume"].sum())
        if not isclose(vol_origen, vol_destino, rel_tol=TOL, abs_tol=TOL):
            raise ValueError(
                "[INTEGRIDAD] Volumen no conservado tras resampleo: "
                f"origen={vol_origen}, destino={vol_destino}."
            )

    if destino["timestamp"][0] < origen["timestamp"][0]:
        raise ValueError("[INTEGRIDAD] Resampleo creo timestamp inicial anterior al origen.")
    if destino["timestamp"][-1] > origen["timestamp"][-1]:
        raise ValueError("[INTEGRIDAD] Resampleo creo timestamp final posterior al origen.")


def verificar_senales(df: pl.DataFrame, senales: pl.Series) -> dict[int, int]:
    if len(senales) != df.height:
        raise ValueError(
            f"[INTEGRIDAD] Longitud de senales incorrecta: {len(senales):,} != {df.height:,}."
        )
    if senales.null_count() > 0:
        raise ValueError("[INTEGRIDAD] Las senales contienen nulos.")

    valores = set(senales.cast(pl.Int8).unique().to_list())
    invalidos = valores - {-1, 0, 1}
    if invalidos:
        raise ValueError(f"[INTEGRIDAD] Senales invalidas: {sorted(invalidos)}.")

    return {
        -1: int((senales == -1).sum()),
        0: int((senales == 0).sum()),
        1: int((senales == 1).sum()),
    }


def verificar_salidas_custom(df: pl.DataFrame, salidas: pl.Series) -> dict[int, int]:
    if len(salidas) != df.height:
        raise ValueError(
            "[INTEGRIDAD] Longitud de salidas custom incorrecta: "
            f"{len(salidas):,} != {df.height:,}."
        )
    if salidas.null_count() > 0:
        raise ValueError("[INTEGRIDAD] Las salidas custom contienen nulos.")

    valores = set(salidas.cast(pl.Int8).unique().to_list())
    invalidos = valores - {-1, 0, 1}
    if invalidos:
        raise ValueError(f"[INTEGRIDAD] Salidas custom invalidas: {sorted(invalidos)}.")

    return {
        -1: int((salidas == -1).sum()),
        0: int((salidas == 0).sum()),
        1: int((salidas == 1).sum()),
    }


def verificar_resultado(
    df: pl.DataFrame,
    senales: pl.Series,
    resultado,
    salidas_custom: pl.Series | None = None,
) -> None:
    total_trades = int(resultado.total_trades)
    trades = list(resultado.trades)
    if total_trades != len(trades):
        raise ValueError("[INTEGRIDAD] total_trades no coincide con len(trades).")

    if len(resultado.equity_curve) != total_trades + 1:
        raise ValueError("[INTEGRIDAD] equity_curve debe tener saldo inicial + un punto por trade.")

    pnl_total = sum(float(t.pnl) for t in trades)
    if not isclose(float(resultado.pnl_total), pnl_total, rel_tol=TOL, abs_tol=TOL):
        raise ValueError("[INTEGRIDAD] pnl_total no coincide con la suma de trades.")

    saldo_esperado = float(resultado.saldo_inicial) + pnl_total
    if not isclose(float(resultado.saldo_final), saldo_esperado, rel_tol=TOL, abs_tol=TOL):
        raise ValueError("[INTEGRIDAD] saldo_final no coincide con saldo_inicial + pnl_total.")

    if not isclose(float(resultado.equity_curve[0]), float(resultado.saldo_inicial), abs_tol=TOL):
        raise ValueError("[INTEGRIDAD] equity_curve no arranca en saldo_inicial.")
    if not isclose(float(resultado.equity_curve[-1]), float(resultado.saldo_final), abs_tol=TOL):
        raise ValueError("[INTEGRIDAD] equity_curve no termina en saldo_final.")

    timestamps = _timestamps_us(df)
    opens = df["open"].cast(pl.Float64).to_list()
    highs = df["high"].cast(pl.Float64).to_list()
    lows = df["low"].cast(pl.Float64).to_list()
    closes = df["close"].cast(pl.Float64).to_list()
    senales_lista = senales.cast(pl.Int8).to_list()
    salidas_custom_lista = (
        salidas_custom.cast(pl.Int8).to_list()
        if salidas_custom is not None
        else None
    )

    for n, trade in enumerate(trades, start=1):
        _verificar_trade(
            n,
            trade,
            timestamps,
            opens,
            highs,
            lows,
            closes,
            senales_lista,
            salidas_custom_lista,
        )


def _verificar_trade(
    n: int,
    trade,
    timestamps: list[int],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    senales: list[int],
    salidas_custom: list[int] | None,
) -> None:
    total = len(timestamps)
    idx_senal = int(trade.idx_señal)
    idx_entrada = int(trade.idx_entrada)
    idx_salida = int(trade.idx_salida)

    if not (0 <= idx_senal < idx_entrada <= idx_salida < total):
        raise ValueError(f"[INTEGRIDAD] Trade {n}: indices fuera de orden.")
    if idx_entrada != idx_senal + 1:
        raise ValueError(f"[INTEGRIDAD] Trade {n}: entrada no ocurre en vela N+1.")
    if int(trade.direccion) != int(senales[idx_senal]):
        raise ValueError(f"[INTEGRIDAD] Trade {n}: direccion no coincide con la senal.")
    if int(trade.ts_señal) != timestamps[idx_senal]:
        raise ValueError(f"[INTEGRIDAD] Trade {n}: timestamp de senal no coincide.")
    if int(trade.ts_entrada) != timestamps[idx_entrada]:
        raise ValueError(f"[INTEGRIDAD] Trade {n}: timestamp de entrada no coincide.")
    if int(trade.ts_salida) != timestamps[idx_salida]:
        raise ValueError(f"[INTEGRIDAD] Trade {n}: timestamp de salida no coincide.")
    if not isclose(float(trade.precio_entrada), opens[idx_entrada], rel_tol=TOL, abs_tol=TOL):
        raise ValueError(f"[INTEGRIDAD] Trade {n}: precio_entrada no es el open de N+1.")

    precio_salida = float(trade.precio_salida)
    motivo = str(trade.motivo_salida)
    if motivo in {"END", "BARS", "CUSTOM"}:
        if not isclose(precio_salida, closes[idx_salida], rel_tol=TOL, abs_tol=TOL):
            raise ValueError(f"[INTEGRIDAD] Trade {n}: salida {motivo} no usa close.")
        if motivo == "CUSTOM":
            if salidas_custom is None:
                raise ValueError(f"[INTEGRIDAD] Trade {n}: salida CUSTOM sin serie auditada.")
            if int(salidas_custom[idx_salida]) != int(trade.direccion):
                raise ValueError(
                    f"[INTEGRIDAD] Trade {n}: salida CUSTOM no coincide con la direccion."
                )
    elif motivo in {"SL", "TP"}:
        if precio_salida < lows[idx_salida] - TOL or precio_salida > highs[idx_salida] + TOL:
            raise ValueError(f"[INTEGRIDAD] Trade {n}: salida {motivo} fuera del rango de vela.")
    else:
        raise ValueError(f"[INTEGRIDAD] Trade {n}: motivo_salida desconocido: {motivo}.")


def _verificar_timestamps(etapa: str, df: pl.DataFrame) -> None:
    if not df["timestamp"].is_sorted():
        raise ValueError(f"[INTEGRIDAD] {etapa}: timestamps no ordenados.")
    duplicados = df.height - df["timestamp"].n_unique()
    if duplicados:
        raise ValueError(f"[INTEGRIDAD] {etapa}: {duplicados:,} timestamps duplicados.")


def _verificar_ohlc(etapa: str, df: pl.DataFrame) -> None:
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"[INTEGRIDAD] {etapa}: falta columna {col}.")
        if df[col].null_count() > 0:
            raise ValueError(f"[INTEGRIDAD] {etapa}: columna {col} contiene nulos.")

    invalidas = df.filter(
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("high") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
    ).height
    if invalidas:
        raise ValueError(f"[INTEGRIDAD] {etapa}: {invalidas:,} velas OHLC incoherentes.")


def _timestamps_us(df: pl.DataFrame) -> list[int]:
    dtype = df.schema["timestamp"]
    if isinstance(dtype, pl.Datetime):
        return df.select(pl.col("timestamp").dt.epoch("us")).to_series().to_list()
    return df["timestamp"].cast(pl.Int64).to_list()
