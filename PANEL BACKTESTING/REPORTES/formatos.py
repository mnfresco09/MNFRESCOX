from __future__ import annotations

from math import isfinite


def formatear_duracion(segundos: float, velas: float | int | None = None) -> str:
    try:
        segundos_float = float(segundos)
    except (TypeError, ValueError):
        segundos_float = 0.0

    if isfinite(segundos_float) and segundos_float > 0:
        total = int(round(segundos_float))
        dias = total // 86_400
        horas = (total % 86_400) // 3_600
        minutos = (total % 3_600) // 60
        if dias:
            return f"{dias}d {horas:02d}h"
        if horas:
            return f"{horas}h {minutos:02d}m"
        if minutos:
            return f"{minutos}m"
        return f"{total}s"

    if velas is not None:
        try:
            velas_float = float(velas)
        except (TypeError, ValueError):
            velas_float = 0.0
        if isfinite(velas_float) and velas_float > 0:
            if velas_float.is_integer():
                velas_txt = str(int(velas_float))
            else:
                velas_txt = f"{velas_float:.1f}"
            unidad = "vela" if velas_txt == "1" else "velas"
            return f"{velas_txt} {unidad}"

    return "-"
