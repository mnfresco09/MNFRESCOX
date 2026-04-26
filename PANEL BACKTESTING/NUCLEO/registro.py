import importlib
import inspect
from pathlib import Path
from NUCLEO.base_estrategia import BaseEstrategia


def cargar_estrategias() -> dict[int, BaseEstrategia]:
    """
    Escanea ESTRATEGIAS/, importa cada archivo .py y recoge
    todas las clases que heredan de BaseEstrategia.
    Devuelve un dict {ID: instancia}.
    No hay que registrar nada a mano: añadir el archivo es suficiente.
    """
    carpeta = Path(__file__).resolve().parents[1] / "ESTRATEGIAS"
    registro: dict[int, BaseEstrategia] = {}
    errores = []

    for archivo in sorted(carpeta.glob("*.py")):
        if archivo.name.startswith("_"):
            continue

        modulo_nombre = f"ESTRATEGIAS.{archivo.stem}"
        try:
            modulo = importlib.import_module(modulo_nombre)
        except Exception as e:
            errores.append(f"  No se pudo importar '{archivo.name}': {e}")
            continue

        for _, clase in inspect.getmembers(modulo, inspect.isclass):
            if not (issubclass(clase, BaseEstrategia) and clase is not BaseEstrategia):
                continue
            if not hasattr(clase, "ID") or not hasattr(clase, "NOMBRE"):
                errores.append(f"  '{clase.__name__}' en '{archivo.name}' no define ID o NOMBRE.")
                continue

            id_estrategia = clase.ID
            if id_estrategia in registro:
                existente = type(registro[id_estrategia]).__name__
                errores.append(
                    f"  ID {id_estrategia} duplicado: '{clase.__name__}' y '{existente}'. "
                    f"Cada estrategia debe tener un ID único."
                )
                continue

            registro[id_estrategia] = clase()

    if errores:
        print("\n[REGISTRO] Problemas al cargar estrategias:\n")
        for e in errores:
            print(e)
        print()

    return registro


def obtener_estrategia(registro: dict, estrategia_id) -> list[BaseEstrategia]:
    """
    Devuelve la lista de estrategias a ejecutar según config.ESTRATEGIA_ID.
    Acepta cualquier forma: "all", 1, "1", [1, 2], "1,2", "1, 2", etc.
    """
    ids_int = _normalizar_ids(estrategia_id, sorted(registro.keys()))
    resultado = []
    for id_ in ids_int:
        if id_ not in registro:
            raise ValueError(
                f"Estrategia con ID {id_} no encontrada. "
                f"IDs disponibles: {sorted(registro.keys())}"
            )
        resultado.append(registro[id_])
    return resultado


def _normalizar_ids(valor, disponibles: list[int]) -> list[int]:
    """Convierte cualquier variante de ESTRATEGIA_ID a lista de ints."""
    # "all" o "ALL" en cualquier forma
    if isinstance(valor, str):
        limpio = valor.strip().lower()
        if limpio == "all":
            return list(disponibles)
        # "1" o "1,2" o "1, 2"
        partes = [p.strip() for p in limpio.split(",") if p.strip()]
        try:
            return [int(p) for p in partes]
        except ValueError:
            raise ValueError(
                f"ESTRATEGIA_ID '{valor}' no es válido. "
                f"Usa un entero, lista de enteros o 'all'."
            )

    if isinstance(valor, int):
        return [valor]

    if isinstance(valor, (list, tuple)):
        result = []
        for item in valor:
            result.extend(_normalizar_ids(item, disponibles))
        return result

    # Cualquier otro numérico (float accidental, etc.)
    try:
        return [int(valor)]
    except (TypeError, ValueError):
        raise ValueError(
            f"ESTRATEGIA_ID '{valor}' no es válido. "
            f"Usa un entero, lista de enteros o 'all'."
        )
