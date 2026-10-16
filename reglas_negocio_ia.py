"""
Campuslands - Gestión Operativa (MVP)
Módulo: Reglas de negocio + Triaje de incidencias con IA
Responsable: Juan Diego

Contenido:
  1. validar_reserva_espacio()   -> evita cruces de salón/horario
  2. validar_prestamo_equipo()   -> bloquea equipos 'En Reparación' o 'Prestado'
  3. triaje_incidencia()         -> prioridad + área (IA con respaldo por palabras clave)

Uso rápido:
    from reglas_negocio_ia import (
        validar_reserva_espacio, validar_prestamo_equipo, triaje_incidencia
    )

Para activar la IA real:
    pip install anthropic
    export ANTHROPIC_API_KEY="tu_clave"      (Windows: set ANTHROPIC_API_KEY=tu_clave)
Si no hay clave o falla la red, el sistema usa el análisis por palabras clave.
"""

import json
import os
import unicodedata
from datetime import datetime

# ---------------------------------------------------------------------------
# CONSTANTES (compartidas con el resto del equipo)
# ---------------------------------------------------------------------------
FORMATO_FECHA = "%Y-%m-%d"
FORMATO_HORA = "%H:%M"

ESTADO_DISPONIBLE = "Disponible"
ESTADO_PRESTADO = "Prestado"
ESTADO_REPARACION = "En Reparación"
ESTADOS_BLOQUEADOS = {ESTADO_PRESTADO, ESTADO_REPARACION}

PRIORIDADES = ("Alta", "Media", "Baja")
AREAS = ("Mantenimiento", "Sistemas", "Bienestar")


# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------
def _normalizar(texto: str) -> str:
    """Minúsculas y sin tildes, para comparar palabras clave de forma robusta."""
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def _a_minutos(hora: str) -> int:
    """Convierte 'HH:MM' a minutos desde las 00:00. Lanza ValueError si es inválida."""
    t = datetime.strptime(hora, FORMATO_HORA)
    return t.hour * 60 + t.minute


# ---------------------------------------------------------------------------
# 1. REGLA: RESERVAS DE ESPACIOS SIN SOLAPAMIENTO
# ---------------------------------------------------------------------------
def validar_reserva_espacio(nueva: dict, reservas_existentes: list) -> dict:
    """
    Valida que una reserva nueva no se cruce con las existentes.

    Cada reserva es un dict con:
        {"salon": "A101", "fecha": "2026-10-20", "hora_inicio": "08:00", "hora_fin": "10:00"}

    Dos reservas chocan si son del MISMO salón, MISMA fecha y sus horas se traslapan.
    Reservas contiguas (una termina 10:00 y otra empieza 10:00) NO se consideran cruce.

    Retorna:
        {"valida": bool, "mensaje": str, "conflicto": dict | None}
    """
    # --- Validación de datos de entrada ---
    campos = ("salon", "fecha", "hora_inicio", "hora_fin")
    faltantes = [c for c in campos if not nueva.get(c)]
    if faltantes:
        return {"valida": False, "mensaje": f"Faltan campos: {', '.join(faltantes)}", "conflicto": None}

    try:
        datetime.strptime(nueva["fecha"], FORMATO_FECHA)
        inicio_nuevo = _a_minutos(nueva["hora_inicio"])
        fin_nuevo = _a_minutos(nueva["hora_fin"])
    except ValueError:
        return {
            "valida": False,
            "mensaje": "Formato inválido. Use fecha AAAA-MM-DD y hora HH:MM.",
            "conflicto": None,
        }

    if inicio_nuevo >= fin_nuevo:
        return {"valida": False, "mensaje": "La hora de inicio debe ser menor que la de fin.", "conflicto": None}

    # --- Búsqueda de conflictos ---
    for r in reservas_existentes:
        if r["salon"] != nueva["salon"] or r["fecha"] != nueva["fecha"]:
            continue
        inicio_r = _a_minutos(r["hora_inicio"])
        fin_r = _a_minutos(r["hora_fin"])
        # Condición clásica de traslape de intervalos
        if inicio_nuevo < fin_r and fin_nuevo > inicio_r:
            return {
                "valida": False,
                "mensaje": (
                    f"El salón {r['salon']} ya está reservado el {r['fecha']} "
                    f"de {r['hora_inicio']} a {r['hora_fin']}."
                ),
                "conflicto": r,
            }

    return {"valida": True, "mensaje": "Reserva válida.", "conflicto": None}


# ---------------------------------------------------------------------------
# 2. REGLA: PRÉSTAMO DE EQUIPOS TECNOLÓGICOS
# ---------------------------------------------------------------------------
def validar_prestamo_equipo(equipo: dict) -> dict:
    """
    Bloquea el préstamo si el equipo está 'En Reparación' o 'Prestado'.

    El equipo es un dict: {"id": "PC-014", "nombre": "Portátil Lenovo", "estado": "Disponible"}

    Retorna:
        {"permitido": bool, "mensaje": str}
    """
    estado = equipo.get("estado")
    nombre = equipo.get("nombre") or equipo.get("id", "El equipo")

    if estado is None:
        return {"permitido": False, "mensaje": f"{nombre} no tiene un estado definido."}

    if estado in ESTADOS_BLOQUEADOS:
        return {"permitido": False, "mensaje": f"{nombre} no se puede prestar: estado '{estado}'."}

    if estado != ESTADO_DISPONIBLE:
        return {"permitido": False, "mensaje": f"Estado desconocido '{estado}' para {nombre}."}

    return {"permitido": True, "mensaje": f"{nombre} está disponible para préstamo."}


def registrar_prestamo(equipo: dict) -> dict:
    """Si la regla lo permite, marca el equipo como 'Prestado'. Útil para integrar con el equipo."""
    resultado = validar_prestamo_equipo(equipo)
    if resultado["permitido"]:
        equipo["estado"] = ESTADO_PRESTADO
    return resultado


# ---------------------------------------------------------------------------
# 3. TRIAJE DE INCIDENCIAS CON IA
# ---------------------------------------------------------------------------
# Palabras clave (sin tildes) para el análisis local / respaldo.
PALABRAS_ALTA = [
    "urgente", "emergencia", "incendio", "humo", "fuego", "cortocircuito", "choque electrico",
    "inundacion", "fuga de gas", "herido", "accidente", "peligro", "riesgo", "amenaza",
    "agresion", "acoso", "crisis", "suicid", "desmayo", "no funciona nada", "caido",
    "servidor caido", "sin internet", "ataque", "hackeo", "virus",
]
PALABRAS_MEDIA = [
    "falla", "danado", "roto", "no enciende", "lento", "intermitente", "gotera", "fuga",
    "no abre", "error", "no conecta", "aire acondicionado", "sin luz", "ansiedad", "estres",
    "conflicto", "problema",
]
PALABRAS_AREA = {
    "Mantenimiento": [
        "luz", "bombillo", "lampara", "puerta", "ventana", "techo", "gotera", "fuga", "agua",
        "bano", "silla", "mesa", "pintura", "aire acondicionado", "ascensor", "inundacion",
        "incendio", "cortocircuito", "electrico", "limpieza", "pared", "cerradura",
    ],
    "Sistemas": [
        "internet", "wifi", "red", "computador", "pc", "portatil", "proyector", "software",
        "servidor", "sistema", "contrasena", "cuenta", "correo", "pantalla", "teclado",
        "mouse", "impresora", "virus", "hackeo", "plataforma", "aplicacion",
    ],
    "Bienestar": [
        "acoso", "agresion", "ansiedad", "estres", "depresion", "crisis", "suicid", "conflicto",
        "bullying", "discriminacion", "salud", "enfermo", "desmayo", "herido", "apoyo",
        "psicolog", "emocional", "amenaza",
    ],
}


def _triaje_por_palabras_clave(texto: str) -> dict:
    """Análisis local determinista: sirve como simulación de IA y como respaldo."""
    t = _normalizar(texto)

    # Prioridad
    if any(p in t for p in PALABRAS_ALTA):
        prioridad = "Alta"
    elif any(p in t for p in PALABRAS_MEDIA):
        prioridad = "Media"
    else:
        prioridad = "Baja"

    # Área: la que tenga más coincidencias (en empate, el orden de AREAS decide)
    puntajes = {a: sum(1 for p in kws if p in t) for a, kws in PALABRAS_AREA.items()}
    mejor = max(AREAS, key=lambda a: puntajes[a])
    area = mejor if puntajes[mejor] > 0 else "Mantenimiento"  # área por defecto

    return {"prioridad": prioridad, "area": area, "fuente": "palabras_clave"}


def _triaje_con_ia(texto: str) -> dict:
    """
    Llama a un modelo de lenguaje (API de Anthropic) para clasificar la incidencia.
    Lanza excepción si no hay clave, librería o respuesta válida (el llamador usa el respaldo).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Sin ANTHROPIC_API_KEY")

    import anthropic  # import local: el módulo funciona aunque la librería no esté instalada

    cliente = anthropic.Anthropic(api_key=api_key)
    sistema = (
        "Eres un clasificador de incidencias de una institución educativa. "
        "Responde SOLO con un JSON válido, sin texto extra ni markdown, con esta forma exacta: "
        '{"prioridad": "Alta|Media|Baja", "area": "Mantenimiento|Sistemas|Bienestar"}. '
        "Alta: riesgo para personas, emergencia o servicio crítico caído. "
        "Media: falla que afecta la operación pero sin riesgo inmediato. "
        "Baja: molestia menor o solicitud no urgente. "
        "Mantenimiento: infraestructura, electricidad, agua, mobiliario. "
        "Sistemas: computadores, red, internet, software, cuentas. "
        "Bienestar: salud, convivencia, acoso, apoyo emocional."
    )
    respuesta = cliente.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=100,
        system=sistema,
        messages=[{"role": "user", "content": f"Incidencia reportada:\n{texto}"}],
    )
    bruto = respuesta.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    datos = json.loads(bruto)

    # Validar que la IA devolvió valores permitidos
    if datos.get("prioridad") not in PRIORIDADES or datos.get("area") not in AREAS:
        raise ValueError(f"Respuesta fuera de los valores permitidos: {datos}")

    return {"prioridad": datos["prioridad"], "area": datos["area"], "fuente": "ia"}


def triaje_incidencia(texto: str, usar_ia: bool = True) -> dict:
    """
    Recibe el texto de una incidencia y devuelve:
        {"prioridad": "Alta|Media|Baja", "area": "Mantenimiento|Sistemas|Bienestar", "fuente": "ia|palabras_clave"}

    Intenta primero con la IA; si no está disponible o falla, usa palabras clave.
    Además, aplica una regla de seguridad: si hay palabras de urgencia crítica,
    la prioridad nunca baja de 'Alta', aunque la IA diga otra cosa.
    """
    if not isinstance(texto, str) or not texto.strip():
        return {"prioridad": "Baja", "area": "Mantenimiento", "fuente": "entrada_vacia"}

    resultado = None
    if usar_ia:
        try:
            resultado = _triaje_con_ia(texto)
        except Exception:
            resultado = None  # cualquier fallo -> respaldo local

    if resultado is None:
        resultado = _triaje_por_palabras_clave(texto)

    # Red de seguridad: urgencias críticas siempre son Alta
    if any(p in _normalizar(texto) for p in PALABRAS_ALTA) and resultado["prioridad"] != "Alta":
        resultado["prioridad"] = "Alta"

    return resultado


# ---------------------------------------------------------------------------
# PRUEBAS RÁPIDAS (python reglas_negocio_ia.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Reservas ---")
    reservas = [{"salon": "A101", "fecha": "2026-10-20", "hora_inicio": "08:00", "hora_fin": "10:00"}]
    print(validar_reserva_espacio(
        {"salon": "A101", "fecha": "2026-10-20", "hora_inicio": "09:00", "hora_fin": "11:00"}, reservas))
    print(validar_reserva_espacio(
        {"salon": "A101", "fecha": "2026-10-20", "hora_inicio": "10:00", "hora_fin": "12:00"}, reservas))

    print("\n--- Equipos ---")
    print(validar_prestamo_equipo({"id": "PC-01", "nombre": "Portátil 01", "estado": "En Reparación"}))
    print(validar_prestamo_equipo({"id": "PC-02", "nombre": "Portátil 02", "estado": "Disponible"}))

    print("\n--- Triaje ---")
    for caso in [
        "Hay humo saliendo del tomacorriente del aula 204, es urgente",
        "El wifi del laboratorio está muy lento",
        "Un estudiante sufre acoso de sus compañeros",
        "Se dañó una silla en la biblioteca",
    ]:
        print(caso, "->", triaje_incidencia(caso))
