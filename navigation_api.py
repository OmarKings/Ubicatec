# API de Navegación para Robot Asistente TEC
# Guía al usuario a diferentes ubicaciones del campus

import math

# ====
# UBICACIONES DEL CAMPUS
# ====
LOCATIONS = {
    "timhortons": {
        "name": "Tim Hortons",
        "lat": 20.613984,
        "lon": -100.403511,
        "aliases": ["tims", "tim", "timhurtons", "cafe", "coffee"]
    },
    "biblioteca": {
        "name": "Biblioteca",
        "lat": 20.613246,
        "lon": -100.403285,
        "aliases": ["library", "biblio", "libros", "estudiar"]
    }
}

# ====
# FUNCIONES DE NAVEGACIÓN
# ====

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula la distancia en metros entre dos coordenadas GPS"""
    R = 6371000  # Radio de la Tierra en metros
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calcula el rumbo (bearing) en grados desde punto 1 a punto 2"""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360


def get_direction_command(bearing):
    """Convierte un bearing en un comando de dirección"""
    # Normalizar bearing a 0-360
    bearing = bearing % 360
    
    if 337.5 <= bearing or bearing < 22.5:
        return "adelante"
    elif 22.5 <= bearing < 67.5:
        return "derecha_ligera"
    elif 67.5 <= bearing < 112.5:
        return "derecha"
    elif 112.5 <= bearing < 157.5:
        return "derecha_atras"
    elif 157.5 <= bearing < 202.5:
        return "atras"
    elif 202.5 <= bearing < 247.5:
        return "izquierda_atras"
    elif 247.5 <= bearing < 292.5:
        return "izquierda"
    else:  # 292.5 <= bearing < 337.5
        return "izquierda_ligera"


def find_location(query):
    """Busca una ubicación basándose en el texto del usuario"""
    query = query.lower().strip()
    
    for loc_id, loc_data in LOCATIONS.items():
        # Buscar en el nombre principal
        if query in loc_data["name"].lower():
            return loc_id, loc_data
        
        # Buscar en los aliases
        for alias in loc_data["aliases"]:
            if alias in query or query in alias:
                return loc_id, loc_data
    
    return None, None


def navigate_to_location(current_lat, current_lon, destination_id):
    """
    Calcula la navegación desde la posición actual hasta el destino
    
    Returns:
        dict con información de navegación o None si no se encuentra
    """
    if destination_id not in LOCATIONS:
        return None
    
    dest = LOCATIONS[destination_id]
    
    # Calcular distancia
    distance = haversine_distance(current_lat, current_lon, dest["lat"], dest["lon"])
    
    # Calcular dirección
    bearing = calculate_bearing(current_lat, current_lon, dest["lat"], dest["lon"])
    direction = get_direction_command(bearing)
    
    # Determinar si ya llegamos (dentro de 5 metros)
    arrived = distance < 5.0
    
    return {
        "destination": dest["name"],
        "distance": round(distance, 1),
        "bearing": round(bearing, 1),
        "direction": direction,
        "arrived": arrived,
        "lat": dest["lat"],
        "lon": dest["lon"]
    }


def get_navigation_instruction(nav_info):
    """Genera una instrucción en lenguaje natural"""
    if nav_info["arrived"]:
        return f"Has llegado a {nav_info['destination']}!"
    
    dist = nav_info["distance"]
    direction = nav_info["direction"]
    
    # Traducir dirección a español natural
    dir_text = {
        "adelante": "sigue adelante",
        "derecha_ligera": "gira ligeramente a la derecha",
        "derecha": "gira a la derecha",
        "derecha_atras": "gira fuertemente a la derecha",
        "atras": "da la vuelta",
        "izquierda_atras": "gira fuertemente a la izquierda",
        "izquierda": "gira a la izquierda",
        "izquierda_ligera": "gira ligeramente a la izquierda"
    }.get(direction, "continua")
    
    return f"{nav_info['destination']}: {dir_text}, {dist}m"


def process_user_query(query):
    """
    Procesa una consulta del usuario y determina si quiere ir a algún lugar
    
    Returns:
        dict con información de la intención del usuario
    """
    query = query.lower().strip()
    
    # Palabras clave de navegación
    nav_keywords = ["llevar", "ir", "donde", "esta", "ubicar", "encontrar", "buscar", "guiar", "camino"]
    
    # Verificar si es una consulta de navegación
    is_navigation = any(keyword in query for keyword in nav_keywords)
    
    if not is_navigation:
        return {"type": "unknown", "query": query}
    
    # Buscar la ubicación mencionada
    loc_id, loc_data = find_location(query)
    
    if loc_id:
        return {
            "type": "navigation",
            "destination_id": loc_id,
            "destination_name": loc_data["name"],
            "query": query
        }
    else:
        return {
            "type": "navigation_unknown",
            "query": query,
            "available_locations": [loc["name"] for loc in LOCATIONS.values()]
        }


# ====
# EJEMPLO DE USO
# ====
if __name__ == "__main__":
    # Ejemplo: Usuario en una posición del campus
    current_lat = 20.613500
    current_lon = -100.403400
    
    # Probar consultas
    queries = [
        "quiero ir a tims",
        "llevame a la biblioteca",
        "donde esta tim hortons",
        "no se donde esta, ayuda"
    ]
    
    print("🗺️  API de Navegación - Pruebas\n")
    
    for query in queries:
        print(f"Query: '{query}'")
        result = process_user_query(query)
        print(f"Resultado: {result}")
        
        if result["type"] == "navigation":
            nav = navigate_to_location(current_lat, current_lon, result["destination_id"])
            instruction = get_navigation_instruction(nav)
            print(f"Instrucción: {instruction}")
        
        print()