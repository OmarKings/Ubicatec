import socket
socket.create_connection(("www.google.com", 80), timeout=5)
print("OK conexión")