import pyttsx3

def hablar(texto):
    engine = pyttsx3.init()

    engine.setProperty('rate', 170)     
    engine.setProperty('volume', 1.0)   

   
    voces = engine.getProperty('voices')
    engine.setProperty('voice', voces[0].id)  

    engine.say(texto)
    engine.runAndWait()

if __name__ == "__main__":
    while True:
        texto = input(" Escribe algo para decir (o 'salir' para terminar): ")
        if texto.lower() == "salir":
            print(" Programa terminado.")
            break
        hablar(texto)
