from googletrans import Translator
import pyttsx3

def traducir_texto(texto, idioma_destino):
    traductor = Translator()
    traduccion = traductor.translate(texto, dest=idioma_destino)
    return traduccion.text

def hablar_texto(texto):
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)
    engine.setProperty('volume', 1.0)
    engine.say(texto)
    engine.runAndWait()

def main():
    print("Traductor de texto a voz")
    print("Ejemplo de idiomas: en (inglés), fr (francés), de (alemán), ja (japonés), it (italiano)")
    print("Escribe 'salir' para terminar.\n")

    idioma_destino = input(" Ingresa el código del idioma destino: ").strip()

    while True:
        texto = input("\n Escribe el texto a traducir: ")
        if texto.lower() == "salir":
            print("👋 Programa terminado.")
            break

        traduccion = traducir_texto(texto, idioma_destino)
        print(f"Traducción ({idioma_destino}): {traduccion}")

        hablar_texto(traduccion)

if __name__ == "__main__":
    main()
