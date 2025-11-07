import speech_recognition as sr
from googletrans import Translator
import pyttsx3

def traducir_y_hablar(texto, idioma_destino):
    traductor = Translator()
    traduccion = traductor.translate(texto, dest=idioma_destino)
    texto_traducido = traduccion.text
    print(f" Traducción ({idioma_destino}): {texto_traducido}")

    engine = pyttsx3.init()
    engine.setProperty('rate', 170)
    engine.setProperty('volume', 1.0)
    engine.say(texto_traducido)
    engine.runAndWait()

def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print(" Ajustando al ruido ambiental... espera un momento.")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
    print("Micrófono listo.")

    idioma_destino = input(" Ingresa el código del idioma destino (ej. en, fr, de, ja, it): ").strip()

    print("\nHabla algo (di 'salir' para terminar):")
    while True:
        try:
            with mic as source:
                print("\n Escuchando...")
                audio = recognizer.listen(source, phrase_time_limit=5)

            texto = recognizer.recognize_google(audio, language="es-MX")
            print(f" Original: {texto}")

            if "salir" in texto.lower():
                print("Saliendo del programa...")
                break

            traducir_y_hablar(texto, idioma_destino)

        except sr.UnknownValueError:
            print("No entendí lo que dijiste.")
        except sr.RequestError:
            print("⚠️ Error con el servicio de reconocimiento (sin conexión o límite alcanzado).")
        except KeyboardInterrupt:
            print("\n Programa interrumpido manualmente.")
            break

if __name__ == "__main__":
    main()
