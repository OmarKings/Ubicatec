import sounddevice as sd
import queue, json
from vosk import Model, KaldiRecognizer

def main():
    q = queue.Queue()
    model = Model(lang="es")
    rec = KaldiRecognizer(model, 16000)

    def callback(indata, frames, time, status):
        q.put(bytes(indata))

    print(" Hablando... (Para terminar di 'salir')")

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                texto = result.get("text", "").lower()
                if texto:
                    print( texto)
                    if "salir" in texto:
                        print("Saliendo del programa.")
                        break

if __name__ == "__main__":
    main()
