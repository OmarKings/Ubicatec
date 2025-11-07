import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("⚠️ No se pudo abrir la cámara.")
        return

    print("Presiona 'q' para salir.")

    screen_w = 1280
    screen_h = 720

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer frame.")
            break

        h, w, _ = frame.shape

        crop_size = int(min(h, w) * 0.8)
        start_x = w // 2 - crop_size // 2
        start_y = h // 2 - crop_size // 2
        end_x = start_x + crop_size
        end_y = start_y + crop_size
        cropped = frame[start_y:end_y, start_x:end_x]
        ratio = min(screen_w / crop_size, (screen_h - 100) / crop_size)
        new_w = int(crop_size * ratio)
        new_h = int(crop_size * ratio)
        resized = cv2.resize(cropped, (new_w, new_h))

        white_bar = np.ones((100, new_w, 3), dtype=np.uint8) * 255

        combined = np.vstack((resized, white_bar))

        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

        y_offset = (screen_h - combined.shape[0]) // 2
        x_offset = (screen_w - combined.shape[1]) // 2

        canvas[y_offset:y_offset + combined.shape[0],
               x_offset:x_offset + combined.shape[1]] = combined

        cv2.imshow("Cámara centrada", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
