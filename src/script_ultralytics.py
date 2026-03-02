from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def interpolar_detecciones(df_detecciones, total_frames=None, extrap_frames=5):
    df_detecciones['cx'] = (df_detecciones['x1'] + df_detecciones['x2']) / 2
    df_detecciones['cy'] = (df_detecciones['y1'] + df_detecciones['y2']) / 2

    start = df_detecciones['frame'].min()
    end = total_frames if total_frames else df_detecciones['frame'].max()
    all_frames = pd.DataFrame({'frame': range(start, end + 1)})

    df_full = pd.merge(all_frames, df_detecciones[['frame', 'cx', 'cy']], on='frame', how='left')
    df_full['cx'] = df_full['cx'].interpolate().bfill()
    df_full['cy'] = df_full['cy'].interpolate().bfill()

    df_full['vx'] = df_full['cx'].diff().fillna(0)
    df_full['vy'] = df_full['cy'].diff().fillna(0)
    df_full['speed'] = np.sqrt(df_full['vx']**2 + df_full['vy']**2)
    df_full['angle'] = np.degrees(np.arctan2(df_full['vy'], df_full['vx']))

    # EXTRAPOLACIÓN HACIA ATRÁS
    extrap_rows = []
    first_row = df_full.iloc[0]
    cx, cy = first_row['cx'], first_row['cy']
    vx, vy = first_row['vx'], first_row['vy']
    for i in range(1, extrap_frames + 1):
        frame_id = int(first_row['frame']) - i
        if frame_id < 1:
            break
        new_cx = cx - vx * i
        new_cy = cy - vy * i
        extrap_rows.append({
            'frame': frame_id,
            'cx': new_cx,
            'cy': new_cy,
            'vx': vx,
            'vy': vy,
            'speed': np.sqrt(vx**2 + vy**2),
            'angle': np.degrees(np.arctan2(vy, vx)),
            'extrapolated': True
        })

    df_full['extrapolated'] = False
    df_extrap = pd.DataFrame(extrap_rows)
    df_resultado = pd.concat([df_extrap, df_full], ignore_index=True).sort_values(by='frame')
    df_resultado.reset_index(drop=True, inplace=True)

    return df_resultado

# Rutas
model_path = '../models/best.pt'
video_path = '../videos/federer_murray_trim.mp4'
output_video = '../outputs/output_ultralytics.mp4'
output_csv = '../outputs/detecciones_ultralytics.csv'
output_csv_interpolado = '../outputs/detecciones_interpoladas.csv'

# Cargar modelo
model = YOLO(model_path)

# Abrir video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frames_raw = []
detecciones = []

# Primera pasada: detección
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    frames_raw.append(frame.copy())

    results = model(frame, verbose=False)
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detecciones.append({
                    'frame': frame_count,
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'confidence': round(conf, 4)
                })

cap.release()
df_detecciones = pd.DataFrame(detecciones)
df_detecciones.to_csv(output_csv, index=False)

if df_detecciones.empty:
    print("⚠️ No se encontraron detecciones.")
    exit()

# Interpolación + extrapolación
df_interpolado = interpolar_detecciones(df_detecciones, total_frames=total_frames, extrap_frames=5)
df_interpolado.to_csv(output_csv_interpolado, index=False)

# Segunda pasada: render
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
print("\n🎬 Renderizando video con interpolación y extrapolación...\n")

for i, frame in enumerate(frames_raw):
    frame_id = i + 1
    log_line = f"[Frame {frame_id}/{total_frames}] "

    # Mostrar número de frame
    cv2.putText(frame, f"Frame: {frame_id}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    row = df_interpolado[df_interpolado['frame'] == frame_id]
    if not row.empty:
        cx, cy = int(row['cx'].values[0]), int(row['cy'].values[0])
        vx, vy = row['vx'].values[0], row['vy'].values[0]
        fx, fy = int(cx + vx * 2), int(cy + vy * 2)
        extrap = row['extrapolated'].values[0]

        # Punto extrapolado en rojo
        if extrap:
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            log_line += "• Extrapolado"
        else:
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            log_line += "• Interpolado"
        cv2.arrowedLine(frame, (cx, cy), (fx, fy), (255, 255, 255), 2, tipLength=0.3)

    # Detecciones reales
    detecciones_frame = df_detecciones[df_detecciones['frame'] == frame_id]
    if not detecciones_frame.empty:
        for _, d in detecciones_frame.iterrows():
            x1, y1, x2, y2, conf = int(d['x1']), int(d['y1']), int(d['x2']), int(d['y2']), d['confidence']
            label = f"Ball {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            log_line += f" ✔️ Detected ({conf:.2f})"

    print(log_line)
    out.write(frame)
    cv2.imshow("Ball Detection + Interpolation", frame)
    key = cv2.waitKey(int(1000 / fps))
    if key == 27:
        break

out.release()
cv2.destroyAllWindows()

print(f"\n✅ Video guardado: {output_video}")
print(f"📄 CSV detecciones: {output_csv}")
print(f"📈 CSV interpolado: {output_csv_interpolado}")

# Gráficos de análisis
plt.figure(figsize=(10, 4))
plt.plot(df_interpolado['frame'], df_interpolado['angle'], marker='o', linestyle='-', color='tab:blue')
plt.title("Ángulo de la pelota por frame")
plt.xlabel("Frame")
plt.ylabel("Ángulo (grados)")
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_angulo_por_frame.png")
plt.close()

angles_rad = np.radians(df_interpolado['angle'])
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
ax.set_theta_zero_location('E')
ax.set_theta_direction(-1)
bins = 36
counts, bin_edges = np.histogram(angles_rad, bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.bar(bin_centers, counts, width=2 * np.pi / bins, bottom=0.0, color='skyblue', edgecolor='black')
ax.set_title("Distribución de ángulos de movimiento", va='bottom')
plt.tight_layout()
plt.savefig("rosa_direcciones.png")
plt.close()

print("📊 Gráficos generados: rosa_direcciones.png, grafico_angulo_por_frame.png")
