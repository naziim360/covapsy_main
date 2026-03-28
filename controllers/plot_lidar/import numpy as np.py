import numpy as np
import math
import matplotlib.pyplot as plt

num_sectors = 5
distances =np.random.rand(360) * 20  # Simulated LiDAR data (360° with random distances)

print("distances shape:", distances.shape)
sector=distances[80:281]
print("type of sector:", sector.shape)
# Convert to mm + filter
sector_mm = np.where((sector > 0) & (sector < 20),
                    sector * 1000.0,
                    20*1000)
print("sector_mm:", sector_mm)

#zones extractions
step = 201 // num_sectors
descrete_sectors= np.asarray(  [sector_mm[0                  :  step-1].min(),
                                sector_mm[step             :  2*step-1].min(),
                                sector_mm[2*step           :  3*step-1].min(),
                                sector_mm[3*step           :  4*step-1].min(),
                                sector_mm[4*step           :  201].min(),],
                                dtype=np.float64)
print("descrete_sectors:", descrete_sectors)
angle_rad =[math.radians(-100 + i * step) for i in range(num_sectors)]
print("angle_rad:", angle_rad)
xs = []
ys = []
i=0
for r in descrete_sectors:
    if r <= 0.0 or math.isinf(r):
        continue
    

    x = r * math.cos(angle_rad[i])
    y = r * math.sin(angle_rad[i])

    xs.append(x)
    ys.append(y)
    i += 1

    
plt.figure(figsize=(6, 6))
plt.scatter(xs, ys, s=5)
plt.plot(0, 0, "ro")  # position du LiDAR

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("LiDAR scan (vue du dessus)")
plt.axis("equal")
plt.grid(True)
plt.show()