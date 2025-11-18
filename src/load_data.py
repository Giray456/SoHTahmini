import os
import glob
import numpy as np
from scipy.io import loadmat

# Proje köküne göre data klasörüne giden yol
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "nasa_dataset")


def load_single_battery(mat_path: str):
    """
    Tek bir .mat dosyasından (örn. B0005.mat) DEŞARJ (discharge) çevrimlerini okur.
    Her discharge için:
        - voltage: Voltage_measured
        - current: Current_measured
        - temperature: Temperature_measured
        - capacity: Capacity (scalar)
    döndürür.
    """
    # MATLAB struct'larını daha okunabilir yapmak için:
    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    # Dosya adı: B0005.mat -> key: "B0005"
    key = os.path.splitext(os.path.basename(mat_path))[0]
    battery = mat[key]

    cycles_out = []

    # battery.cycle bazen tek struct, bazen struct array olabiliyor
    cycles = np.atleast_1d(battery.cycle)

    for cycle in cycles:
        # cycle.type bir MATLAB char array -> string'e çevir
        c_type = str(cycle.type).strip().lower()

        # Sadece discharge çevrimlerini al
        if c_type != "discharge":
            continue

        data = cycle.data  # MATLAB struct -> attribute erişimi

        # Bütün verileri numpy array'e çevirip düzleştiriyoruz
        voltage = np.asarray(data.Voltage_measured).flatten()
        current = np.asarray(data.Current_measured).flatten()
        temperature = np.asarray(data.Temperature_measured).flatten()
        capacity = float(data.Capacity)  # scalar

        cycles_out.append(
            {
                "voltage": voltage,
                "current": current,
                "temperature": temperature,
                "capacity": capacity,
            }
        )

    return cycles_out


def load_all_batteries():
    """
    data/nasa_dataset içindeki tüm Bxxxx.mat dosyalarını okuyup
    discharge çevrimlerini tek listede birleştirir.
    """
    pattern = os.path.join(DATA_DIR, "B*.mat")
    files = sorted(glob.glob(pattern))

    all_cycles = []

    for f in files:
        print(f"[INFO] Loading {os.path.basename(f)}")
        cycles = load_single_battery(f)
        all_cycles.extend(cycles)

    return all_cycles
