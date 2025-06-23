def compute_cwater(T, A):
    # KH based on temperature (T in Celsius)
    KH = (
        76.81167
        - 3.348169 * T
        + 0.1198361 * T**2
        - 0.0031048148 * T**3
        + 0.00003566666668 * T**4
    )

    # Cwater based on your full original expression
    Cwater = (
        (44.01 / 22.4)
        * A
        * 273 / (273 + T)
        * (101325 / 100100)
        / (1000 * 1000)
        / 44.01
        * KH * KH
        * 44
    )

    return Cwater


# Example usage
T = 30       # Temperature in °C
A = 5000# CO2 air concentration in ppm

cwater_value = compute_cwater(T, A)
print(f"Cwater at T={T}°C and A={A} ppm: {cwater_value:.4f} mg/L")

