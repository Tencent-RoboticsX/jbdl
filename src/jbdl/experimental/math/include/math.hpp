#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif


int add(int i, int j);
int subtract(int i, int j);

template <typename T>
inline void sincos(const T& x, T* sx, T* cx) {
  *sx = sin(x);
  *cx = cos(x);
}

template <typename T>
void compute_eccentric_anomaly(const T& mean_anom, const T& ecc, T* sin_ecc_anom, T* cos_ecc_anom) {
  const T tol = 1e-12;
  T g, E = (mean_anom < M_PI) ? mean_anom + 0.85 * ecc : mean_anom - 0.85 * ecc;
  for (int i = 0; i < 20; ++i) {
    sincos(E, sin_ecc_anom, cos_ecc_anom);
    g = E - ecc * (*sin_ecc_anom) - mean_anom;
    if (fabs(g) <= tol) return;
    E -= g / (1 - ecc * (*cos_ecc_anom));
  }
}


